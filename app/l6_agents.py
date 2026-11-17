"""L6 Agentic layer: AI agents (L6.1) and the orchestrator (L6.2).

Agents decide which tools to use and how to read the results. They act only
through ctx.tool(), which routes every call through the L7 action guard. The
orchestrator compiles a plan into ordered nodes, runs them one at a time,
persists state after each node so runs can pause for humans and resume, and
supports steering (edit parameters and recompute downstream).
"""
import json
import logging
import re
import threading
import time
import traceback
import uuid

import numpy as np
import pandas as pd

from . import auth
from . import config
from . import l1_data as db
from . import l2_policy as P
from . import l3_models as M
from . import l5_tools as T
from . import l7_control as C
from . import datasets as DS
from . import dataset_report as RP
from . import dataset_dashboard as DD
from . import llm

money = T.money
_log = logging.getLogger("analytixa.agents")

SCOPES = {
    "planner": {"C10", "M16", "C12", "M18"},
    "retriever": {"D1", "D2", "D3", "D4", "D5", "D8", "D9", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19"},
    "credit_agent": {"M10", "X2", "X3", "X10", "X12", "C3", "N7"},
    "bi_agent": {"M16", "C10", "D13", "X9", "C13", "C14"},
    "churn_agent": {"M8", "C4", "C7", "X10"},
    "nba_agent": {"N5", "C11", "C5", "N4"},
    "transaction_monitor": {"M1", "M4", "M5", "N1", "N2"},
    "case_investigator": {"M17", "N3", "D3"},
    "cash_agent": {"M14", "C5", "N9"},
    "verifier": {"C3", "C7", "X11", "D13", "D17"},
    "narrator": {"M17", "N12"},
}

AGENT_LABELS = {
    "planner": "Planner", "retriever": "Retriever", "credit_agent": "Alternative-data credit agent",
    "bi_agent": "Conversational BI agent", "churn_agent": "Churn sentinel", "nba_agent": "Next-best-action agent",
    "transaction_monitor": "Transaction monitoring and AML typology agent", "case_investigator": "Case investigation agent",
    "cash_agent": "Cash forecasting agent", "verifier": "Verifier", "narrator": "Narrator",
}

GUARD = C.Guard(T.REG, SCOPES)


class StopRun(Exception):
    pass


class Ctx:
    def __init__(self, row):
        self.run_id = row["run_id"]
        self.use_case = row["use_case"]
        self.prompt = row["prompt"]
        self.role = row["role"]
        self.mode = row["mode"]
        self.created_by = row.get("created_by")
        self.state = json.loads(row["state"] or "{}")
        self.state.setdefault("out", {})
        self.state.setdefault("params", {})
        self.state.setdefault("decisions", {})
        self.plan = json.loads(row["plan"] or "[]")
        self.node_id = None
        self.calls, self.blocked = [], []
        self.policy_versions = self.state.get("policy_versions", {})

    def tool(self, agent, tool_id, **args):
        return GUARD.call(self, agent, tool_id, **args)

    def out(self, key, value):
        self.state["out"][key] = value

    def get(self, key, default=None):
        return self.state["out"].get(key, default)

    @property
    def params(self):
        return self.state["params"]


def say(ctx, speaker, text, data=None):
    db.execute("INSERT INTO dialogue (ts, run_id, speaker, text, data) VALUES (?,?,?,?,?)",
               (db.now(), ctx.run_id, speaker, text, json.dumps(data, default=str) if data else None))


def narrate(ctx, instruction, facts, fallback):
    """Narrator: LLM text from verified facts, or the deterministic template."""
    cache = ctx.state.setdefault("llm_cache", {})
    k = instruction[:60]
    if k not in cache:
        cache[k] = ctx.tool("narrator", "M17", instruction=instruction, facts=facts) or ""
    text = cache[k]
    return (text, "llm") if text else (fallback, "template")


# ═══════════════════════════ USE CASE 1: CREDIT ═══════════════════════════

def cr_intake(ctx):
    m = re.search(r"APP-\d{4}", ctx.prompt.upper())
    app_id = ctx.params.get("app_id") or (m.group(0) if m else None)
    if not app_id:
        app_id = ctx.tool("retriever", "D4")["application"]["app_id"]
    ctx.params["app_id"] = app_id
    return f"Application {app_id} selected. Policy: loan_policy v{ctx.policy_versions.get('loan_policy')}."


def cr_retrieve(ctx):
    app_id = ctx.params["app_id"]
    rec = ctx.tool("retriever", "D4", app_id=app_id)
    if not rec:
        raise C.Blocked(f"Application {app_id} not found.")
    cons = ctx.tool("retriever", "D9", customer_id=rec["application"]["customer_id"], purpose="alternative_data")
    if cons["consented"]:
        ctx.tool("retriever", "D8", app_id=app_id)
    else:
        # without consent the alternative data is not used: the model sees neutral values
        for k in ("utility_ontime_ratio", "airtime_monthly_avg", "mobile_money_inflow_avg"):
            rec["application"][k] = None
    ctx.out("record", rec)
    ctx.out("consent", cons)
    c = rec["customer"]
    a = rec["application"]
    return (f"{c['name']} ({c['customer_id']}), {c['segment']}, {c['branch']} branch. Requests {money(a['amount'])} "
            f"over {a['tenor_months']} months for {a['purpose'].lower()}. Alternative-data consent: "
            f"{'yes' if cons['consented'] else 'no'}.")


def apply_overrides(ctx, rec):
    rec = json.loads(json.dumps(rec))
    for k, v in ctx.params.get("corrections", {}).items():
        rec["application"][k] = v
    return rec


def cr_quality(ctx):
    rec = ctx.get("record")
    dq = ctx.tool("retriever", "D14", target="application", ids=ctx.params["app_id"])
    ctx.out("quality", dq)
    blocking = [i for i in dq["issues"] if i["severity"] == "blocking"]
    if blocking and not ctx.params.get("corrections"):
        d = C.require(ctx, "data_review", "Data problems found in this application", {
            "issues": dq["issues"], "quality_score": dq["score"],
            "editable": {i["field"]: rec["application"].get(i["field"]) for i in blocking},
            "instruction": "Enter corrected values from the source documents and approve, or reject to return the application."})
        if d.get("edits"):
            ctx.params["corrections"] = {k: float(v) for k, v in d["edits"].items() if v not in (None, "")}
            dq = {**dq, "corrected": ctx.params["corrections"], "score": 0.9}
            ctx.out("quality", dq)
    return f"Quality score {dq['score']:.2f}. " + ("; ".join(i["field"] + " " + i["issue"] for i in dq["issues"]) or "No issues.")


def model_inputs(x):
    m = {k: x[k] for k in ["monthly_income", "dti", "tenure_months", "utility_ontime_ratio", "airtime_monthly_avg",
                           "mobile_money_inflow_avg", "bureau_score", "prior_default", "informal_income"]}
    m["dti"] = m["dti"] if m["dti"] is not None else 2.0
    m["monthly_income"] = m["monthly_income"] or 1000
    # neutral values (training medians) when alternative data is absent or not consented
    if m["utility_ontime_ratio"] is None:
        m["utility_ontime_ratio"] = 0.77
    if m["airtime_monthly_avg"] is None:
        m["airtime_monthly_avg"] = 6000.0
    if m["mobile_money_inflow_avg"] is None:
        m["mobile_money_inflow_avg"] = 0.65 * m["monthly_income"]
    return m


def score_application(ctx, rec, **ov):
    x = T.credit_inputs(rec, ov.get("amount"), ov.get("tenor"), ov.get("income"), ov.get("extra_installment", 0.0))
    return ctx.tool("credit_agent", "M10", inputs=model_inputs(x))


def cr_score(ctx):
    rec = apply_overrides(ctx, ctx.get("record"))
    sc = score_application(ctx, rec)
    rc = ctx.tool("credit_agent", "X2", attr=sc["attr"])
    ctx.out("score", {"pd": sc["pd"], "attr": sc["attr"], "model": sc["model"], "auc": sc["auc"]})
    ctx.out("reasons", rc)
    return f"Probability of default {sc['pd']:.1%} ({sc['model']}, holdout AUC {sc['auc']})."


def cr_policy(ctx):
    rec = apply_overrides(ctx, ctx.get("record"))
    pd_ = ctx.get("score")["pd"]
    res = ctx.tool("credit_agent", "C3", rec=rec, pd_value=pd_)
    ctx.out("rules", res)
    ctx.state["credit"] = {"pd": pd_, "hard_fail": any(r["kind"] == "hard" and not r["pass"] for r in res["rules"]),
                           "original_rec": res["recommendation"]}
    fails = [f"{r['id']} {r['name']}" for r in res["rules"] if not r["pass"]]
    return f"Recommendation: {res['recommendation'].upper()}. " + (f"Failed: {', '.join(fails)}." if fails else "All rules pass.")


def _rules_fn(ctx):
    def fn(rec, amount=None, tenor=None, collateral=None):
        sc = score_application(ctx, rec, amount=amount, tenor=tenor)
        return T.evaluate_rules(rec, sc["pd"], amount=amount, tenor=tenor, collateral=collateral)
    return fn


def batch_fairness(ctx):
    if not ctx.state.get("fairness_cache"):
        ctx.state["fairness_cache"] = ctx.tool("credit_agent", "X12")
    return ctx.state["fairness_cache"]


def cr_recourse(ctx):
    rec = apply_overrides(ctx, ctx.get("record"))
    rules = ctx.get("rules")
    cf = []
    if rules["recommendation"] != "approve" and rec["application"]["declared_income"]:
        cf = ctx.tool("credit_agent", "X3", rec=rec, rules_fn=_rules_fn(ctx))
    fair = batch_fairness(ctx)
    ctx.out("counterfactuals", cf)
    ctx.out("fairness", fair)
    g = fair.get("gender", {})
    return (f"{len(cf)} recourse option(s). Portfolio approval-rate ratio by gender {g.get('ratio', 1):.2f} "
            f"(threshold {fair['threshold']}): {'pass' if fair['pass'] else 'FAIL'}.")


def cr_verify(ctx):
    rec = apply_overrides(ctx, ctx.get("record"))
    rules = ctx.get("rules")
    x = rules["inputs"]
    pol = P.body("loan_policy")
    a = rec["application"]
    # independent recomputation of the installment and DTI
    inst = a["amount"] * (1 + pol["interest_rate_annual"] * a["tenor_months"] / 12) / a["tenor_months"]
    exist = sum(l["monthly_installment"] for l in rec["existing_loans"] if l["status"] != "closed")
    dti = (inst + exist) / a["declared_income"] if a["declared_income"] else None
    agree = dti is not None and x["dti"] is not None and abs(dti - x["dti"]) < 1e-3
    recheck = ctx.tool("verifier", "C3", rec=rec, pd_value=ctx.get("score")["pd"])
    agree = agree and recheck["recommendation"] == rules["recommendation"]
    xq = ctx.tool("verifier", "X11", attr=ctx.get("score")["attr"], inputs=model_inputs(x))
    stab = xq["stability"]
    pd_ = ctx.get("score")["pd"]
    p9 = next(r for r in pol["rules"] if r["id"] == "R9")["params"]
    margin = min(abs(pd_ - p9["approve_below"]), abs(pd_ - p9["decline_above"])) / 0.1
    conf = C.confidence({"verifier_agreement": 1.0 if agree else 0.0, "data_quality": ctx.get("quality")["score"],
                         "explanation_stability": stab, "decision_margin": min(1.0, margin),
                         "faithfulness": 1.0 if xq["faithful"] else 0.0})
    ctx.state["confidence"] = conf["score"]
    ctx.out("verification", {"independent_dti": dti, "agree": agree, "explanation": xq, "confidence": conf})
    ctx.out("escalated", conf["score"] < C.floor("high"))
    return (f"Verifier {'agrees' if agree else 'DISAGREES'}. Explanation faithful: {xq['faithful']}, stability {stab:.2f}. "
            f"Confidence {conf['score']:.2f} (weakest signal: {conf['weakest']}).")


def credit_evidence(ctx):
    rec = apply_overrides(ctx, ctx.get("record"))
    a = rec["application"]
    rules = ctx.get("rules")
    return {
        "application": {k: a[k] for k in ["app_id", "customer_id", "amount", "tenor_months", "purpose", "collateral_value",
                                          "declared_income", "bureau_score", "utility_ontime_ratio"]},
        "customer": {k: rec["customer"].get(k) for k in ["name", "segment", "branch", "age", "tenure_months", "employer_type"]},
        "recommendation": rules["recommendation"], "rules": rules["rules"], "inputs": rules["inputs"],
        "pd": ctx.get("score")["pd"], "reasons": ctx.get("reasons"), "counterfactuals": ctx.get("counterfactuals"),
        "fairness": ctx.get("fairness"), "verification": ctx.get("verification"),
        "officer_limit": P.body("loan_policy")["officer_delegated_limit"],
        "dry_run": {"system": "Loan origination (simulated)", "write": rules["recommendation"], "app_id": a["app_id"]},
        "asserted_facts": ctx.params.get("asserted", {}), "escalated": ctx.get("escalated"),
    }


def cr_decide(ctx):
    rules = ctx.get("rules")
    app_id = ctx.params["app_id"]
    rec_ = rules["recommendation"]
    env = P.body("loan_policy")["auto_decision_envelope"]
    in_env = (env["enabled"] and rec_ == "approve" and rules["inputs"]["amount"] <= env["max_amount"]
              and ctx.get("score")["pd"] <= env["max_pd"] and not ctx.get("escalated"))
    if in_env and ctx.mode == "haara":
        ctx.out("final", {"decision": "approve", "by": "envelope", "comment": "Inside the auto-decision envelope (A3)."})
        C.notify(ctx, "loan_officer", f"{app_id} approved inside envelope", "Clean approval inside the auto-decision envelope.")
        return "Clean approval inside the auto-decision envelope. Loan officer notified (A3)."
    title = "Credit decision" + (" (escalated: low confidence)" if ctx.get("escalated") else "")
    d = C.require(ctx, "credit_decision", title, credit_evidence(ctx), key=f"credit_decision:{app_id}")
    fd = (d.get("edits") or {}).get("final_decision")
    decision = fd or rec_
    final = {"decision": decision, "by": d.get("by") or "autonomous", "comment": d.get("comment")}
    amount = rules["inputs"]["amount"]
    flipped = ctx.params.get("asserted") and ctx.state["credit"].get("original_rec") != "approve"
    if ctx.mode != "autonomous" and decision == "approve" and (ctx.state["credit"]["hard_fail"] or rec_ != "approve" or flipped
                                                              or amount > P.body("loan_policy")["officer_delegated_limit"]):
        why = ("Approval overrides a hard policy rule" if ctx.state["credit"]["hard_fail"] else
               "Approval overrides the recommendation" if rec_ != "approve" else
               "Approval rests on facts asserted by the officer that changed the recommendation" if flipped else
               "Amount exceeds the officer's delegated limit")
        e = C.require(ctx, "credit_exception", f"Credit exception: {why}",
                      {**credit_evidence(ctx), "officer_decision": final, "exception_reason": why},
                      key=f"credit_exception:{app_id}", role=C.route("credit_exception", ctx.role))
        final["exception_approved_by"] = e.get("by")
        ctx.state["credit"]["exception_approved"] = True
    ctx.out("final", final)
    return f"Final decision: {decision.upper()} by {final['by']}."


def cr_write(ctx):
    final = ctx.get("final")
    rules = ctx.get("rules")
    app_id = ctx.params["app_id"]
    if final["by"] not in ("envelope", "autonomous"):
        # the officer authorised exactly this write at the decision gate, after seeing its dry run
        ctx.state["decisions"][f"action:N7:{app_id}:{final['decision']}"] = {"decision": "approve", "by": final["by"]}
        ctx.state["last_approver"] = final["by"]
    reasons = [r["reason"] for r in ctx.get("reasons")["adverse"]] + [f"{r['id']} {r['name']}" for r in rules["rules"] if not r["pass"]]
    ctx.tool("credit_agent", "N7", app_id=app_id, decision=final["decision"], amount=rules["inputs"]["amount"],
             reason_codes=reasons, decided_by=final["by"])
    return f"Decision written to the loan origination system with {len(reasons)} reason code(s)."


def cr_memo(ctx):
    ev = credit_evidence(ctx)
    final = ctx.get("final")
    failed = [f"{r['id']} {r['name']} ({r['value']} vs {r['threshold']})" for r in ev["rules"] if not r["pass"]]
    fallback = (f"Credit memo for {ev['application']['app_id']}. Applicant {ev['customer']['name']} requested "
                f"{money(ev['application']['amount'])} over {ev['application']['tenor_months']} months. "
                f"The agent recommended {ev['recommendation']}. Model PD {ev['pd']:.1%}. "
                + (f"Policy rules failed: {'; '.join(failed)}. " if failed else "All policy rules passed. ")
                + (f"Main risk factors: {', '.join(r['reason'].lower() for r in ev['reasons']['adverse'][:3])}. " if ev['reasons']['adverse'] else "")
                + (f"Recourse: {' '.join(c['text'] for c in ev['counterfactuals'])} " if ev["counterfactuals"] else "")
                + f"Final decision: {final['decision']} by {final['by']}.")
    text, src = narrate(ctx, "Write a credit memo of 5 to 7 sentences for the file. Cite rule IDs.",
                        {"evidence": ev, "final": final}, fallback)
    ctx.state["result"] = {"type": "credit", "memo": text, "memo_source": src, "evidence": ev, "final": final}
    return "Credit memo drafted."


# credit challenge dialogue (officer <-> agent), used while the decision gate is open

def parse_challenge(text):
    out = {}
    t = text.lower().replace(",", "")
    nums = [float(n) * (1e6 if s in ("m", "million") else 1e3 if s in ("k", "thousand") else 1)
            for n, s in re.findall(r"(\d+(?:\.\d+)?)\s*(m|million|k|thousand)?\b", t)]
    big = [n for n in nums if n >= 1000]
    if re.search(r"salary|income|earn|pay rise|increase", t) and big:
        out["income"] = max(big)
    if re.search(r"collateral|pledge|security|guarant", t) and big:
        out["collateral"] = max(big)
    m = re.search(r"(\d+)\s*months?", t)
    if m and re.search(r"tenor|months|duration|longer|spread", t):
        out["tenor"] = int(m.group(1))
    if re.search(r"amount|reduce|lower|instead|only|borrow", t) and big and "income" not in out and "collateral" not in out:
        out["amount"] = min(big)
    if re.search(r"paid off|cleared|settled|repaid", t):
        out["clear_existing"] = True
    return out


def respond_to_challenge(ctx, text, role, who=None):
    rec = apply_overrides(ctx, ctx.get("record"))
    parsed = None
    if llm.available():
        parsed = llm.chat_json(
            "Extract what-if facts from a loan officer's challenge. Keys allowed: income (monthly NGN), amount (NGN), "
            "tenor (months), collateral (NGN value), clear_existing (true if existing loans were repaid). "
            "Omit keys that are not stated. If the officer asks a question without new facts, return {}.", text)
        if parsed is not None:
            parsed = {k: v for k, v in parsed.items() if k in ("income", "amount", "tenor", "collateral", "clear_existing") and v}
    if parsed is None:
        parsed = parse_challenge(text)
    say(ctx, _actor(role, who), text)
    before = ctx.get("rules")
    if not parsed:
        failed = [r for r in before["rules"] if not r["pass"]]
        reply = ("I found no new fact to test in that. " +
                 (f"The recommendation rests on {', '.join(r['id'] + ' ' + r['name'] + ' (' + r['value'] + ')' for r in failed)}. "
                  if failed else "Every policy rule passes. ") +
                 "Give me a changed fact, such as a new salary, a smaller amount, a longer tenor, collateral or a repaid loan, and I will re-run the checks.")
        say(ctx, "credit_agent", reply)
        return {"reply": reply, "changed": False}
    extra = 0.0
    if parsed.get("clear_existing"):
        extra = -sum(l["monthly_installment"] for l in rec["existing_loans"] if l["status"] != "closed")
    ov = {"income": float(parsed["income"]) if "income" in parsed else None,
          "amount": float(parsed["amount"]) if "amount" in parsed else None,
          "tenor": int(parsed["tenor"]) if "tenor" in parsed else None}
    sc = score_application(ctx, rec, extra_installment=extra, **ov)
    after = ctx.tool("credit_agent", "C3", rec=rec, pd_value=sc["pd"], collateral=parsed.get("collateral"),
                     extra_installment=extra, **ov)
    changes = []
    for b, a in zip(before["rules"], after["rules"]):
        if b["pass"] != a["pass"] or b["value"] != a["value"]:
            changes.append(f"{a['id']} {a['name']}: {b['value']} to {a['value']} ({'pass' if a['pass'] else 'fail'})")
    facts = ", ".join((f"{k} {money(v)}" if k in ("income", "amount", "collateral") else
                       ("existing loans repaid" if k == "clear_existing" else f"{k} {v}")) for k, v in parsed.items())
    if after["recommendation"] != before["recommendation"]:
        reply = (f"With {facts}, I re-ran every rule. {'; '.join(changes)}. PD moves from {ctx.get('score')['pd']:.1%} to {sc['pd']:.1%}. "
                 f"I revise my recommendation from {before['recommendation']} to {after['recommendation']}. "
                 "These facts are officer-asserted, so attach evidence (payslip, statement or valuation) to the file before approval.")
    else:
        still = [r for r in after["rules"] if not r["pass"]]
        reply = (f"With {facts}, I re-ran every rule. " + ("; ".join(changes) + ". " if changes else "No rule changed. ") +
                 f"My recommendation stays {after['recommendation']}" +
                 (f" because {', '.join(r['id'] + ' ' + r['name'] + ' still fails at ' + r['value'] for r in still)}." if still else ".") +
                 (" An approval would need a credit exception from the credit manager." if any(r["kind"] == "hard" for r in still) else ""))
    say(ctx, "credit_agent", reply, {"what_if": parsed, "pd": sc["pd"], "recommendation": after["recommendation"]})
    if after["recommendation"] != before["recommendation"]:
        ctx.params["asserted"] = {**ctx.params.get("asserted", {}), **parsed}
        ctx.out("rules", after)
        ctx.out("score", {**ctx.get("score"), "pd": sc["pd"], "attr": sc["attr"]})
        ctx.out("reasons", ctx.tool("credit_agent", "X2", attr=sc["attr"]))
        ctx.state["credit"] = {**ctx.state.get("credit", {}), "pd": sc["pd"],
                               "hard_fail": any(r["kind"] == "hard" and not r["pass"] for r in after["rules"])}
    return {"reply": reply, "changed": after["recommendation"] != before["recommendation"], "what_if": parsed}


# ═══════════════════════════ USE CASE 2: CONVERSATIONAL BI ═══════════════════════════

def bi_intent(ctx):
    q = ctx.params.get("question") or ctx.prompt
    intent = ctx.params.get("intent")
    if not intent:
        intent = ctx.tool("planner", "M16", question=q) if llm.available() else None
        if not intent or not intent.get("metric"):
            rules = ctx.tool("planner", "C10", question=q)
            intent = rules if (rules.get("metric") or not intent) else intent
        intent.setdefault("dimensions", [])
        intent.setdefault("filters", {})
        intent["filters"] = {k: v for k, v in (intent.get("filters") or {}).items() if v}
        ctx.params["intent"] = intent
    if not intent.get("metric"):
        ctx.state["result"] = {"type": "clarify", "question": intent.get("clarify")}
        say(ctx, "bi_agent", intent.get("clarify"))
        raise StopRun("clarify")
    plain = T.describe_intent(intent)
    ctx.out("interpretation", {"text": plain, "intent": intent})
    # HITL on meaning: the person who asked confirms the interpretation before the query runs
    # when the policy says so, or when the wording is complex and a rule parser read it.
    check = P.body("autonomy").get("bi_interpretation_check", "when_unsure")
    unsure = intent.get("source") == "rules" and re.search(T.COMPLEX_WORDS, q.lower())
    if intent.get("source") != "corrected" and (check == "always" or (check == "when_unsure" and unsure)):
        C.require(ctx, "interpretation", "Check how your question was understood",
                  {"question": q, "interpretation": plain, "intent": intent,
                   "why": "The wording has a condition such as not, except or only, so the agent asks you to confirm."
                   if unsure else "Bank policy asks you to confirm every interpretation."}, level_="A1")
    return f"Understood as: {plain} ({intent.get('source', 'edited')})"


def bi_query(ctx):
    i = ctx.params["intent"]
    try:
        res = ctx.tool("retriever", "D13", metric=i["metric"], dimensions=i["dimensions"], filters=i["filters"],
                       period=i.get("period"))
    except (ValueError, KeyError) as e:
        ctx.state["result"] = {"type": "clarify", "question": str(e)}
        raise StopRun("clarify")
    ctx.out("bi", res)
    return f"{len(res['rows'])} row(s) returned from the governed semantic query."


def bi_verify(ctx):
    res = ctx.get("bi")
    i = ctx.params["intent"]
    checks = []
    vals = [r["value"] for r in res["rows"] if r["value"] is not None]
    if res["unit"] == "%":
        checks.append(("percent values within 0-100", all(0 <= v <= 100 for v in vals)))
    checks.append(("result is not empty", len(vals) > 0))
    md = P.body("semantic_layer")["metrics"][i["metric"]]
    if i["dimensions"] and md["sql"].startswith(("SUM", "COUNT(*)")):
        tot = ctx.tool("verifier", "D13", metric=i["metric"], dimensions=[], filters=i["filters"], period=i.get("period"))
        t = tot["rows"][0]["value"] or 0
        checks.append(("group values add up to the total", abs(sum(vals) - t) <= max(1, abs(t) * 1e-6)))
    tr = ctx.tool("bi_agent", "X9", result=res)
    ok = sum(c[1] for c in checks) / len(checks)
    conf = C.confidence({"verifier_checks": ok, "intent_source": 0.9 if i.get("source") == "llm" else 0.8})
    ctx.state["confidence"] = conf["score"]
    ctx.out("bi_checks", {"checks": [{"check": c, "pass": p} for c, p in checks], "confidence": conf, "transparency": tr})
    if conf["score"] < C.floor("low"):
        C.require(ctx, "release", "Low-confidence answer held at the release gate",
                  {"checks": checks, "rows": res["rows"][:20], "transparency": tr})
    return f"{sum(c[1] for c in checks)} of {len(checks)} verifier checks passed. Confidence {conf['score']:.2f}."


def bi_answer(ctx):
    res = ctx.get("bi")
    rows = res["rows"]
    fmt = (lambda v: f"{v:.1f}%") if res["unit"] == "%" else ((lambda v: money(v)) if res["unit"] == "NGN" else (lambda v: f"{v:,.0f}"))
    rows = [r for r in rows if r.get("value") is not None]
    if rows and len(rows) == 1 and len(rows[0]) == 1:
        fb = f"{res['label']}: {fmt(rows[0]['value'])}. Definition: {res['definition']}"
    elif rows:
        dims = [k for k in rows[0] if k != "value"]
        valued = [r for r in rows if r["value"] is not None] or rows
        top = max(valued, key=lambda r: r["value"] or 0)
        low = min(valued, key=lambda r: r["value"] or 0)
        fb = (f"{res['label']} by {', '.join(dims)}. Highest: {' '.join(str(top[d]) for d in dims)} at {fmt(top['value'])}. "
              f"Lowest: {' '.join(str(low[d]) for d in dims)} at {fmt(low['value'])}. Definition: {res['definition']}")
    else:
        fb = "No data matched the question."
    if res.get("note"):
        fb += " " + res["note"]
    text, src = narrate(ctx, "Answer the user's question in 2 to 4 sentences using the query result.",
                        {"question": ctx.prompt, "result": rows[:30], "metric": res["label"], "unit": res["unit"],
                         "definition": res["definition"], "note": res.get("note")}, fb)
    ctx.state["result"] = {"type": "bi", "answer": text, "answer_source": src, "result": res,
                           "checks": ctx.get("bi_checks"), "intent": ctx.params["intent"],
                           "interpretation": T.describe_intent(ctx.params["intent"])}
    return "Answer released with query transparency."


# ═══════════════════════════ USE CASE 3: CHURN AND RETENTION ═══════════════════════════

CITIES = ["Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan", "Enugu", "Kaduna"]


def ch_plan(ctx):
    q = ctx.prompt.lower()
    p = ctx.params
    p.setdefault("city", next((c for c in CITIES if c.lower() in q), None))
    p.setdefault("segment", "Retail" if "retail" in q else ("SME" if "sme" in q else None))
    p.setdefault("min_tenure", 6)
    pol = P.body("churn_policy")
    p.setdefault("threshold", pol["risk_threshold"])
    p.setdefault("launch", bool(re.search(r"\b(launch|send|contact|reach out)\b", q)))
    plan = {"cohort": f"Active {p['segment'] or 'retail and SME'} customers in {p['city'] or 'all cities'} with at least {p['min_tenure']} months tenure",
            "definition": pol["definition"], "horizon_days": pol["horizon_days"], "risk_threshold": p["threshold"],
            "steps": ["D16 cohort and D14 quality", "M8 churn model with attributions", "C4 drivers, C7 backtest, X10 fairness",
                      "N5 suppression, C11 conflicts, C5 budget allocation", "M17 report", "N4 outreach only if authorised"],
            "models": ["churn-v1"], "checkpoints": ["data review if issues", "escalation if low confidence",
                                                    "strategy approval", "report release", "outreach authorisation if launched"]}
    ctx.out("plan", plan)
    if P.body("autonomy")["plan_approval"].get("churn"):
        d = C.require(ctx, "plan", "Approve the churn analysis plan", {"plan": plan, "editable": {
            "city": p["city"] or "", "segment": p["segment"] or "", "min_tenure": p["min_tenure"], "threshold": p["threshold"]}})
        for k, v in (d.get("edits") or {}).items():
            if k in ("city", "segment"):
                p[k] = v or None
            elif k in ("min_tenure", "threshold") and v not in (None, ""):
                p[k] = float(v) if k == "threshold" else int(float(v))
    return f"Active {p['segment'] or 'retail and SME'} customers in {p['city'] or 'all cities'}, tenure at least {p['min_tenure']} months, threshold {p['threshold']:.0%}."


def _cohort(ctx):
    p = ctx.params
    return ctx.tool("retriever", "D16", city=p["city"], segment=p["segment"], min_tenure=p["min_tenure"])


def _branch_names(ctx):
    return {k: v["name"] for k, v in ctx.tool("retriever", "D5").items()}


def ch_cohort(ctx):
    fr = _cohort(ctx)
    dq = ctx.tool("retriever", "D14", target="cohort", ids=fr)
    ctx.out("cohort_quality", dq)
    if dq["issues"]:
        C.require(ctx, "data_review", "Cohort data issues", {"issues": dq["issues"], "rows": dq["rows"]})
    return f"{len(fr)} customers in the cohort. Quality score {dq['score']:.2f}."


def ch_score(ctx):
    fr = _cohort(ctx)
    sc = ctx.tool("churn_agent", "M8", cohort=fr)
    th = ctx.params["threshold"]
    ctx.state["churn_scores"] = {cid: {"p": round(float(p), 3), "drivers": d} for cid, p, d in zip(fr.index, sc["p"], sc["drivers"])}
    at = int((sc["p"] >= th).sum())
    ctx.out("churn_summary", {"cohort": len(fr), "at_risk": at, "threshold": th, "model": sc["model"], "auc": sc["auc"]})
    return f"{at} of {len(fr)} customers at or above {th:.0%} churn risk ({sc['model']}, holdout AUC {sc['auc']})."


def ch_analyse(ctx):
    fr = _cohort(ctx)
    s = ctx.state["churn_scores"]
    fr = fr.assign(p=[s[c]["p"] for c in fr.index])
    fr["at_risk"] = fr.p >= ctx.params["threshold"]
    drivers = ctx.tool("churn_agent", "C4", cohort=fr, flag_col="at_risk")
    bt = ctx.tool("churn_agent", "C7", model_name="churn")
    fair = ctx.tool("churn_agent", "X10", outcomes=fr.reset_index(), positive_col="at_risk")
    seg = fr.groupby("branch_id").agg(n=("p", "size"), at_risk=("at_risk", "sum"), mean_p=("p", "mean")).reset_index()
    seg["branch"] = seg.branch_id.map(_branch_names(ctx))
    flagged = []
    for r in seg.itertuples():
        conf = C.confidence({"backtest_auc": bt["auc_holdout"], "segment_size": min(1, r.n / 40),
                             "fairness": 1.0 if fair["pass"] else 0.5})
        if conf["score"] < C.floor("medium"):
            flagged.append({"branch": r.branch, "customers": int(r.n), "at_risk": int(r.at_risk), "confidence": conf})
    ctx.out("churn_analysis", {"drivers": drivers[:6], "backtest": bt, "fairness": fair,
                               "segments": seg.drop(columns="branch_id").round(3).to_dict("records"), "low_confidence": flagged})
    if flagged:
        d = C.require(ctx, "escalation", f"{len(flagged)} branch segment(s) below the confidence floor",
                      {"segments": flagged, "options": "Approve to keep them with a caveat, or list branches in exclude_branches to drop them.",
                       "editable": {"exclude_branches": ""}})
        ex = (d.get("edits") or {}).get("exclude_branches")
        if ex:
            ctx.params["exclude_branches"] = [x.strip() for x in str(ex).split(",") if x.strip()]
    sig = [d for d in drivers if d["p_value"] < .05]
    return (f"{len(sig)} significant driver(s): {', '.join(d['feature'] for d in sig[:4]) or 'none'}. "
            f"Backtest AUC {bt['auc_holdout']}. Fairness {'pass' if fair['pass'] else 'FAIL'}. {len(flagged)} low-confidence segment(s).")


def ch_strategy(ctx):
    pol = P.body("churn_policy")
    fr = _cohort(ctx)
    names = _branch_names(ctx)
    excl = set(ctx.params.get("exclude_branches", []))
    s = ctx.state["churn_scores"]
    at = [(cid, s[cid]) for cid in fr.index if s[cid]["p"] >= ctx.params["threshold"] and names[fr.loc[cid, "branch_id"]] not in excl]
    ids = [c for c, _ in at]
    sup = ctx.tool("nba_agent", "N5", customer_ids=ids)
    conflicts = ctx.tool("nba_agent", "C11", customer_ids=ids)
    removed = set(ctx.params.get("remove_offers", []))
    cands, suppressed = [], []
    for cid, sc in at:
        if sup.get(cid) and ctx.mode != "autonomous":
            suppressed.append({"customer_id": cid, "why": sup[cid]})
            continue
        row = fr.loc[cid]
        topics = set(sc["drivers"]) | ({row.top_complaint} if isinstance(row.top_complaint, str) else set())
        offer = next((o for o in pol["offers"] if o["id"] not in removed and row.segment in o["segments"]
                      and topics.intersection(o["drivers"])), None)
        offer = offer or next((o for o in pol["offers"] if o["id"] == "O2" and o["id"] not in removed), None)
        if offer:
            cands.append({"customer_id": cid, "p": sc["p"], "value": float(row.monthly_income), "cost": offer["cost"],
                          "offer": offer["name"], "offer_id": offer["id"], "drivers": sc["drivers"],
                          "channel": "relationship manager" if offer["id"] == "O3" else "SMS and app",
                          "suppression": sup.get(cid, [])})
    budget = float(ctx.params.get("budget", pol["campaign_budget"]))
    alloc = ctx.tool("nba_agent", "C5", candidates=cands, budget=budget)
    by_offer = (pd.DataFrame(alloc["chosen"]).groupby("offer").agg(customers=("customer_id", "size"), cost=("cost", "sum"))
                .reset_index().to_dict("records")) if alloc["chosen"] else []
    strat = {"chosen": alloc["chosen"], "by_offer": by_offer, "spent": alloc["spent"], "budget": budget,
             "suppressed": suppressed, "conflicts_open_cases": conflicts, "not_funded": alloc["not_funded"]}
    ctx.out("strategy", strat)
    reasons = pd.Series([w for s_ in suppressed for w in s_["why"]]).value_counts().to_dict() if suppressed else {}
    d = C.require(ctx, "churn_strategy", "Approve the retention strategy", {
        "by_offer": by_offer, "customers": len(alloc["chosen"]), "spent": alloc["spent"], "budget": budget,
        "suppressed": len(suppressed), "suppression_reasons": reasons, "fairness": ctx.get("churn_analysis")["fairness"],
        "offers": [{"id": o["id"], "name": o["name"], "cost": o["cost"]} for o in pol["offers"]],
        "editable": {"budget": budget, "remove_offers": ",".join(sorted(removed)), "launch": bool(ctx.params.get("launch"))}})
    ed = d.get("edits") or {}
    changed = False
    if ed.get("budget") not in (None, "") and float(ed["budget"]) != budget:
        ctx.params["budget"] = float(ed["budget"])
        changed = True
    new_removed = sorted(x.strip() for x in str(ed.get("remove_offers") or "").split(",") if x.strip())
    if "remove_offers" in ed and new_removed != sorted(removed):
        ctx.params["remove_offers"] = new_removed
        changed = True
    if "launch" in ed:
        ctx.params["launch"] = ed["launch"] in (True, "true", "on", "1", 1)
    if changed:
        # the plan changed: the strategy is recomputed and goes back to the manager
        ctx.state["decisions"].pop(f"churn_strategy:{ctx.node_id}", None)
        return ch_strategy(ctx)
    return (f"{len(alloc['chosen'])} customers funded for {money(alloc['spent'])} of {money(budget)}. "
            f"{len(suppressed)} suppressed (DND, consent or open case).")


def ch_report(ctx):
    a, s, sm = ctx.get("churn_analysis"), ctx.get("strategy"), ctx.get("churn_summary")
    fb = (f"{sm['at_risk']} of {sm['cohort']} customers in the cohort have a 90-day churn risk of at least {sm['threshold']:.0%}. "
          f"The strongest drivers are {', '.join(d['feature'] for d in a['drivers'][:3])}. "
          f"The model's held-out AUC is {a['backtest']['auc_holdout']}. "
          f"The strategy funds {len(s['chosen'])} offers for {money(s['spent'])}. "
          f"{len(s['suppressed'])} customers are suppressed for do-not-disturb, missing consent or an open financial-crime case.")
    text, src = narrate(ctx, "Write a churn and retention summary for a retail manager in 5 to 7 sentences.",
                        {"summary": sm, "drivers": a["drivers"][:5], "backtest": a["backtest"], "strategy_by_offer": s["by_offer"],
                         "spent": s["spent"], "suppressed": len(s["suppressed"]), "fairness": a["fairness"]}, fb)
    top = sorted(s["chosen"], key=lambda c: -c["p"])[:25]
    for c in top:
        c["name"] = ctx.tool("retriever", "D3", customer_id=c["customer_id"])["name"]
    C.require(ctx, "release", "Release the churn report", {"summary": text, "headline": sm})
    ctx.state["result"] = {"type": "churn", "summary": text, "summary_source": src, "analysis": a, "strategy": s,
                           "top": top, "headline": sm}
    return "Report released."


def ch_outreach(ctx):
    s = ctx.get("strategy")
    if not ctx.params.get("launch"):
        ctx.state["result"]["outreach"] = {"sent": 0, "blocked": 0, "note": "Outreach not requested."}
        return "Outreach not requested. No customer was contacted."
    sent, blocked, unsafe = 0, 0, []
    for c in s["chosen"]:
        try:
            ctx.tool("nba_agent", "N4", customer_id=c["customer_id"], offer=c["offer"], channel=c["channel"], cost=c["cost"])
            sent += 1
            if c.get("suppression"):
                unsafe.append({"customer_id": c["customer_id"], "why": c["suppression"]})
        except C.Blocked:
            blocked += 1
    ctx.state["unsafe_actions"] = ctx.state.get("unsafe_actions", []) + [{"tool": "N4", **u} for u in unsafe]
    ctx.state["result"]["outreach"] = {"sent": sent, "blocked": blocked, "unsafe": len(unsafe)}
    return f"{sent} offers scheduled, {blocked} blocked by the guard."


# ═══════════════════════════ USE CASE 4: FRAUD AND AML ═══════════════════════════

def aml_plan(ctx):
    m = re.search(r"(\d+)\s*days", ctx.prompt.lower())
    ctx.params.setdefault("days", int(m.group(1)) if m else 7)
    pol = P.body("aml_policy")
    plan = {"window_days": ctx.params["days"], "detectors": ["M1 isolation forest", "M4 structuring, mule and card-fraud rules", "M5 risk fusion"],
            "thresholds": {"alert": pol["alert_score"], "hold": pol["hold_score"], "cash_reporting": pol["cash_reporting_threshold"]},
            "actions": ["N2 open cases (A3)", "N1 holds inside the envelope (A3), outside it (A1)", "N3 STR filing (always A1)"]}
    ctx.out("plan", plan)
    if P.body("autonomy")["plan_approval"].get("aml"):
        d = C.require(ctx, "plan", "Approve the AML surveillance plan", {"plan": plan, "editable": {"days": ctx.params["days"]}})
        if (d.get("edits") or {}).get("days"):
            ctx.params["days"] = int(float(d["edits"]["days"]))
    return f"Monitor the last {ctx.params['days']} days of transactions."


def aml_detect(ctx):
    dq = ctx.tool("retriever", "D14", target="transactions", window_days=ctx.params["days"])
    tx = ctx.tool("retriever", "D1", days=ctx.params["days"])
    an = ctx.tool("transaction_monitor", "M1", tx=tx)
    hits = ctx.tool("transaction_monitor", "M4", tx=tx)
    risk = ctx.tool("transaction_monitor", "M5", hits=hits, tx=tx, anomaly_scores=an)
    pol = P.body("aml_policy")
    alerts = sorted([{"customer_id": k, **v} for k, v in risk.items() if v["score"] >= pol["alert_score"]], key=lambda a: -a["score"])
    watch = sorted([{"customer_id": k, **v} for k, v in risk.items() if v["score"] < pol["alert_score"]], key=lambda a: -a["score"])
    ctx.out("aml_quality", dq)
    ctx.out("alerts", alerts)
    ctx.out("watchlist", watch[:30])
    ctx.out("aml_stats", {"transactions": len(tx), "typology_hits": len(hits), "alerts": len(alerts), "watchlist": len(watch)})
    return (f"{len(tx):,} distinct transactions scored ({dq['issues'][0]['issue'] + ' removed' if dq['issues'] else 'no duplicates'}). "
            f"{len(hits)} typology hits, {len(alerts)} alerts at or above {pol['alert_score']}, "
            f"{len(watch)} anomaly-only items on the watchlist.")


def aml_triage(ctx):
    pol = P.body("aml_policy")
    env = P.body("envelopes")["fraud_hold"]
    cases, wanted = [], []
    for a in ctx.get("alerts"):
        c = ctx.tool("transaction_monitor", "N2", customer_id=a["customer_id"], typology=a["typology"], score=a["score"],
                     narrative="; ".join(a["hit"]["conditions"]))
        cases.append({**c, "customer_id": a["customer_id"], "typology": a["typology"], "score": a["score"]})
        if a["typology"] == "card_fraud" and a["score"] >= pol["hold_score"]:
            wanted.append({"txn_id": a["hit"]["txn_ids"][0], "amount": a["hit"]["total"], "customer_id": a["customer_id"],
                           "within_envelope": env["enabled"] and a["hit"]["total"] <= env["max_amount"]})
    outside = [w for w in wanted if not w["within_envelope"]]
    if outside and ctx.mode == "haara":
        # one batched authorisation for every hold outside the envelope, with its dry run
        d = C.require(ctx, "action", f"Authorise {len(outside)} payment hold(s) outside the envelope", {
            "dry_run": {"system": "Payment switch (simulated)", "holds": [f"{w['txn_id']} {money(w['amount'])} 24h" for w in outside]},
            "level": "A1", "envelope": f"outside envelope (max {money(env['max_amount'])})", "class": "external",
            "editable": {"hold_txns": ",".join(w["txn_id"] for w in outside)}},
            key="action:N1:batch", role=C.route("fraud_hold", ctx.role))
        keep = (d.get("edits") or {}).get("hold_txns")
        keep = {x.strip() for x in keep.split(",")} if isinstance(keep, str) else {w["txn_id"] for w in outside}
        for w in outside:
            if w["txn_id"] in keep and d.get("by"):
                ctx.state["decisions"][f"action:N1:{w['txn_id']}"] = {"decision": "approve", "by": d["by"]}
        wanted = [w for w in wanted if w["within_envelope"] or w["txn_id"] in keep]
    holds = []
    for w in wanted:
        ctx.tool("transaction_monitor", "N1", customer_id=w["customer_id"], txn_id=w["txn_id"], amount=w["amount"], hours=24)
        holds.append(w)
    ctx.out("cases", cases)
    ctx.out("holds", holds)
    return (f"{len(cases)} cases opened (A3, compliance notified). {len(holds)} temporary holds placed, "
            f"{sum(w['within_envelope'] for w in holds)} inside the envelope.")


def aml_investigate(ctx):
    alerts = {a["customer_id"]: a for a in ctx.get("alerts")}
    drafts = []
    for c in ctx.get("cases"):
        if c["typology"] not in ("structuring", "mule_pass_through"):
            continue
        a = alerts[c["customer_id"]]
        prof = ctx.tool("case_investigator", "D3", customer_id=c["customer_id"])
        fb = (f"Suspicious transaction report draft. Subject {prof['name']} ({c['customer_id']}), {prof['segment']} customer of the "
              f"{prof['branch']} branch. Typology: {a['typology'].replace('_', ' ')}. Grounds: {'; '.join(a['hit']['conditions'])}. "
              f"Transactions: {', '.join(a['hit']['txn_ids'][:8])}. Total {money(a['hit']['total'])}. Risk score {a['score']:.2f}.")
        cache = ctx.state.setdefault("llm_cache", {})
        k = "str:" + c["case_id"]
        if k not in cache:
            cache[k] = ctx.tool("case_investigator", "M17",
                                instruction="Draft a concise suspicious transaction report narrative in 5 sentences. Do not tip off the customer.",
                                facts={"case": c, "grounds": a["hit"]["conditions"], "txns": a["hit"]["txn_ids"][:8],
                                       "total": a["hit"]["total"]}) or ""
        drafts.append({"case_id": c["case_id"], "customer_id": c["customer_id"], "typology": a["typology"], "score": a["score"],
                       "grounds": a["hit"]["conditions"], "draft": cache[k] or fb, "source": "llm" if cache[k] else "template"})
    ctx.out("str_drafts", drafts)
    if not drafts:
        ctx.params["file_cases"] = []
        return "No case meets the STR typologies."
    d = C.require(ctx, "aml_case", f"Review {len(drafts)} STR draft(s) for filing", {
        "drafts": drafts, "editable": {"file_cases": ",".join(x["case_id"] for x in drafts)},
        "note": "Only cases listed in file_cases will be filed with the NFIU, under your name."})
    chosen = (d.get("edits") or {}).get("file_cases")
    chosen = [x.strip() for x in chosen.split(",") if x.strip()] if isinstance(chosen, str) else [x["case_id"] for x in drafts]
    if d.get("by"):
        for cid in chosen:
            ctx.state["decisions"][f"action:N3:{cid}"] = {"decision": "approve", "by": d["by"]}
        ctx.state["last_approver"] = d["by"]
    ctx.params["file_cases"] = chosen
    return f"{len(chosen)} of {len(drafts)} STR drafts approved for filing by {d.get('by') or 'autonomous agent'}."


def aml_file(ctx):
    drafts = {x["case_id"]: x for x in ctx.get("str_drafts", [])}
    filed = []
    for cid in ctx.params.get("file_cases", []):
        if cid in drafts:
            ctx.tool("case_investigator", "N3", case_id=cid, customer_id=drafts[cid]["customer_id"], body=drafts[cid]["draft"])
            filed.append(cid)
    ctx.state["result"] = {"type": "aml", "stats": ctx.get("aml_stats"), "alerts": ctx.get("alerts")[:40],
                           "watchlist": ctx.get("watchlist", []),
                           "cases": ctx.get("cases"), "holds": ctx.get("holds"), "drafts": list(drafts.values()), "filed": filed,
                           "quality": ctx.get("aml_quality")}
    return f"{len(filed)} STR(s) filed with the NFIU gateway (simulated)."


# ═══════════════════════════ USE CASE 5: ATM CASH ═══════════════════════════

def cash_load(ctx):
    d = ctx.tool("retriever", "D12")
    dq = ctx.tool("retriever", "D14", target="atm")
    ctx.out("cash_quality", dq)
    return f"{len(d['atms'])} ATMs, {len(d['history']):,} daily records. Quality score {dq['score']:.2f}."


def cash_forecast(ctx):
    d = ctx.tool("retriever", "D12")
    pol = P.body("cash_policy")
    H = pol["horizon_days"]
    plans, back = [], []
    hist = d["history"]
    for atm in d["atms"].itertuples():
        fc = ctx.tool("cash_agent", "M14", history=hist, atm_id=atm.atm_id, days=H)
        f = fc["forecast"]
        back.append({"atm_id": atm.atm_id, **fc["backtest"]})
        bal = fc["balance"]
        cum = f.forecast.cumsum().values + pol["safety_days"] * f.forecast.mean()
        floor_amt = pol["min_fill_ratio"] * atm.capacity
        breach = next((i for i, c in enumerate(cum) if bal - c < floor_amt), None)
        order = None
        if breach is not None:
            day = max(0, breach - 1)
            need = atm.capacity - max(0, bal - f.forecast.iloc[:day].sum())
            order = {"atm_id": atm.atm_id, "delivery_date": f.iloc[day]["date"], "amount": float(round(min(need, atm.capacity), -5))}
        plans.append({"atm_id": atm.atm_id, "location": atm.location, "capacity": atm.capacity, "balance": bal,
                      "week_forecast": float(f.forecast.sum()), "week_high": float(f.high.sum()),
                      "breach_day": None if breach is None else f.iloc[breach]["date"], "order": order,
                      "daily": f.to_dict("records")})
    ctx.out("cash_plans", plans)
    ctx.out("cash_backtest", back)
    cov = float(np.mean([b["coverage"] for b in back]))
    conf = C.confidence({"interval_coverage": cov, "data_quality": ctx.get("cash_quality")["score"]})
    ctx.state["confidence"] = conf["score"]
    ctx.out("cash_conf", conf)
    n = sum(1 for p in plans if p["order"])
    return (f"{n} of {len(plans)} ATMs need cash within {H} days. Backtest interval coverage {cov:.0%}, "
            f"MAPE {np.mean([b['mape'] for b in back]):.1%}.")


def cash_execute(ctx):
    done = []
    env = P.body("envelopes")["cash_orders"]
    for p in ctx.get("cash_plans"):
        o = p["order"]
        if not o:
            continue
        ctx.tool("cash_agent", "N9", atm_id=o["atm_id"], delivery_date=o["delivery_date"], amount=o["amount"])
        done.append({**o, "within_envelope": o["amount"] <= env["max_order_per_atm"]})
    ctx.out("orders", done)
    ctx.state["result"] = {"type": "cash", "plans": ctx.get("cash_plans"), "orders": done, "backtest": ctx.get("cash_backtest"),
                           "confidence": ctx.get("cash_conf"), "envelope": env}
    return f"{len(done)} replenishment orders issued ({sum(1 for d in done if d['within_envelope'])} inside the envelope)."


def cash_report(ctx):
    r = ctx.state["result"]
    tot = sum(o["amount"] for o in r["orders"])
    fb = (f"Cash plan for the next {P.body('cash_policy')['horizon_days']} days. {len(r['orders'])} ATMs need replenishment, "
          f"total {money(tot)}. Orders inside the envelope ran automatically and branch operations was notified. "
          f"Orders outside it waited for branch operations approval. Forecast interval coverage on the last 14 days was "
          f"{np.mean([b['coverage'] for b in r['backtest']]):.0%}.")
    text, src = narrate(ctx, "Summarise the ATM cash plan for branch operations in 4 sentences.",
                        {"orders": r["orders"], "backtest": r["backtest"][:5], "total": tot}, fb)
    r["summary"], r["summary_source"] = text, src
    return "Cash plan summary written."


# ═══════════════════════════ USE CASE 6: QUESTIONS ABOUT YOUR OWN DATA ═══════════════════════════

def ds_check(ctx):
    prof = ctx.tool("retriever", "D18", dataset_id=ctx.params["dataset_id"])
    ctx.params["dataset_name"] = prof["name"]
    ctx.out("profile", {k: prof[k] for k in ("columns", "issues", "score", "rows", "max_date", "values")})
    warn = [i for i in prof["issues"] if i["severity"] == "warning"]
    return (f"{prof['rows']:,} rows and {len(prof['columns'])} columns. Data quality score {prof['score']:.2f}"
            + (f", {len(warn)} warning(s) shown with the answer." if warn else ", no warnings."))


def _is_report(plan):
    """A plan computed from all rows in pandas (full analysis or dashboard), not by one SQL query."""
    return (plan or {}).get("type") in ("report", "dashboard")


def _is_dash(plan):
    return (plan or {}).get("type") == "dashboard"


def _mod(plan):
    return DD if _is_dash(plan) else RP


def _validate_any(plan, cols):
    return _mod(plan).validate(plan, cols) if _is_report(plan) else DS.validate(plan, cols)


def _hash_any(plan):
    return _mod(plan).plan_hash(plan) if _is_report(plan) else DS.plan_hash(plan)


def _describe_any(plan, cols):
    return _mod(plan).describe(plan, cols) if _is_report(plan) else DS.describe(plan, cols)


def _plan_ledger(ctx, event, plan, **extra):
    db.ledger(event, actor=extra.pop("actor", "L6.planner"), run_id=ctx.run_id, node_id="n2",
              payload={"plan_version": plan["version"], "plan_hash": _hash_any(plan), "plan": plan,
                       "dataset_id": ctx.params.get("dataset_id"), **extra})


def _history(ctx, plan, how, text=""):
    ctx.params.setdefault("plan_history", []).append(
        {"version": plan["version"], "hash": _hash_any(plan), "how": how, "text": text[:300], "ts": db.now(),
         "plan": _describe_any(plan, (ctx.get("profile") or {}).get("columns") or [])})


def ds_intent(ctx):
    prof = ctx.get("profile")
    cols = prof["columns"]
    q = ctx.params.get("question") or ctx.prompt
    plan = ctx.params.get("intent")
    if not plan and DD.is_dashboard(q):
        plan = DD.validate({**DD.build_plan(q, cols, prof["values"], RP.distinct_counts(ctx.params["dataset_id"])),
                            "version": 1}, cols)
        ctx.params["intent"] = plan
    elif not plan and RP.is_report(q):
        plan = RP.validate({**RP.build_plan(q, cols, prof["values"], RP.distinct_counts(ctx.params["dataset_id"])),
                            "version": 1}, cols)
        ctx.params["intent"] = plan
    if _is_dash(plan):
        return _dashboard_plan(ctx, plan, cols, q)
    if _is_report(plan):
        return _report_plan_gate(ctx, plan, cols, q)
    if not plan:
        plan = ctx.tool("planner", "M18", question=q, columns=cols, values=prof["values"]) if llm.available() else None
        if not plan or not plan.get("metrics"):
            rules = ctx.tool("planner", "C12", question=q, columns=cols, values=prof["values"], max_date=prof.get("max_date"))
            plan = rules if (rules.get("metrics") or not plan) else plan
        if plan.get("metrics"):
            try:
                plan = DS.validate({**plan, "version": 1}, cols)
            except ValueError as e:
                plan = {"metrics": [], "clarify": str(e)}
        if not plan.get("metrics"):
            ctx.state["result"] = {"type": "clarify", "question": plan.get("clarify")}
            say(ctx, "data_agent", plan.get("clarify"))
            raise StopRun("clarify")
        ctx.params["intent"] = plan
    plan = DS.validate(plan, cols)                 # every run of this step re-checks the stored plan
    ctx.params["intent"] = plan
    if not ctx.params.get("plan_history"):         # the first version is always on record, however it was made
        _history(ctx, plan, "question", q)
        _plan_ledger(ctx, "plan_created", plan, question=q)
    plain = DS.describe(plan, cols)
    h = DS.plan_hash(plan)
    ctx.out("interpretation", {"text": plain, "intent": plan, "plan_version": plan["version"], "plan_hash": h})
    found = DS.conflicts(plan, ctx.params.get("constraints"), cols)
    check = P.body("autonomy").get("bi_interpretation_check", "when_unsure")
    unsure = plan.get("source") == "rules" and re.search(
        T.COMPLEX_WORDS + r"|\b(above|below|over|under|more than|less than|at least|at most|top|bottom)\b", q.lower())
    worded = plan.get("source") in ("corrected_text", "llm")     # the agent read the person's words again: they check the result
    if found or (check != "after_only" and (check == "always" or (check == "when_unsure" and (unsure or worded)))):
        why = ("Your correction and this plan disagree. Correct the plan, or restate what you need."
               if found else "You corrected the question in your own words, so the agent asks you to confirm how it read them."
               if plan.get("source") == "corrected_text" else
               "The wording has a condition such as not, except, above or top, so the agent asks you to confirm."
               if unsure else "Bank policy asks you to confirm every interpretation.")
        C.require(ctx, "interpretation", "Check how your question was understood",
                  {"question": q, "interpretation": plain, "intent": plan, "plan_version": plan["version"],
                   "plan_hash": h, "conflicts": found, "notes": plan.get("notes", []), "why": why},
                  key=f"interpretation:v{plan['version']}:{h}", level_="A1")
    return f"Plan version {plan['version']}: {plain} ({plan.get('source', 'edited')})"


def _dashboard_plan(ctx, plan, cols, q):
    """A dashboard is cheap to rebuild and every filter is visible, so it runs at once. The person confirms or
    corrects it afterwards. Bank policy can still require confirmation first (bi_interpretation_check: always)."""
    plan = DD.validate(plan, cols)
    ctx.params["intent"] = plan
    if not ctx.params.get("plan_history"):
        _history(ctx, plan, "question", q)
        _plan_ledger(ctx, "plan_created", plan, question=q)
    h = DD.plan_hash(plan)
    plain = DD.describe(plan, cols)
    ctx.out("interpretation", {"text": plain, "intent": plan, "plan_version": plan["version"], "plan_hash": h})
    check = P.body("autonomy").get("bi_interpretation_check", "when_unsure")
    if check == "always" or plan.get("source") == "corrected_text":
        C.require(ctx, "interpretation", "Check how your dashboard request was understood",
                  {"question": q, "interpretation": plain, "intent": plan, "plan_version": plan["version"], "plan_hash": h,
                   "conflicts": [], "notes": plan.get("notes", []),
                   "why": "You corrected the dashboard in your own words, so the agent asks you to confirm how it read them."
                   if plan.get("source") == "corrected_text" else "Bank policy asks you to confirm every interpretation."},
                  key=f"interpretation:v{plan['version']}:{h}", level_="A1")
    return f"Dashboard plan version {plan['version']}: {plain}"


def _report_plan_gate(ctx, plan, cols, q):
    """A full analysis runs many computations, so the person approves the whole plan first (plan approval, A1)."""
    plan = RP.validate(plan, cols)
    ctx.params["intent"] = plan
    if not ctx.params.get("plan_history"):
        _history(ctx, plan, "question", q)
        _plan_ledger(ctx, "plan_created", plan, question=q)
    h = RP.plan_hash(plan)
    plain = RP.describe(plan, cols)
    ctx.out("interpretation", {"text": plain, "intent": plan, "plan_version": plan["version"], "plan_hash": h})
    C.require(ctx, "plan", "Approve the analysis plan before it runs",
              {"plan": {row[0]: row[1] for row in RP.plan_details(plan, cols)}, "question": q, "interpretation": plain,
               "plan_version": plan["version"], "plan_hash": h, "notes": plan.get("notes", []),
               "why": "This is a full analysis with many calculations. Check the measures, formulas and sections first."},
              key=f"plan:v{plan['version']}:{h}", level_="A1")
    return f"Report plan version {plan['version']} approved: {plain}"


def ds_query(ctx):
    plan = ctx.params["intent"]
    if _is_dash(plan):
        prof = ctx.get("profile")
        frame = ctx.tool("retriever", "D19", dataset_id=ctx.params["dataset_id"])
        dash = ctx.tool("bi_agent", "C14", frame=frame, plan=plan, columns=prof["columns"], profile={"issues": prof["issues"]})
        ctx.out("data", {"dashboard": dash, "plan_hash": DD.plan_hash(plan), "plan_version": plan["version"],
                         "rows_in_dataset": prof["rows"], "dataset": ctx.params.get("dataset_name")})
        return (f"Dashboard data built: {len(dash['cube']['rows']):,} cells from {dash['rows']:,} rows"
                + (f", {len(dash['items']['values'])} {dash['items']['label']} values ranked." if dash.get("items") else "."))
    if _is_report(plan):
        prof = ctx.get("profile")
        frame = ctx.tool("retriever", "D19", dataset_id=ctx.params["dataset_id"])
        rep_ = ctx.tool("bi_agent", "C13", frame=frame, plan=plan, columns=prof["columns"], profile={"issues": prof["issues"]})
        ctx.out("data", {"report": rep_, "plan_hash": RP.plan_hash(plan), "plan_version": plan["version"],
                         "rows_in_dataset": prof["rows"], "dataset": ctx.params.get("dataset_name")})
        return f"Full analysis computed: {len(rep_['sections'])} sections from {rep_['rows']:,} rows."
    try:
        res = ctx.tool("retriever", "D17", dataset_id=ctx.params["dataset_id"], intent=plan)
    except (ValueError, KeyError) as e:
        ctx.state["result"] = {"type": "clarify", "question": str(e)}
        raise StopRun("clarify")
    if res["plan_hash"] != DS.plan_hash(DS.validate(plan, ctx.get("profile")["columns"])):
        raise ValueError("The query result belongs to a different plan version.")
    ctx.out("data", res)
    return f"{len(res['rows'])} row(s) from the query for plan version {res['plan_version']}. {res['sql_check']}"


def _current(ctx):
    """The plan every downstream step must use, and a guard against results from an older plan."""
    plan = _validate_any(ctx.params["intent"], ctx.get("profile")["columns"])
    res = ctx.get("data")
    if not res or res.get("plan_hash") != _hash_any(plan):
        raise ValueError("The stored result was computed for an older plan. Run the query again.")
    return plan, res


def ds_verify(ctx):
    plan, res = _current(ctx)
    if _is_report(plan):
        checks = []
        for m in plan["measures"]:                       # every reported total is checked against SQL on the stored table
            tot = ctx.tool("verifier", "D17", dataset_id=ctx.params["dataset_id"], intent={"metrics": [{"agg": "sum", "column": m}]})
            t = tot["rows"][0][f"total_{m}"] if tot["rows"] else None
            r = (res.get("report") or res.get("dashboard"))["totals"].get(m)
            checks.append((f"total {m} matches SQL ({t:,.2f})" if t is not None else f"total {m} matches SQL",
                           (t is None and r is None) or (t is not None and r is not None and abs(t - r) <= max(0.01, abs(t) * 1e-9))))
        if _is_dash(plan):                               # the page recomputes from the cube, so the cube must add up too
            cm = res["dashboard"]["cube_matches_totals"]
            checks.append(("dashboard cells add up to the totals", all(cm.values())))
            checks.append((f"every one of the {res['dashboard']['rows']:,} rows is in the dashboard",
                           sum(r[-1] for r in res["dashboard"]["cube"]["rows"]) == res["dashboard"]["rows"]))
        if not checks:
            checks.append(("rows counted", True))
        ok = sum(p for _, p in checks) / len(checks)
        conf = C.confidence({"verifier_checks": ok, "intent_source": 0.9})
        ctx.state["confidence"] = conf["score"]
        ctx.out("checks", {"checks": [{"check": c, "pass": p} for c, p in checks], "confidence": conf, "data_warnings": [],
                           "plan_hash": res["plan_hash"]})
        if not all(p for _, p in checks):
            C.require(ctx, "release", "A total did not match the source data, so you review the report first",
                      {"checks": checks, "summary": "Release only if you accept the difference."}, key=f"release:{res['plan_hash']}")
        return f"{sum(p for _, p in checks)} of {len(checks)} totals match the stored data."
    rows = res["rows"]
    checks = [("result is not empty", bool(rows) and any(r.get(m["alias"]) is not None for r in rows for m in plan["metrics"]))]
    checks.append(("query matches the plan", True))           # D17 refuses to run SQL that does not match
    if plan["dimensions"] and not plan.get("limit") and len(rows) < 500:
        checks.append((f"one row for each {' and '.join(res['dims'])} ({res['expected_groups']} expected)",
                       len(rows) == res["expected_groups"]))
    if plan["dimensions"] and not plan.get("limit") and len(rows) < 500:
        for m in [m for m in plan["metrics"] if m["agg"] in ("sum", "count")]:
            tot = ctx.tool("verifier", "D17", dataset_id=ctx.params["dataset_id"],
                           intent={**plan, "metrics": [m], "dimensions": [], "limit": None, "sort": None})
            t = (tot["rows"][0][m["alias"]] if tot["rows"] else 0) or 0
            vals = [r[m["alias"]] for r in rows if r.get(m["alias"]) is not None]
            checks.append((f"{m['alias']} groups add up to the total", abs(sum(vals) - t) <= max(1e-6, abs(t) * 1e-6)))
    issues = [x for x in ctx.get("profile")["issues"] if x["severity"] == "warning"]
    used = {m["column"] for m in plan["metrics"]} | {d.split(":")[0] for d in plan["dimensions"]} | set(plan["filters"])
    labels = {c["name"]: c["label"] for c in ctx.get("profile")["columns"]}
    relevant = [x for x in issues if x["column"] in [labels.get(u) for u in used] or x["column"] == "(all)"]
    ok = sum(c[1] for c in checks) / len(checks)
    conf = C.confidence({"verifier_checks": ok, "intent_source": 0.9 if plan.get("source") != "rules" else 0.8,
                         "data_quality": max(0.5, 1 - 0.1 * len(relevant))})
    ctx.state["confidence"] = conf["score"]
    ctx.out("checks", {"checks": [{"check": c, "pass": p} for c, p in checks], "confidence": conf, "data_warnings": relevant,
                       "plan_hash": res["plan_hash"]})
    if conf["score"] < C.floor("low") or not all(p for _, p in checks):
        C.require(ctx, "release", "An answer check failed, so you review the answer first",
                  {"checks": checks, "rows": rows[:20], "summary": "A check did not pass. Release only if the figures look right."},
                  key=f"release:{res['plan_hash']}")
    return f"{sum(c[1] for c in checks)} of {len(checks)} checks passed. Confidence {conf['score']:.2f}."


def _fmt(v):
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:,.2f}"


def _explain(plan, res):
    """Plain sentences built only from the computed figures."""
    rows, dims = res["rows"], res["dims"]
    key = lambda r: " ".join(str(r[d]) for d in dims)
    out = []
    for m, meta in zip(plan["metrics"], res["metrics"]):
        vals = [r for r in rows if r.get(m["alias"]) is not None]
        if not vals:
            continue
        if not dims:
            out.append(f"{meta['label']} is {_fmt(vals[0][m['alias']])}.")
            continue
        hi, lo = max(vals, key=lambda r: r[m["alias"]]), min(vals, key=lambda r: r[m["alias"]])
        if m["agg"] in ("sum", "count") and len(vals) > 1:
            tot = sum(r[m["alias"]] for r in vals)
            share = f" ({hi[m['alias']] / tot:.0%} of the total)" if tot else ""
            out.append(f"{key(hi)} has the highest {meta['label'].lower()} at {_fmt(hi[m['alias']])}{share}, and {key(lo)} the lowest at {_fmt(lo[m['alias']])}.")
        elif len(vals) > 1:
            out.append(f"{meta['label']} is highest for {key(hi)} at {_fmt(hi[m['alias']])} and lowest for {key(lo)} at {_fmt(lo[m['alias']])}.")
        else:
            out.append(f"{meta['label']} for {key(hi)} is {_fmt(hi[m['alias']])}.")
    sums = [m for m in plan["metrics"] if m["agg"] == "sum"]
    if dims and len(sums) >= 2 and len(rows) > 1:
        a, b = sums[0]["alias"], sums[1]["alias"]
        top_a = max(rows, key=lambda r: r.get(a) or 0)
        top_b = max(rows, key=lambda r: r.get(b) or 0)
        if key(top_a) != key(top_b):
            out.append(f"{key(top_a)} leads on {res['metrics'][0]['label'].lower()}, but {key(top_b)} leads on "
                       f"{res['metrics'][1]['label'].lower()}, so the largest group is not the strongest on both.")
    return " ".join(out)


def ds_answer(ctx):
    plan, res = _current(ctx)
    if _is_dash(plan):
        cols = ctx.get("profile")["columns"]
        ctx.state["result"] = {"type": "dashboard", "dashboard": res["dashboard"], "checks": ctx.get("checks"), "intent": plan,
                               "plan_version": plan["version"], "plan_hash": res["plan_hash"],
                               "plan_details": DD.plan_details(plan, cols), "dataset_id": ctx.params["dataset_id"],
                               "dataset_name": ctx.params.get("dataset_name"), "notes": plan.get("notes", []),
                               "history": ctx.params.get("plan_history", []), "interpretation": DD.describe(plan, cols),
                               "rows_in_dataset": res["rows_in_dataset"]}
        _plan_ledger(ctx, "plan_executed", plan, actor="L6.narrator", totals=res["dashboard"]["totals"])
        return f"Dashboard released for plan version {plan['version']}."
    if _is_report(plan):
        cols = ctx.get("profile")["columns"]
        ctx.state["result"] = {"type": "report", "report": res["report"], "checks": ctx.get("checks"), "intent": plan,
                               "plan_version": plan["version"], "plan_hash": res["plan_hash"],
                               "plan_details": RP.plan_details(plan, cols), "dataset_id": ctx.params["dataset_id"],
                               "dataset_name": ctx.params.get("dataset_name"), "notes": plan.get("notes", []),
                               "history": ctx.params.get("plan_history", []), "interpretation": RP.describe(plan, cols),
                               "rows_in_dataset": res["rows_in_dataset"]}
        _plan_ledger(ctx, "plan_executed", plan, actor="L6.narrator", sections=[s["id"] for s in res["report"]["sections"]],
                     totals=res["report"]["totals"])
        return f"Full analysis released for plan version {plan['version']}."
    if ctx.get("checks", {}).get("plan_hash") != res["plan_hash"]:
        raise ValueError("The checks were run for an older plan. Run the checks again.")
    rows = res["rows"]
    if not rows:
        fb = "No rows in your dataset matched the question."
    else:
        fb = _explain(plan, res)
    warn = ctx.get("checks")["data_warnings"]
    text, src = narrate(ctx, "Answer the person's question about their own dataset in 2 to 5 sentences using only the "
                             "query result. Explain what the figures show when asked.",
                        {"question": ctx.prompt, "plan": DS.describe(plan, ctx.get("profile")["columns"]), "result": rows[:30],
                         "metrics": res["metrics"], "warnings": warn}, fb)
    ctx.state["result"] = {"type": "dataset", "answer": text, "answer_source": src, "result": res,
                           "checks": ctx.get("checks"), "intent": plan, "plan_version": plan["version"],
                           "plan_hash": res["plan_hash"], "dataset_id": ctx.params["dataset_id"],
                           "dataset_name": ctx.params.get("dataset_name"), "notes": plan.get("notes", []),
                           "history": ctx.params.get("plan_history", []),
                           "interpretation": DS.describe(plan, ctx.get("profile")["columns"])}
    _plan_ledger(ctx, "plan_executed", plan, actor="L6.narrator", sql=res["sql"], params=res["params"], rows=len(rows))
    return f"Answer released for plan version {plan['version']} with the query used."


PLANS = {
    "credit": [("n1", "Intake", "planner", cr_intake), ("n2", "Retrieve applicant data", "retriever", cr_retrieve),
               ("n3", "Data quality check", "retriever", cr_quality), ("n4", "Score and explain", "credit_agent", cr_score),
               ("n5", "Apply loan policy", "credit_agent", cr_policy), ("n6", "Recourse and fairness", "credit_agent", cr_recourse),
               ("n7", "Independent verification", "verifier", cr_verify), ("n8", "Officer review and decision", "credit_agent", cr_decide),
               ("n9", "Write to loan origination", "credit_agent", cr_write), ("n10", "Credit memo", "narrator", cr_memo)],
    "bi": [("n1", "Interpret the question", "planner", bi_intent), ("n2", "Governed semantic query", "retriever", bi_query),
           ("n3", "Verify the result", "verifier", bi_verify), ("n4", "Answer with transparency", "narrator", bi_answer)],
    "churn": [("n1", "Plan", "planner", ch_plan), ("n2", "Build cohort", "retriever", ch_cohort),
              ("n3", "Score churn risk", "churn_agent", ch_score), ("n4", "Drivers, backtest and fairness", "churn_agent", ch_analyse),
              ("n5", "Retention strategy", "nba_agent", ch_strategy), ("n6", "Report", "narrator", ch_report),
              ("n7", "Outreach (if authorised)", "nba_agent", ch_outreach)],
    "aml": [("n1", "Plan", "planner", aml_plan), ("n2", "Detect and score", "transaction_monitor", aml_detect),
            ("n3", "Triage: cases and holds", "transaction_monitor", aml_triage),
            ("n4", "Investigate and draft STRs", "case_investigator", aml_investigate),
            ("n5", "File approved STRs", "case_investigator", aml_file)],
    "cash": [("n1", "Load cash data", "retriever", cash_load), ("n2", "Forecast and plan", "cash_agent", cash_forecast),
             ("n3", "Issue orders", "cash_agent", cash_execute), ("n4", "Summary", "narrator", cash_report)],
    "upload": [("n1", "Check your data", "retriever", ds_check), ("n2", "Interpret the question", "planner", ds_intent),
               ("n3", "Query your dataset", "retriever", ds_query), ("n4", "Verify the result", "verifier", ds_verify),
               ("n5", "Answer with the query used", "narrator", ds_answer)],
}

USE_CASES = {
    "credit": "Alternative-data credit decision", "bi": "Conversational BI", "churn": "Churn prediction and retention",
    "aml": "Fraud and AML monitoring", "cash": "ATM cash forecasting and replenishment",
    "upload": "Questions about your own data",
}


def route_prompt(prompt: str) -> str:
    q = prompt.lower()
    if re.search(r"app-\d{4}|loan application|credit decision|assess .*application|should we (give|grant|approve)", q):
        return "credit"
    if re.search(r"churn|leave|attrition|retention|retain", q):
        return "churn"
    if re.search(r"fraud|\baml\b|money laundering|suspicious|structuring|mule|surveillance|monitor .*transactions", q):
        return "aml"
    if re.search(r"\batms?\b|cash replenish|cash plan|replenishment|cash-in-transit|vault", q):
        return "cash"
    if llm.available():
        out = llm.chat_json("Classify a bank staff request into one use case: credit, bi, churn, aml, cash. "
                            'Return {"use_case": "..."}.', prompt)
        if out and out.get("use_case") in PLANS:
            return out["use_case"]
    return "bi"


# ═══════════════════════════ ORCHESTRATOR (L6.2) ═══════════════════════════

_threads = {}
_run_locks = {}


def _actor(role, who):
    return auth.actor(who) if who else f"human:{role}"


def _by(role, who):
    return auth.label(who) if who else role


def create_run(prompt, role, mode="haara", use_case=None, params=None, who=None) -> str:
    uc = use_case or route_prompt(prompt)
    run_id = "RUN-" + uuid.uuid4().hex[:8]
    plan = [{"id": n, "title": t, "agent": a, "status": "pending", "summary": None} for n, t, a, _ in PLANS[uc]]
    state = {"out": {}, "params": params or {}, "decisions": {}, "pointer": 0, "policy_versions": P.versions(),
             "models": [m["version"] for m in M.registry()]}
    db.execute("INSERT INTO runs (run_id, created_at, updated_at, use_case, prompt, role, mode, status, plan, state, "
               "result, created_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (run_id, db.now(), db.now(), uc, prompt, role, mode, "running", json.dumps(plan), json.dumps(state), None,
                who["email"] if who else None))
    db.ledger("run_created", actor=_actor(role, who), run_id=run_id, payload={"use_case": uc, "prompt": prompt, "mode": mode,
                                                                             "policy_versions": state["policy_versions"]})
    return run_id


def _save(ctx, status):
    db.execute("UPDATE runs SET status=?, plan=?, state=?, updated_at=? WHERE run_id=?",
               (status, json.dumps(ctx.plan), json.dumps(ctx.state, default=_json_default), db.now(), ctx.run_id))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)


def load(run_id):
    return Ctx(db.one("SELECT * FROM runs WHERE run_id=?", (run_id,)))


def execute(run_id):
    lock = _run_locks.setdefault(run_id, threading.Lock())
    with lock:
        _execute(run_id)


def advance(run_id, seconds=4.0) -> bool:
    """Serverless mode: move a running task forward inside a request, for about `seconds`.

    A lease in the runs table makes sure only one request works on a task at a time, across every server
    instance. A lease left by an instance that was shut down expires after 90 seconds, and the task resumes from
    its last saved step. Returns True when this request did some work."""
    got = db.query("UPDATE runs SET lease_until = now() + interval '90 seconds' WHERE run_id=? AND status='running' "
                   "AND (lease_until IS NULL OR lease_until < now()) RETURNING run_id", (run_id,))
    if not got:
        return False
    try:
        _execute(run_id, deadline=time.time() + seconds)
    finally:
        db.execute("UPDATE runs SET lease_until=NULL WHERE run_id=?", (run_id,))
    return True


def _execute(run_id, deadline=None):
    """Run nodes from the pointer until the run finishes, pauses or waits for a human.
    With a deadline, it also stops after the step that passes it and leaves the task running for the next call."""
    ctx = load(run_id)
    nodes = PLANS[ctx.use_case]
    first = ctx.state["pointer"]
    while ctx.state["pointer"] < len(nodes):
        if deadline and ctx.state["pointer"] > first and time.time() > deadline:   # always at least one step per call
            _save(ctx, "running")
            return
        if ctx.state.get("pause"):
            ctx.state.pop("pause", None)
            _save(ctx, "paused")
            return
        i = ctx.state["pointer"]
        nid, title, agent, fn = nodes[i]
        ctx.node_id = nid
        ctx.plan[i]["status"] = "running"
        _save(ctx, "running")
        try:
            summary = fn(ctx)
            if ctx.mode == "uniform":
                C.require(ctx, "review", f"Review step: {title}", {"summary": summary}, key=f"uniform:{nid}", level_="A2")
            ctx.plan[i].update(status="done", summary=summary, tools=len(ctx.calls), blocked=list(ctx.blocked))
            db.ledger("node_completed", actor=agent, run_id=run_id, node_id=nid, payload={"title": title, "summary": summary})
            ctx.calls, ctx.blocked = [], []
            ctx.state["pointer"] = i + 1
        except C.NeedsApproval as na:
            if not db.one("SELECT cp_id FROM checkpoints WHERE run_id=? AND status='open'", (run_id,)):
                ctx.state["open_checkpoint"] = C.create_checkpoint(ctx, na)
            ctx.plan[i].update(status="waiting", summary=f"Waiting for {na.role}: {na.title}")
            _save(ctx, "awaiting_approval")
            return
        except StopRun:
            ctx.plan[i].update(status="done", summary="Needs clarification from the user.")
            _save(ctx, "needs_input")
            return
        except C.Blocked as b:
            ctx.plan[i].update(status="blocked", summary=str(b), blocked=list(ctx.blocked))
            ctx.state["blocked_reason"] = str(b)
            db.ledger("run_stopped", actor="L7", run_id=run_id, node_id=nid, payload={"reason": str(b)})
            _save(ctx, "stopped")
            return
        except Exception as e:
            ctx.plan[i].update(status="failed", summary=f"{type(e).__name__}: {e}")
            ctx.state["error"] = traceback.format_exc()[-3000:]
            _log.exception("task %s (%s) failed at step %s %s", run_id, ctx.use_case, nid, title)
            _save(ctx, "failed")
            return
    _save(ctx, "completed")
    db.ledger("run_completed", actor="L6.2.orchestrator", run_id=run_id,
              payload={"use_case": ctx.use_case, "skipped_gates": ctx.state.get("skipped_gates", [])})


def start(run_id, background=True):
    if background and config.serverless():      # no threads: status requests move the task forward (advance)
        return None
    if background:
        t = threading.Thread(target=execute, args=(run_id,), daemon=True)
        _threads[run_id] = t
        t.start()
        return t
    execute(run_id)


def _deny_decision(cp_id, role, who, reason):
    db.ledger("decision_denied", actor=_actor(role, who), payload={"cp_id": cp_id, "reason": reason})
    raise PermissionError(reason)


def decide(cp_id, decision, role, comment="", edits=None, background=True, who=None, resume=True):
    cp = db.one("SELECT * FROM checkpoints WHERE cp_id=?", (cp_id,))
    if not cp or cp["status"] != "open":
        db.ledger("decision_denied", actor=_actor(role, who), payload={"cp_id": cp_id, "reason": "not open"})
        raise ValueError("Checkpoint is not open.")
    if role != cp["assigned_role"]:
        _deny_decision(cp_id, role, who, f"This checkpoint is assigned to {cp['assigned_role']}. You are acting as {role}.")
    if who and cp.get("assigned_user") and who["email"].lower() != cp["assigned_user"].lower():
        _deny_decision(cp_id, role, who, "This decision belongs to the person who asked the question.")
    if who:
        u = db.one("SELECT role, active FROM app_users WHERE lower(email)=lower(?)", (who["email"],))
        if not u or not u["active"] or u["role"] != role:
            _deny_decision(cp_id, role, who, "Your account is not active in this role.")
    ctx = load(cp["run_id"])
    ev = json.loads(cp["evidence"])
    key = ev["key"]
    if ctx.use_case == "upload" and cp["kind"] in ("interpretation", "plan") and decision == "approve":
        problem = _upload_plan_problem(ctx, ev.get("plan_version"), ev.get("plan_hash"))
        if problem:
            _deny_decision(cp_id, role, who, problem)
    by = _by(role, who)
    C.record_decision(cp, decision, role, comment, edits, ctx.use_case, by=by, actor=_actor(role, who))
    ctx.state["decisions"][key] = {"decision": decision, "by": by, "comment": comment, "edits": edits or {},
                                   "node": cp["node_id"], "ts": db.now(), "by_role": role,
                                   "by_email": who["email"] if who else None, "policy_versions": P.versions(),
                                   "plan_version": ev.get("plan_version"), "plan_hash": ev.get("plan_hash")}
    ctx.state["last_approver"] = by
    ctx.state.pop("open_checkpoint", None)
    _save(ctx, "running")
    if resume:
        start(ctx.run_id, background)


def steer(run_id, node_id, params, role, background=True, who=None):
    """Steering bus: change parameters and recompute from a node onward."""
    ctx = load(run_id)
    ids = [n[0] for n in PLANS[ctx.use_case]]
    i = ids.index(node_id)
    ctx.state["params"].update(params or {})
    ctx.state["pointer"] = i
    ctx.state.pop("pause", None)
    ctx.state.pop("result", None)
    ctx.state.pop("interpretation_confirmed", None)
    if ctx.use_case == "upload":           # results of an older plan must never reach a later step
        for k in ("data", "checks", "interpretation"):
            ctx.state["out"].pop(k, None)
    ctx.state["decisions"] = {k: v for k, v in ctx.state["decisions"].items()
                              if v.get("node") in ids and ids.index(v["node"]) < i}
    for n in ctx.plan[i:]:
        n.update(status="pending", summary=None)
    for cp in db.query("SELECT cp_id FROM checkpoints WHERE run_id=? AND status='open'", (run_id,)):
        db.execute("UPDATE checkpoints SET status='superseded' WHERE cp_id=?", (cp["cp_id"],))
    db.ledger("run_steered", actor=_actor(role, who), run_id=run_id, node_id=node_id, payload={"params": params})
    _save(ctx, "running")
    start(run_id, background)


def pause(run_id, role, who=None):
    ctx = load(run_id)
    ctx.state["pause"] = True
    _save(ctx, db.one("SELECT status FROM runs WHERE run_id=?", (run_id,))["status"])
    db.ledger("run_pause_requested", actor=_actor(role, who), run_id=run_id)


def correct_interpretation(run_id, text, role, intent=None, who=None):
    """The person who asked says the agent misread the question. Their correction
    replaces the interpretation and the answer is recomputed from the first step."""
    ctx = load(run_id)
    if ctx.use_case not in ("bi", "upload"):
        raise ValueError("Corrections apply to questions about the bank or your data.")
    if who and ctx.created_by and who["email"].lower() != ctx.created_by.lower():
        raise PermissionError("Only the person who asked can correct how the question was understood.")
    if ctx.use_case == "upload":
        return _correct_dataset(ctx, text, role, intent, who)
    old = ctx.params.get("intent") or {}
    if intent:                                   # structured edit from the form
        new = {"metric": intent.get("metric") or old.get("metric"), "dimensions": intent.get("dimensions") or [],
               "filters": {k: v for k, v in (intent.get("filters") or {}).items() if v not in (None, "", [])},
               "period": intent.get("period") or None}
    else:                                        # plain-language correction
        new = None
        if llm.available():
            new = llm.chat_json(
                "A bank analyst says an agent misread their question. Return the corrected intent as JSON with keys "
                "metric, dimensions, filters, period, using the same vocabulary as the current intent. A filter value is "
                "a string (is), a list (is one of) or {\"not\": value} (is not). Change only what the analyst corrects.",
                json.dumps({"question": ctx.prompt, "current_intent": old, "correction": text}))
            if not (new and new.get("metric") in P.body("semantic_layer")["metrics"]):
                new = None
        if new is None:
            fix = T.intent_rules(None, text)
            new = dict(old)
            if fix.get("metric"):
                new["metric"] = fix["metric"]
            if fix.get("dimensions"):
                new["dimensions"] = fix["dimensions"]
            filters = dict(old.get("filters") or {})
            filters.update(fix.get("filters") or {})
            low = text.lower()
            if re.search(r"\b(overall|in total|total only|no breakdown|not broken down|single total)\b", low):
                new["dimensions"] = []
            for k, words in (("segment", "segments"), ("region", "regions")):
                if k not in (fix.get("filters") or {}) and re.search(rf"\b(all|every|any) {words}\b", low):
                    filters.pop(k, None)
            if "city" not in (fix.get("filters") or {}) and \
                    re.search(r"\b(all cities|every city|any city|no city|whole bank|all branches)\b", text.lower()):
                filters.pop("city", None)
            new["filters"] = filters
            if fix.get("period"):
                new["period"] = fix["period"]
            if re.search(r"\b(all (available )?data|all time|no period)\b", text.lower()):
                new["period"] = None
    if new.get("metric") not in P.body("semantic_layer")["metrics"]:
        raise ValueError("Choose a measure the bank has defined.")
    new["source"] = "corrected"
    say(ctx, _actor(role, who), text or "Corrected the interpretation in the form.")
    say(ctx, "bi_agent", "Understood. I will now answer: " + T.describe_intent(new))
    db.execute("INSERT INTO oversight_memory (ts, run_id, use_case, kind, decision, changed, role, comment) "
               "VALUES (?,?,?,?,?,?,?,?)", (db.now(), run_id, "bi", "interpretation", "corrected", 1, role, text or ""))
    db.ledger("interpretation_corrected", actor=_actor(role, who), run_id=run_id, node_id="n1",
              payload={"from": old, "to": new, "text": text})
    steer(run_id, "n1", {"intent": new}, role, who=who)
    return {"interpretation": T.describe_intent(new), "intent": new}


MORE = r"\b(too|also|as well|in addition|additionally|add|plus)\b"


def _correct_dataset(ctx, text, role, intent=None, who=None):
    """A correction of how a question about an uploaded dataset was read. It produces a new plan version.

    An explicit instruction to discard the plan replaces the whole plan. A correction that restates the measures
    and the grouping replaces those parts and the filters. A partial correction changes only what it names.
    Words and form together must agree, otherwise the person is asked which one they mean."""
    prof = ctx.get("profile") or {}
    cols, values = prof.get("columns") or [], prof.get("values") or {}
    prev = ctx.params.get("intent") or {}
    if text and not intent and (_is_dash(prev) or DD.is_dashboard(text)):
        dist = RP.distinct_counts(ctx.params["dataset_id"])
        if _is_dash(prev) and not DD.is_dashboard(text):
            new = DD.validate(DD.revise(prev, text, cols, dist), cols)
        else:
            new = DD.validate({**DD.build_plan(text, cols, values, dist), "version": prev.get("version", 0) + 1,
                               "source": "corrected_text"}, cols)
        say(ctx, _actor(role, who), text)
        say(ctx, "data_agent", f"Plan version {new['version']}: " + DD.describe(new, cols))
        db.ledger("interpretation_corrected", actor=_actor(role, who), run_id=ctx.run_id, node_id="n2",
                  payload={"from_version": prev.get("version"), "to_version": new["version"],
                           "mode": "update" if _is_dash(prev) else "replace", "from": prev, "to": new, "text": text,
                           "dataset_id": ctx.params.get("dataset_id")})
        hist = ctx.params.get("plan_history", []) + [{"version": new["version"], "hash": DD.plan_hash(new), "how": "update",
                                                       "text": text[:300], "ts": db.now(), "plan": DD.describe(new, cols)}]
        steer(ctx.run_id, "n2", {"intent": new, "constraints": None, "plan_history": hist}, role, who=who)
        return {"interpretation": DD.describe(new, cols), "intent": new, "plan_version": new["version"], "mode": "update"}
    if _is_report(prev) or (text and not intent and RP.is_report(text)):
        ver = prev.get("version", 0) + 1
        if text and not intent and RP.is_report(text):
            new = RP.validate({**RP.build_plan(text, cols, values, RP.distinct_counts(ctx.params["dataset_id"])),
                               "version": ver}, cols)
        else:
            new = DS.validate({**(intent or DS.interpret(text, cols, values, prof.get("max_date"))), "version": ver,
                               "source": "corrected_form" if intent else "corrected_text"}, cols)
        say(ctx, _actor(role, who), text or "Set the plan in the form.")
        say(ctx, "data_agent", f"Plan version {new['version']}: " + _describe_any(new, cols))
        db.ledger("interpretation_corrected", actor=_actor(role, who), run_id=ctx.run_id, node_id="n2",
                  payload={"from_version": prev.get("version"), "to_version": new["version"], "mode": "replace",
                           "from": prev, "to": new, "text": text, "dataset_id": ctx.params.get("dataset_id")})
        hist = ctx.params.get("plan_history", []) + [{"version": new["version"], "hash": _hash_any(new), "how": "replace",
                                                       "text": (text or "")[:300], "ts": db.now(), "plan": _describe_any(new, cols)}]
        steer(ctx.run_id, "n2", {"intent": new, "constraints": DS.constraints(text, cols) if text and not _is_report(new) else None,
                                 "plan_history": hist}, role, who=who)
        return {"interpretation": _describe_any(new, cols), "intent": new, "plan_version": new["version"], "mode": "replace"}
    old = DS.validate(prev, cols) if prev.get("metrics") or prev.get("measure") else None
    ver = (old or {}).get("version", 0) + 1
    text = (text or "").strip()
    low = " " + text.lower() + " "
    cons = DS.constraints(text, cols) if text else None
    if intent:                                              # exact settings from the form
        new = DS.validate({**intent, "version": ver, "source": "corrected_form"}, cols)
        found = DS.conflicts(new, cons, cols) if cons else []
        if found:
            db.ledger("correction_conflict", actor=_actor(role, who), run_id=ctx.run_id, node_id="n2",
                      payload={"text": text, "form": new, "conflicts": found})
            raise ValueError("Your words and the form settings disagree. " + " ".join(found) + " Change one of them.")
        mode = "form"
    else:
        new = None
        if llm.available():
            out = llm.chat_json(
                "A person says an agent misread their question about a spreadsheet. Return the corrected analytical plan "
                "as JSON with the same keys as the current plan. If the person says to discard or replace the plan, "
                "build it from their words only. Otherwise change only what they correct. Use only the columns given.",
                json.dumps({"question": ctx.prompt, "current_plan": old, "correction": text,
                            "columns": [{"column": c["name"], "label": c["label"], "role": c["role"]} for c in cols
                                        if c["role"] not in ("personal", "empty")]}))
            try:
                new = DS.validate({**out, "version": ver, "source": "corrected_text"}, cols) if out else None
            except ValueError:
                new = None
            mode = "llm"
        if new is None:
            fix = DS.interpret(text, cols, values, prof.get("max_date"))
            fix_m = [m for m in fix["metrics"] if m.get("column") or m["agg"] in ("count", "count_distinct")]
            explicit = bool(DS.REPLACE.search(low)) or old is None
            restate = bool(fix_m) and bool(fix["dimensions"] or fix.get("dimensions_only") or fix.get("excluded_dimensions"))
            more = re.search(MORE, low)
            if explicit:
                mode = "replace"
                if not fix_m:
                    raise ValueError("Say what the new plan should measure, for example total sales by category.")
                new = {k: fix.get(k) for k in DS.PLAN_FIELDS}
                new["metrics"] = fix_m
            else:
                mode = "restate" if restate and not more else "update"
                new = json.loads(json.dumps(old))
                agg, _ = DS._agg_word(low)
                if fix_m:
                    new["metrics"] = (old["metrics"] + fix_m) if more and mode == "update" else fix_m
                elif agg and agg != "count":
                    if len(old["metrics"]) != 1:
                        raise ValueError("Say which measure should change, for example: use the average profit margin.")
                    new["metrics"] = [{"agg": agg, "column": old["metrics"][0]["column"]}]
                if fix["dimensions"] or fix.get("dimensions_only") or mode == "restate":
                    if more and mode == "update" and not fix.get("dimensions_only"):
                        new["dimensions"] = list(dict.fromkeys(old["dimensions"] + fix["dimensions"]))
                    else:
                        new["dimensions"] = fix["dimensions"]
                if re.search(r"\b(overall|in total|total only|no breakdown|not broken down|single total)\b", low):
                    new["dimensions"] = []
                if mode == "restate":
                    new["filters"] = dict(fix["filters"])
                else:
                    filters = dict(new.get("filters") or {})
                    for k, v in (fix.get("filters") or {}).items():
                        o = filters.get(k)
                        neg = lambda x: isinstance(x, dict) and "not" in x
                        lst = lambda x: x if isinstance(x, list) else [x]
                        if more and o is not None and neg(o) == neg(v) and (neg(v) or not isinstance(v, dict)):
                            vals = list(dict.fromkeys(lst(o["not"] if neg(o) else o) + lst(v["not"] if neg(v) else v)))
                            v = {"not": vals} if neg(v) else vals
                        filters[k] = v
                    for c in cols:
                        ph = c["label"].lower()
                        if re.search(rf"\b(all|every|any)\s+{re.escape(ph)}s?\b|\bno filter on {re.escape(ph)}\b", low) \
                                and c["name"] not in (fix.get("filters") or {}):
                            filters.pop(c["name"], None)
                    new["filters"] = filters
                if fix.get("period"):
                    new["period"] = fix["period"]
                if fix.get("limit"):
                    new["limit"], new["sort"] = fix["limit"], fix["sort"]
                if fix.get("output_format") != "auto":
                    new["output_format"] = fix["output_format"]
                new["explain"] = bool(old.get("explain") or fix.get("explain"))
            for a_txt, b_txt in re.findall(r"\b(?:use|show|group by|break down by|by|measure)?\s*([a-z_ ]{2,40}?)\s+instead of\s+(?:the\s+)?([a-z_ ]{2,40}?)(?=[,.;]|\s+(?:and|but)\s|\s*$)", low):
                usable = [c for c in cols if c["role"] not in ("personal", "empty")]
                to, frm = DS._best(" " + a_txt + " ", usable), DS._best(" " + b_txt + " ", usable)
                if not (to and frm) or to["name"] == frm["name"]:
                    continue
                dims = new.get("dimensions") or []
                if any(d.split(":")[0] == frm["name"] for d in dims) and to["role"] in ("category", "identifier", "date"):
                    new["dimensions"] = [(to["name"] if to["role"] != "date" else f"{to['name']}:month")
                                         if d.split(":")[0] == frm["name"] else d for d in dims]       # "region instead of category"
                elif to["role"] == "measure":
                    new["metrics"] = [{**m, "column": to["name"]} if m.get("column") == frm["name"] else m
                                      for m in new.get("metrics") or []]                             # "profit instead of sales"
            for d in fix.get("excluded_dimensions") or []:
                new["dimensions"] = [x for x in new.get("dimensions") or [] if x.split(":")[0] != d]
            if re.search(r"\b(no filters?|without (?:any )?filters?|do not filter|don't filter)\b", low):
                new["filters"] = {}
            if re.search(r"\b(all (available )?data|all time|all rows|no period|whole period)\b", low):
                new["period"] = None
            new["notes"] = fix.get("notes", [])
            new = DS.validate({**new, "version": ver, "source": "corrected_text"}, cols)
    if old and DS.plan_hash(new) == DS.plan_hash(old):
        say(ctx, _actor(role, who), text or "Corrected the interpretation in the form.")
        say(ctx, "data_agent", "Your correction gives the same plan as before, version " + str(old["version"]) +
            ". Say what is still wrong, or set it exactly in the form.")
        db.ledger("correction_unchanged", actor=_actor(role, who), run_id=ctx.run_id, node_id="n2",
                  payload={"text": text, "plan_version": old["version"]})
        return {"interpretation": DS.describe(old, cols), "intent": old, "plan_version": old["version"], "unchanged": True}
    say(ctx, _actor(role, who), text or "Corrected the interpretation in the form.")
    say(ctx, "data_agent", f"Plan version {new['version']}: " + DS.describe(new, cols))
    db.execute("INSERT INTO oversight_memory (ts, run_id, use_case, kind, decision, changed, role, comment) "
               "VALUES (?,?,?,?,?,?,?,?)", (db.now(), ctx.run_id, "upload", "interpretation", "corrected", 1, role, text or ""))
    db.ledger("interpretation_corrected", actor=_actor(role, who), run_id=ctx.run_id, node_id="n2",
              payload={"from_version": (old or {}).get("version"), "to_version": new["version"], "mode": mode,
                       "from": old, "to": new, "from_hash": DS.plan_hash(old) if old else None, "to_hash": DS.plan_hash(new),
                       "text": text, "dataset_id": ctx.params.get("dataset_id")})
    hist = ctx.params.get("plan_history", []) + [{"version": new["version"], "hash": DS.plan_hash(new), "how": mode,
                                                   "text": text[:300], "ts": db.now(), "plan": DS.describe(new, cols)}]
    steer(ctx.run_id, "n2", {"intent": new, "constraints": cons, "plan_history": hist}, role, who=who)
    return {"interpretation": DS.describe(new, cols), "intent": new, "plan_version": new["version"], "mode": mode}


def _upload_plan_problem(ctx, plan_version=None, plan_hash=None):
    """Why a confirmation cannot apply to the current plan, or an empty string."""
    cols = (ctx.get("profile") or {}).get("columns") or []
    plan = _validate_any(ctx.params["intent"], cols)
    if plan_version is not None and int(plan_version) != plan["version"]:
        return f"The plan changed after you saw it. Review version {plan['version']} before confirming."
    if plan_hash and plan_hash != _hash_any(plan):
        return f"The plan changed after you saw it. Review version {plan['version']} before confirming."
    found = [] if _is_report(plan) else DS.conflicts(plan, ctx.params.get("constraints"), cols)
    if found:
        return "Your last correction and this plan disagree. " + " ".join(found) + " Correct the plan before confirming it."
    return ""


def confirm_interpretation(run_id, role, who=None, plan_version=None, plan_hash=None):
    """The person who asked confirms the answer used the right meaning (review after, A2).
    For a dataset question the confirmation names the plan version the person saw, and applies only to it."""
    ctx = load(run_id)
    if who and ctx.created_by and who["email"].lower() != ctx.created_by.lower():
        raise PermissionError("Only the person who asked can confirm the interpretation.")
    if ctx.use_case == "upload":
        if plan_version is None and plan_hash is None:
            raise ValueError("Say which plan version you are confirming.")
        res = ctx.state.get("result") or {}
        problem = _upload_plan_problem(ctx, plan_version, plan_hash) or \
            ("" if res.get("plan_hash") == _hash_any(_validate_any(ctx.params["intent"], ctx.get("profile")["columns"]))
             else "The answer shown was computed for an older plan. Wait for the new answer.")
        if problem:
            db.ledger("confirmation_refused", actor=_actor(role, who), run_id=run_id, node_id="n2",
                      payload={"plan_version_seen": plan_version, "reason": problem})
            raise ValueError(problem)
        plan = _validate_any(ctx.params["intent"], ctx.get("profile")["columns"])
        db.execute("INSERT INTO oversight_memory (ts, run_id, use_case, kind, decision, changed, role, comment) "
                   "VALUES (?,?,?,?,?,?,?,?)", (db.now(), run_id, "upload", "interpretation", "confirmed", 0, role, ""))
        db.ledger("plan_confirmed", actor=_actor(role, who), run_id=run_id, node_id="n2",
                  payload={"plan_version": plan["version"], "plan_hash": _hash_any(plan), "plan": plan})
        ctx.state["interpretation_confirmed"] = f"{_by(role, who)}, plan version {plan['version']}"
        _save(ctx, db.one("SELECT status FROM runs WHERE run_id=?", (run_id,))["status"])
        return {"ok": True, "plan_version": plan["version"]}
    db.execute("INSERT INTO oversight_memory (ts, run_id, use_case, kind, decision, changed, role, comment) "
               "VALUES (?,?,?,?,?,?,?,?)", (db.now(), run_id, ctx.use_case, "interpretation", "confirmed", 0, role, ""))
    db.ledger("interpretation_confirmed", actor=_actor(role, who), run_id=run_id, node_id="n2" if ctx.use_case == "upload" else "n1",
              payload={"intent": ctx.params.get("intent")})
    ctx.state["interpretation_confirmed"] = _by(role, who)
    _save(ctx, db.one("SELECT status FROM runs WHERE run_id=?", (run_id,))["status"])
    return {"ok": True}


def challenge(run_id, text, role, who=None):
    ctx = load(run_id)
    cp = db.one("SELECT * FROM checkpoints WHERE run_id=? AND status='open' AND kind IN ('credit_decision','credit_exception')", (run_id,))
    if not cp:
        raise ValueError("No open credit decision to discuss.")
    ctx.node_id = cp["node_id"]
    out = respond_to_challenge(ctx, text, role, who)
    if out["changed"]:
        ev = credit_evidence(ctx)
        ev["key"] = json.loads(cp["evidence"])["key"]
        db.execute("UPDATE checkpoints SET evidence=? WHERE cp_id=?", (json.dumps(ev, default=_json_default), cp["cp_id"]))
    db.ledger("officer_challenge", actor=_actor(role, who), run_id=run_id, payload={"text": text, "changed": out["changed"],
                                                                                   "what_if": out.get("what_if")})
    _save(ctx, "awaiting_approval")
    return out
