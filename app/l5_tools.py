"""L5 Services and tools: the agents' entire action space.

Each tool declares its sub-layer, action class (read, compute, internal_write,
external), an optional policy precheck, an optional envelope test and a
dry-run preview. Agents never import this module's functions directly. They
call tools by ID through the L7 action guard.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from . import features as F
from . import l1_data as db
from . import l2_policy as P
from . import l3_models as M
from . import l4_xai as X
from . import llm

AS_OF = pd.Timestamp("2026-09-30")
REG = {}


def tool(tid, name, layer, cls, policy="", model=None, precheck=None, envelope=None, preview=None,
         key=None, route="escalation", ikey=None, ikey_scope="global"):
    def wrap(fn):
        REG[tid] = {"id": tid, "name": name, "layer": layer, "class": cls, "fn": fn, "policy": policy,
                    "model": model, "precheck": precheck, "envelope": envelope, "preview": preview,
                    "key": key, "route": route, "ikey": ikey, "ikey_scope": ikey_scope, "doc": (fn.__doc__ or "").strip()}
        return fn
    return wrap


def money(x):
    return f"NGN {x:,.0f}"


# ═════════════════════════ L5.1 DATA ═════════════════════════

@tool("D2", "Core banking query", "L5.1", "read")
def core_banking(ctx, customer_id):
    """Accounts and loans of one customer."""
    return {"accounts": db.query("SELECT account_id, product, status FROM accounts WHERE customer_id=?", (customer_id,)),
            "loans": db.query("SELECT loan_id, product, outstanding, monthly_installment, dpd, status FROM loans "
                              "WHERE customer_id=?", (customer_id,))}


def _pii_allowed(ctx):
    return P.body("entitlements").get(ctx.role, {}).get("pii", False)


@tool("D15", "PII masking", "L5.1", "compute")
def mask(ctx, record):
    """Masks name and phone for roles without PII entitlement."""
    r = dict(record)
    if not _pii_allowed(ctx):
        if "name" in r and r["name"]:
            r["name"] = r["name"].split()[0][0] + ". " + "*" * 5
        if "phone" in r and r["phone"]:
            r["phone"] = r["phone"][:4] + "*****" + r["phone"][-2:]
    return r


@tool("D3", "Customer 360", "L5.1", "read")
def customer360(ctx, customer_id):
    """Customer profile joined with branch."""
    c = db.one("SELECT c.*, b.name branch FROM customers c JOIN branches b ON b.branch_id=c.branch_id "
               "WHERE customer_id=?", (customer_id,))
    return mask(ctx, c) if c else None


@tool("D9", "Consent and purpose check", "L5.1", "read")
def consent(ctx, customer_id, purpose):
    """Checks recorded consent for a purpose (alternative data or marketing)."""
    c = db.one("SELECT consent_alt_data, consent_marketing, dnd FROM customers WHERE customer_id=?", (customer_id,))
    if purpose == "alternative_data":
        return {"purpose": purpose, "consented": bool(c["consent_alt_data"])}
    return {"purpose": purpose, "consented": bool(c["consent_marketing"]) and not c["dnd"], "dnd": bool(c["dnd"])}


def _alt_precheck(ctx, app_id):
    a = db.one("SELECT c.consent_alt_data FROM loan_applications a JOIN customers c ON c.customer_id=a.customer_id "
               "WHERE app_id=?", (app_id,))
    return None if a and a["consent_alt_data"] else "No recorded consent to use alternative data for this customer (NDPA 2023 purpose limitation)."


@tool("D8", "Alternative data", "L5.1", "read", precheck=_alt_precheck)
def alt_data(ctx, app_id):
    """Consented utility, airtime and mobile money data for an applicant."""
    a = db.one("SELECT utility_ontime_ratio, airtime_monthly_avg, mobile_money_inflow_avg FROM loan_applications "
               "WHERE app_id=?", (app_id,))
    return a


@tool("D4", "Loan application record", "L5.1", "read")
def application(ctx, app_id=None):
    """Application joined with customer profile, existing loans and 6-month average inflow.
    Without an ID it returns the oldest pending application."""
    if not app_id:
        app_id = db.one("SELECT app_id FROM loan_applications WHERE status='pending' ORDER BY applied_on LIMIT 1")["app_id"]
    rec = F.applicant_record(app_id)
    if rec:
        rec["customer"] = mask(ctx, rec["customer"])
    return rec


def _quality_issues_app(a):
    issues = []
    if a.get("declared_income") in (None, 0) or (isinstance(a.get("declared_income"), float) and np.isnan(a["declared_income"])):
        issues.append({"field": "declared_income", "issue": "missing", "severity": "blocking"})
    if a.get("amount") is not None and a["amount"] <= 0:
        issues.append({"field": "amount", "issue": f"non-positive value {a['amount']:,.0f}", "severity": "blocking"})
    u = a.get("utility_ontime_ratio")
    if u is not None and not (0 <= u <= 1):
        issues.append({"field": "utility_ontime_ratio", "issue": f"out of range ({u})", "severity": "blocking"})
    return issues


@tool("D14", "Data quality and profiling", "L5.1", "compute")
def data_quality(ctx, target, ids=None, window_days=None):
    """Completeness, validity and duplicate checks. Returns issues and a 0-1 quality score."""
    if target == "application":
        a = db.one("SELECT * FROM loan_applications WHERE app_id=?", (ids,))
        issues = _quality_issues_app(a)
        dup = db.one("SELECT COUNT(*) n FROM loan_applications WHERE customer_id=? AND status='pending'",
                     (a["customer_id"],))["n"]
        if dup > 1:
            issues.append({"field": "app_id", "issue": f"{dup} pending applications for the same customer", "severity": "warning"})
        score = 1.0 - 0.5 * sum(i["severity"] == "blocking" for i in issues) - 0.1 * sum(i["severity"] == "warning" for i in issues)
        return {"issues": issues, "score": max(0.0, round(score, 2))}
    if target == "transactions":
        since = str((AS_OF - pd.Timedelta(days=window_days or 7)).date())
        tot = db.one("SELECT COUNT(*) n FROM transactions WHERE ts>=?", (since,))["n"]
        distinct = db.one("SELECT COUNT(DISTINCT txn_id) n FROM transactions WHERE ts>=?", (since,))["n"]
        neg = db.one("SELECT COUNT(*) n FROM transactions WHERE ts>=? AND amount<=0", (since,))["n"]
        issues = []
        if tot > distinct:
            issues.append({"field": "txn_id", "issue": f"{tot - distinct} duplicate transaction rows", "severity": "warning"})
        if neg:
            issues.append({"field": "amount", "issue": f"{neg} non-positive amounts", "severity": "warning"})
        return {"rows": tot, "distinct": distinct, "issues": issues,
                "score": round(distinct / tot if tot else 1, 3), "action": "duplicates removed by txn_id before scoring"}
    if target == "cohort":
        fr = ids
        miss = int(fr[F.CHURN_FEATURES].isna().sum().sum())
        issues = [{"field": "features", "issue": f"{miss} missing feature values", "severity": "warning"}] if miss else []
        return {"rows": len(fr), "issues": issues, "score": 1.0 if not miss else 0.9}
    if target == "atm":
        h = pd.DataFrame(db.query("SELECT atm_id, MAX(date) last FROM atm_cash GROUP BY atm_id"))
        stale = h[h["last"] < str(AS_OF.date())]
        issues = [{"field": "atm_cash", "issue": f"{len(stale)} ATMs with stale data", "severity": "warning"}] if len(stale) else []
        return {"rows": len(h), "issues": issues, "score": 1.0 - .05 * len(stale)}
    return {"issues": [], "score": 1.0}


def _metric_allowed(ctx, metric, **_):
    ent = P.body("entitlements").get(ctx.role, {"metrics": []})
    ok = ent["metrics"] == "*" or metric in ent["metrics"]
    return None if ok else f"Role {ctx.role} is not entitled to the metric '{metric}'."


QUARTERS = {"q1": ("2026-01-01", "2026-03-31"), "q2": ("2026-04-01", "2026-06-30"), "q3": ("2026-07-01", "2026-09-30")}


def period_range(period):
    if not period:
        return None
    p = period.lower()
    if p in ("last quarter", "this quarter", "q3", "q3 2026"):
        return QUARTERS["q3"]
    if p in ("q2", "q2 2026", "previous quarter"):
        return QUARTERS["q2"]
    if p in ("last month", "september", "sep"):
        return ("2026-09-01", "2026-09-30")
    if p in ("august", "aug"):
        return ("2026-08-01", "2026-08-31")
    m = re.match(r"last (\d+) days", p)
    if m:
        return (str((AS_OF - pd.Timedelta(days=int(m.group(1)))).date()), "2026-09-30")
    return None


def build_semantic_sql(metric, dimensions=(), filters=None, period=None):
    sl = P.body("semantic_layer")
    md = sl["metrics"][metric]
    fact = md["fact"]
    sel, groups = [], []
    for d in dimensions or []:
        col = sl["dimensions"].get(d, {}).get(fact)
        if not col:
            raise ValueError(f"Dimension '{d}' is not defined for {md['label']}.")
        sel.append(f"{col} AS {d}")
        groups.append(col)
    where, params = [], []
    for k, v in (filters or {}).items():
        col = sl["dimensions"].get(k, {}).get(fact)
        if not col:
            raise ValueError(f"Filter '{k}' is not defined for {md['label']}.")
        neg = isinstance(v, dict) and "not" in v
        vals = v["not"] if neg else v
        vals = vals if isinstance(vals, list) else [vals]
        if not vals:
            continue
        ph = ", ".join("?" for _ in vals)
        where.append(f"{col} {'NOT IN' if neg else 'IN'} ({ph})")
        params += vals
    rng = period_range(period)
    tcol = sl["time_columns"].get(fact)
    if rng and tcol:
        where.append(f"{tcol}::date BETWEEN ? AND ?")
        params += list(rng)
    src = sl["facts"][fact]
    sql = f"SELECT {', '.join(sel + [md['sql'] + ' AS value'])} FROM {src}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if groups:
        order = "month" if "month" in (dimensions or []) else "value DESC"
        sql += " GROUP BY " + ", ".join(groups) + " ORDER BY " + order
    return sql, params, md, rng, bool(rng and not tcol)


@tool("D13", "Semantic query", "L5.1", "read", policy="semantic_layer", precheck=_metric_allowed)
def semantic_query(ctx, metric, dimensions=(), filters=None, period=None):
    """Governed query: SQL is generated only from metric and dimension definitions."""
    sql, params, md, rng, snapshot = build_semantic_sql(metric, dimensions, filters, period)
    rows = db.query(sql, params)
    for r in rows:
        if r.get("value") is not None:
            r["value"] = round(r["value"], 2)
    return {"metric": metric, "label": md["label"], "unit": md["unit"], "definition": md["definition"],
            "rows": rows, "sql": sql, "params": params, "period": rng,
            "note": "Loans are a point-in-time snapshot, so the period filter does not apply." if snapshot else None}


@tool("D5", "Branch register", "L5.1", "read")
def branches(ctx):
    """Branch IDs, names, cities and regions."""
    return {b["branch_id"]: b for b in db.query("SELECT * FROM branches")}


@tool("D1", "Transaction window", "L5.1", "read")
def txn_window(ctx, days=7):
    """Distinct transactions in the monitoring window (replayed stream)."""
    since = str((AS_OF - pd.Timedelta(days=days)).date())
    tx = pd.DataFrame(db.query("SELECT DISTINCT t.txn_id, t.customer_id, t.ts, t.amount, t.direction, t.channel, "
                               "t.counterparty, c.monthly_income FROM transactions t JOIN customers c "
                               "ON c.customer_id=t.customer_id WHERE t.ts>=? ORDER BY t.txn_id", (since,)))
    return tx


@tool("D12", "Cash logistics data", "L5.1", "read")
def cash_data(ctx):
    """ATM register and 180 days of withdrawals and balances."""
    return {"atms": pd.DataFrame(db.query("SELECT * FROM atms")),
            "history": pd.DataFrame(db.query("SELECT * FROM atm_cash ORDER BY date"))}


@tool("D16", "Churn cohort", "L5.1", "read")
def churn_cohort(ctx, city=None, segment=None, min_tenure=6):
    """Active customers matching the cohort definition, with point-in-time churn features (D6)."""
    fr = F.churn_frame()
    if city:
        fr = fr[fr.city == city]
    if segment:
        fr = fr[fr.segment == segment]
    return fr[fr.tenure_months >= min_tenure]


# ═════════════════════════ L5.2 MODEL ═════════════════════════

def credit_inputs(rec, amount=None, tenor=None, income=None, extra_installment=0.0):
    a, c = rec["application"], rec["customer"]
    pol = P.body("loan_policy")
    amount = a["amount"] if amount is None else amount
    tenor = a["tenor_months"] if tenor is None else tenor
    income = a["declared_income"] if income is None else income
    inst = amount * (1 + pol["interest_rate_annual"] * tenor / 12) / tenor if tenor else 0
    existing = sum(l["monthly_installment"] for l in rec["existing_loans"] if l["status"] != "closed") + extra_installment
    dti = (existing + inst) / income if income else None
    return {"amount": amount, "tenor_months": tenor, "monthly_income": income, "new_installment": round(inst, 2),
            "existing_installments": round(existing, 2), "dti": round(dti, 4) if dti is not None else None,
            "tenure_months": c["tenure_months"], "utility_ontime_ratio": a["utility_ontime_ratio"],
            "airtime_monthly_avg": a["airtime_monthly_avg"], "mobile_money_inflow_avg": a["mobile_money_inflow_avg"],
            "bureau_score": a["bureau_score"],
            "prior_default": int(any(l["dpd"] >= 90 for l in rec["existing_loans"])),
            "informal_income": int(c["employer_type"] == "informal")}


def _no_protected(ctx, inputs, **_):
    prot = set(P.body("fairness")["protected_attributes"])
    used = prot.intersection(inputs.keys())
    return f"Protected attribute(s) {sorted(used)} cannot be model inputs." if used else None


@tool("M10", "Alternative-data credit model", "L5.2", "compute", model="credit_pd-v1", precheck=_no_protected)
def credit_pd(ctx, inputs):
    """Probability of default for one applicant, with exact additive attributions (X1)."""
    fr = F.credit_frame(pd.DataFrame([inputs]))
    pd_, z, m = M.credit_score(fr)
    return {"pd": round(float(pd_[0]), 4), "attr": X.linear_attributions(m, z[0]), "model": m["version"],
            "auc": m["metrics"]["auc_holdout"]}


@tool("M8", "Churn model", "L5.2", "compute", model="churn-v1")
def churn_model(ctx, cohort):
    """90-day churn probability per customer, with attributions."""
    p, z, m = M.churn_score(cohort)
    attrs = [X.linear_attributions(m, z[i]) for i in range(len(cohort))]
    return {"p": p, "attrs": attrs, "drivers": [X.churn_drivers(a) for a in attrs], "model": m["version"], "auc": m["metrics"]["auc_holdout"],
            "base_rate": m["metrics"]["base_rate"]}


@tool("M1", "Streaming anomaly detection", "L5.2", "compute", model="anomaly-v1")
def anomaly(ctx, tx):
    """Isolation-forest anomaly score per transaction (0-1)."""
    s, m = M.anomaly_score(tx)
    return s


@tool("M4", "Typology detectors", "L5.2", "compute", policy="aml_policy")
def typology(ctx, tx):
    """Rule detectors for structuring, mule pass-through and card fraud, from AML policy."""
    pol = P.body("aml_policy")
    tx = tx.copy()
    tx["ts"] = pd.to_datetime(tx.ts)
    hits = []
    thr = pol["cash_reporting_threshold"]
    s = pol["structuring"]
    cash = tx[(tx.channel == "branch_cash") & (tx.direction == "credit") &
              (tx.amount >= s["band_low_ratio"] * thr) & (tx.amount < thr)]
    for cid, g in cash.groupby("customer_id"):
        g = g.sort_values("ts")
        if len(g) >= s["min_deposits"] and (g.ts.max() - g.ts.min()).days <= s["window_days"]:
            hits.append({"customer_id": cid, "typology": "structuring", "txn_ids": list(g.txn_id),
                         "total": float(g.amount.sum()),
                         "conditions": [f"{len(g)} cash deposits between {money(s['band_low_ratio'] * thr)} and {money(thr)}",
                                        f"within {(g.ts.max() - g.ts.min()).days + 1} day(s) (limit {s['window_days']})",
                                        f"combined {money(g.amount.sum())} exceeds the reporting threshold"]})
    mu = pol["mule"]
    inbound = tx[(tx.channel == "transfer") & (tx.direction == "credit")]
    for cid, g_all in inbound.groupby("customer_id"):
        g_all = g_all.sort_values("ts").reset_index(drop=True)
        if len(g_all) < mu["min_inbound"]:
            continue
        # sliding window: the densest burst of inbound transfers within window_hours
        best, j = None, 0
        for i in range(len(g_all)):
            while g_all.ts[i] - g_all.ts[j] > pd.Timedelta(hours=mu["window_hours"]):
                j += 1
            if i - j + 1 >= mu["min_inbound"] and (best is None or i - j > best[1] - best[0]):
                best = (j, i)
        if best is None:
            continue
        g = g_all.iloc[best[0]:best[1] + 1]
        span_h = (g.ts.max() - g.ts.min()).total_seconds() / 3600
        out = tx[(tx.customer_id == cid) & (tx.direction == "debit") & (tx.ts >= g.ts.min()) &
                 (tx.ts <= g.ts.max() + pd.Timedelta(hours=24))].amount.sum()
        if out >= mu["passthrough_ratio"] * g.amount.sum():
            hits.append({"customer_id": cid, "typology": "mule_pass_through", "txn_ids": list(g.txn_id),
                         "total": float(g.amount.sum()),
                         "conditions": [f"{len(g)} inbound transfers from distinct senders in {span_h:.1f} hours",
                                        f"{out / g.amount.sum():.0%} moved out within 24 hours (threshold {mu['passthrough_ratio']:.0%})"]})
    cf = pol["card_fraud"]
    pos = tx[(tx.channel == "POS") & (tx.direction == "debit")]
    pos = pos[(pos.amount >= cf["multiple_of_income"] * pos.monthly_income) &
              (pos.ts.dt.hour.between(cf["night_start"], cf["night_end"]))]
    for r in pos.itertuples():
        hits.append({"customer_id": r.customer_id, "typology": "card_fraud", "txn_ids": [r.txn_id], "total": float(r.amount),
                     "conditions": [f"POS spend {money(r.amount)} is {r.amount / r.monthly_income:.1f} times monthly income",
                                    f"at {r.ts:%H:%M}, inside the night window", f"merchant: {r.counterparty}"]})
    return hits


@tool("M5", "Risk scoring ensemble", "L5.2", "compute", policy="aml_policy")
def risk_score(ctx, hits, tx, anomaly_scores):
    """Fuses typology hits and anomaly scores into a customer risk score."""
    base = {"structuring": .85, "mule_pass_through": .9, "card_fraud": .82}
    tx = tx.assign(anom=anomaly_scores)
    an = tx.groupby("customer_id").anom.max()
    out = {}
    for h in hits:
        s = base[h["typology"]] * .8 + .2 * float(an.get(h["customer_id"], 0))
        if h["customer_id"] not in out or s > out[h["customer_id"]]["score"]:
            out[h["customer_id"]] = {"score": round(s, 3), "typology": h["typology"], "hit": h}
    # anomaly-only alerts
    for cid, a in an.items():
        if cid not in out and a >= .995:
            row = tx[(tx.customer_id == cid)].sort_values("anom").iloc[-1]
            out[cid] = {"score": round(.45 + .1 * a, 3), "typology": "anomaly", "hit": {
                "customer_id": cid, "typology": "anomaly", "txn_ids": [row.txn_id], "total": float(row.amount),
                "conditions": [f"anomaly score {a:.2f} on {row.channel} {row.direction} of {money(row.amount)}"]}}
    return out


@tool("M14", "Cash demand forecaster", "L5.2", "compute", model="cash_demand-v1")
def cash_forecast(ctx, history, atm_id, days):
    """Per-ATM withdrawals forecast with a 90% interval."""
    h = history[history.atm_id == atm_id]
    fc, meta = M.cash_forecast(h, AS_OF + pd.Timedelta(days=1), days)
    bt, _ = M.cash_forecast(h.iloc[:-14], pd.Timestamp(h.iloc[-14]["date"]), 14)
    actual = h.iloc[-14:].withdrawals.values
    cover = float(np.mean((actual >= bt.low.values) & (actual <= bt.high.values)))
    mape = float(np.mean(np.abs(actual - bt.forecast.values) / np.maximum(actual, 1)))
    return {"forecast": fc, "meta": meta, "backtest": {"coverage": round(cover, 2), "mape": round(mape, 3)},
            "balance": float(h.iloc[-1].closing_balance)}


@tool("M16", "LLM text-to-query", "L5.2", "compute", model="llm")
def text_to_query(ctx, question):
    """LLM maps a question to a metric, dimensions, filters and period from the semantic layer."""
    sl = P.body("semantic_layer")
    sys = ("You translate bank questions into a query intent over a governed semantic layer. "
           f"Metrics: {list(sl['metrics'])}. Dimensions: {list(sl['dimensions'])}. "
           "Cities: Lagos, Abuja, Port Harcourt, Kano, Ibadan, Enugu, Kaduna. Segments: Retail, SME. "
           "Periods: 'last quarter', 'last month', 'last N days' or null. "
           "A filter value is a string (is), a list (is one of) or {\"not\": value or list} (is not). "
           "Read negation carefully: 'not Lagos', 'outside Lagos' and 'except Lagos' all mean {\"not\": \"Lagos\"}. "
           'Return {"metric": ..., "dimensions": [...], "filters": {"city": ...}, "period": ...}. '
           'If no metric fits, return {"metric": null, "clarify": "<question to ask the user>"}.')
    out = llm.chat_json(sys, question)
    if out and (out.get("metric") in sl["metrics"] or out.get("clarify")):
        out["source"] = "llm"
        return out
    return None


@tool("M17", "LLM summariser", "L5.2", "compute", model="llm")
def summarise(ctx, instruction, facts):
    """LLM drafts narrative text from verified facts only."""
    sys = ("You write short, plain-English banking summaries for staff. Use only the facts given. "
           "Do not invent numbers. Short active sentences. No markdown symbols.")
    return llm.chat(sys, f"{instruction}\n\nFACTS:\n{json.dumps(facts, default=str)[:6000]}", max_tokens=600)


# ═════════════════════════ L5.3 XAI ═════════════════════════

@tool("X2", "Reason codes", "L5.3", "compute")
def reasons(ctx, attr):
    """Plain-language adverse and supporting reasons from attributions."""
    return X.reason_codes(attr)


@tool("X3", "Counterfactual recourse", "L5.3", "compute", policy="loan_policy")
def counterfactual(ctx, rec, rules_fn):
    """Smallest actionable changes that would move the decision to approve."""
    pol = P.body("loan_policy")
    a = rec["application"]
    opts = []
    if a["amount"] and a["amount"] > 0 and a["declared_income"]:
        for f in np.arange(.95, .1, -.05):
            amt = round(a["amount"] * f, -3)
            res = rules_fn(rec, amount=amt)
            if res["recommendation"] == "approve":
                opts.append({"change": "reduce amount", "to": amt,
                             "text": f"Reduce the amount to {money(amt)} ({(1 - f):.0%} less) at the same tenor."})
                break
        max_t = next(r for r in pol["rules"] if r["id"] == "R7")["params"]["max_tenor_months"]
        for t in [12, 18, 24, 30, 36, 48]:
            if t <= a["tenor_months"] or t > max_t:
                continue
            res = rules_fn(rec, tenor=t)
            if res["recommendation"] == "approve":
                opts.append({"change": "extend tenor", "to": t, "text": f"Extend the tenor to {t} months."})
                break
        cov = next(r for r in pol["rules"] if r["id"] == "R5")["params"]["collateral_waiver_coverage"]
        res = rules_fn(rec, collateral=a["amount"] * cov)
        if res["recommendation"] == "approve" and (a["collateral_value"] or 0) < a["amount"] * cov:
            opts.append({"change": "add collateral", "to": round(a["amount"] * cov, -3),
                         "text": f"Pledge collateral worth at least {money(a['amount'] * cov)}."})
    return opts


@tool("X9", "Query transparency", "L5.3", "compute", policy="semantic_layer")
def transparency(ctx, result):
    """SQL, metric definition, filters and data freshness behind an answer."""
    return {"sql": result["sql"], "params": result["params"], "definition": result["definition"],
            "as_of": P.body("semantic_layer")["as_of"], "period": result["period"]}


@tool("X10", "Fairness diagnostics", "L5.3", "compute", policy="fairness")
def fairness(ctx, outcomes, positive_col):
    """Disparate impact across monitored attributes (four-fifths rule)."""
    pol = P.body("fairness")
    out = {a: X.disparate_impact(outcomes, a, positive_col) for a in pol["monitored_attributes"] if a in outcomes}
    out["threshold"] = pol["min_disparate_impact"]
    out["protected"] = pol["protected_attributes"]
    # pass/fail is judged on protected attributes; other monitored attributes are reported for review
    out["pass"] = all(out[a]["ratio"] >= pol["min_disparate_impact"] for a in pol["protected_attributes"] if a in out)
    out["watch"] = [a for a in pol["monitored_attributes"] if a in out and a not in pol["protected_attributes"]
                    and out[a]["ratio"] < pol["min_disparate_impact"]]
    return out


@tool("X11", "Explanation quality checker", "L5.3", "compute", model="credit_pd-v1")
def explanation_quality(ctx, attr, inputs):
    """Faithfulness (attributions add up to the score) and stability of the top reasons under 2% input noise."""
    fr = F.credit_frame(pd.DataFrame([inputs]))
    m = M.load("credit_pd")
    stab = X.stability(None, fr, lambda f: X.linear_attributions(m, m["scaler"].transform(f[m["features"]])[0]))
    return {"faithful": attr["faithful"], "faithfulness_error": attr["faithfulness_error"], "stability": stab}


@tool("X12", "Portfolio fairness simulation", "L5.3", "compute", policy="fairness", model="credit_pd-v1")
def portfolio_fairness(ctx):
    """Applies the current policy and model to every pending application and tests approval rates by group (X10)."""
    key = (P.get("loan_policy")[1], P.get("fairness")[1])
    if key in _PF_CACHE:
        return _PF_CACHE[key]
    rows = []
    for a in db.query("SELECT app_id FROM loan_applications"):
        rec = F.applicant_record(a["app_id"])
        if not rec["application"]["declared_income"] or rec["application"]["amount"] <= 0:
            continue
        if not rec["customer"]["consent_alt_data"]:
            for k in ("utility_ontime_ratio", "airtime_monthly_avg", "mobile_money_inflow_avg"):
                rec["application"][k] = None
        x = credit_inputs(rec)
        x["utility_ontime_ratio"] = 0.77 if x["utility_ontime_ratio"] is None else x["utility_ontime_ratio"]
        x["airtime_monthly_avg"] = 6000.0 if x["airtime_monthly_avg"] is None else x["airtime_monthly_avg"]
        if x["mobile_money_inflow_avg"] is None:
            x["mobile_money_inflow_avg"] = 0.65 * x["monthly_income"]
        p, _, _ = M.credit_score(F.credit_frame(pd.DataFrame([x])))
        r = evaluate_rules(rec, float(p[0]))
        rows.append({"gender": rec["customer"]["gender"], "region": rec["customer"]["region"],
                     "approve": int(r["recommendation"] == "approve")})
    _PF_CACHE[key] = fairness(ctx, pd.DataFrame(rows), "approve")
    return _PF_CACHE[key]


_PF_CACHE = {}


# ═════════════════════════ L5.4 COGNITIVE ═════════════════════════

def evaluate_rules(rec, pd_value=None, amount=None, tenor=None, income=None, collateral=None, extra_installment=0.0):
    """Deterministic loan policy engine (C3). Returns each rule with value, threshold and evidence."""
    pol = P.body("loan_policy")
    a, c = rec["application"], rec["customer"]
    x = credit_inputs(rec, amount, tenor, income, extra_installment)
    collateral = a["collateral_value"] if collateral is None else collateral
    R = {r["id"]: r for r in pol["rules"]}
    res = []

    def add(rid, ok, value, threshold, evidence):
        r = R[rid]
        if r["enabled"]:
            res.append({"id": rid, "name": r["name"], "kind": r["kind"], "pass": bool(ok), "value": value,
                        "threshold": threshold, "evidence": evidence, "text": r["text"]})

    p = R["R1"]["params"]
    age_mat = c["age"] + x["tenor_months"] / 12
    add("R1", c["age"] >= p["min_age"] and age_mat <= p["max_age_at_maturity"], f"{c['age']} now, {age_mat:.1f} at maturity",
        f"{p['min_age']} to {p['max_age_at_maturity']}", "customers.age, application tenor")
    add("R2", c["bvn_verified"] == 1, "verified" if c["bvn_verified"] else "not verified", "verified", "customers.bvn_verified")
    add("R3", c["tenure_months"] >= R["R3"]["params"]["min_tenure_months"], f"{c['tenure_months']} months",
        f">= {R['R3']['params']['min_tenure_months']} months", "customers.tenure_months")
    mx = R["R4"]["params"]["max_dti"]
    add("R4", x["dti"] is not None and x["dti"] <= mx, f"{x['dti']:.1%}" if x["dti"] is not None else "unknown",
        f"<= {mx:.0%}", f"({money(x['existing_installments'])} existing + {money(x['new_installment'])} new) / {money(x['monthly_income'] or 0)} income")
    p5 = R["R5"]["params"]
    inflow = rec["avg_monthly_inflow"]
    waived = collateral >= p5["collateral_waiver_coverage"] * x["amount"]
    add("R5", waived or x["amount"] <= p5["max_multiple_of_inflow"] * inflow,
        f"{x['amount'] / inflow:.1f}x inflow" + (" (collateral waiver)" if waived else "") if inflow else "no inflow",
        f"<= {p5['max_multiple_of_inflow']}x or collateral >= {p5['collateral_waiver_coverage']:.0%}",
        f"amount {money(x['amount'])}, 6-month average inflow {money(inflow)}, collateral {money(collateral)}")
    worst = max([l["dpd"] for l in rec["existing_loans"]] or [0])
    add("R6", worst <= R["R6"]["params"]["max_dpd"], f"{worst} days past due", f"<= {R['R6']['params']['max_dpd']}", "loans.dpd")
    add("R7", x["tenor_months"] <= R["R7"]["params"]["max_tenor_months"], f"{x['tenor_months']} months",
        f"<= {R['R7']['params']['max_tenor_months']}", "application.tenor_months")
    if a["bureau_score"] is None:
        ok = bool(c["consent_alt_data"]) and (a["utility_ontime_ratio"] or 0) >= R["R8"]["params"]["min_utility_ontime"]
        u = a["utility_ontime_ratio"]
        add("R8", ok, f"thin file, consent {'yes' if c['consent_alt_data'] else 'no'}, utility on-time "
                      + (f"{u:.0%}" if u is not None else "not available"),
            f"consent and >= {R['R8']['params']['min_utility_ontime']:.0%}", "D9 consent, D8 alternative data")
    else:
        add("R8", True, f"bureau score {a['bureau_score']:.0f}", "not applicable", "credit bureau")
    if pd_value is not None:
        p9 = R["R9"]["params"]
        add("R9", pd_value < p9["approve_below"], f"{pd_value:.1%}",
            f"< {p9['approve_below']:.0%} approve, > {p9['decline_above']:.0%} decline", "M10 credit model")
    hard_fail = [r for r in res if not r["pass"] and r["kind"] == "hard"]
    soft_fail = [r for r in res if not r["pass"] and r["kind"] == "soft"]
    if pd_value is not None and R["R9"]["enabled"] and pd_value > R["R9"]["params"]["decline_above"]:
        # above the decline band R9 acts as a hard rule: list it once, as a hard failure
        r9 = next(r for r in res if r["id"] == "R9")
        soft_fail = [r for r in soft_fail if r["id"] != "R9"]
        if r9 not in hard_fail:
            hard_fail.append(r9)
    rec_ = "decline" if hard_fail else ("refer" if soft_fail else "approve")
    return {"rules": res, "recommendation": rec_, "inputs": x,
            "failed": [r["id"] for r in hard_fail + soft_fail]}


@tool("C3", "Eligibility and policy engine", "L5.4", "compute", policy="loan_policy")
def policy_engine(ctx, rec, pd_value=None, **overrides):
    """Applies every enabled loan policy rule and returns a recommendation with evidence."""
    return evaluate_rules(rec, pd_value, **overrides)


@tool("C4", "Driver analysis", "L5.4", "compute")
def driver_analysis(ctx, cohort, flag_col):
    """Compares feature means between flagged and other customers with Welch t-tests."""
    out = []
    for f in F.CHURN_FEATURES:
        a, b = cohort[cohort[flag_col]][f], cohort[~cohort[flag_col]][f]
        if len(a) > 2 and len(b) > 2:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            out.append({"feature": F.CHURN_LABELS[f], "at_risk_mean": round(float(a.mean()), 3),
                        "others_mean": round(float(b.mean()), 3), "p_value": float(p)})
    return sorted(out, key=lambda r: r["p_value"])


@tool("C7", "Backtest and statistical tests", "L5.4", "compute")
def backtest(ctx, model_name):
    """Returns held-out validation metrics recorded in the model registry."""
    m = M.load(model_name)
    return {k: v for k, v in m["metrics"].items() if k != "holdout_ids"} | {"version": m["version"]}


def open_case_customers():
    return {r["customer_id"] for r in db.query("SELECT customer_id FROM ext_cases WHERE status='open'")}


@tool("C11", "Cross-agent conflict detector", "L5.4", "read")
def conflicts(ctx, customer_ids):
    """Flags customers with open financial-crime cases (anti-tipping-off)."""
    oc = open_case_customers()
    return [c for c in customer_ids if c in oc]


@tool("N5", "Contact suppression check", "L5.5", "read", policy="churn_policy")
def suppression(ctx, customer_ids):
    """Do-not-disturb, missing marketing consent and open-case suppression."""
    rows = pd.DataFrame(db.query("SELECT customer_id, dnd, consent_marketing FROM customers"))
    rows = rows[rows.customer_id.isin(customer_ids)].set_index("customer_id")
    oc = open_case_customers()
    out = {}
    for cid in customer_ids:
        r = rows.loc[cid]
        why = []
        if r.dnd:
            why.append("do-not-disturb")
        if not r.consent_marketing:
            why.append("no marketing consent")
        if cid in oc and P.body("aml_policy")["anti_tipping_off"]:
            why.append("open financial-crime case")
        out[cid] = why
    return out


@tool("C5", "Optimisation solver", "L5.4", "compute")
def allocate_offers(ctx, candidates, budget):
    """Greedy budget allocation by expected retained value per naira of offer cost."""
    cand = sorted(candidates, key=lambda c: -c["p"] * c["value"] / c["cost"])
    chosen, spent = [], 0
    for c in cand:
        if spent + c["cost"] <= budget:
            chosen.append(c)
            spent += c["cost"]
    return {"chosen": chosen, "spent": spent, "budget": budget, "not_funded": len(cand) - len(chosen)}


CITY_VALUES = ["Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan", "Enugu", "Kaduna"]
REGION_VALUES = ["South West", "North Central", "South South", "North West", "South East"]
# words that turn "Lagos" into "not Lagos": not, outside, except, excluding, other than, apart from, besides, non-
NEG = (r"\b(?:not|outside(?: of)?|except(?: for)?|excluding|exclude|without|other than|apart from|besides|non)"
       r"[\s-]+(?:in\s+|the\s+|from\s+|city\s+(?:of\s+)?|branches\s+in\s+)?")
COMPLEX_WORDS = r"\b(not|except|excluding|exclude|outside|without|other than|apart from|besides|only|between|versus|vs|compare|non)\b"


def filter_value(inc, exc):
    """One included value, several included values, or excluded values."""
    if exc:
        return {"not": exc if len(exc) > 1 else exc[0]}
    if inc:
        return inc if len(inc) > 1 else inc[0]
    return None


def describe_intent(intent):
    """The interpretation in plain words, for the person who asked to check."""
    sl = P.body("semantic_layer")
    md = sl["metrics"].get(intent.get("metric") or "", {})
    parts = [md.get("label", intent.get("metric") or "unknown measure")]
    if intent.get("dimensions"):
        parts.append("by " + " and ".join(intent["dimensions"]))
    for k, v in (intent.get("filters") or {}).items():
        if isinstance(v, dict) and "not" in v:
            vals = v["not"] if isinstance(v["not"], list) else [v["not"]]
            parts.append(f"{k} is not {vals[0]}" if len(vals) == 1 else
                         f"{k} is neither {vals[0]} nor {vals[1]}" if len(vals) == 2 else
                         f"{k} is none of {', '.join(vals[:-1])} and {vals[-1]}")
        else:
            vals = v if isinstance(v, list) else [v]
            parts.append(f"{k} is {' or '.join(vals)}")
    parts.append(f"period {intent.get('period')}" if intent.get("period") else "all available data")
    return ", ".join(parts) + "."


@tool("C10", "Dialogue and intent manager", "L5.4", "compute", policy="semantic_layer")
def intent_rules(ctx, question):
    """Rule-based fallback parser for questions over the semantic layer."""
    q = question.lower()
    sl = P.body("semantic_layer")
    syn = {"delinquen": "delinquency_rate", "arrear": "delinquency_rate", "npl": "npl_ratio", "non-performing": "npl_ratio",
           "loan book": "loan_book", "outstanding": "loan_book", "number of loans": "loan_count", "how many loans": "loan_count",
           "customers": "customer_count", "income": "avg_income", "deposit": "deposit_inflow", "inflow": "deposit_inflow",
           "transactions": "txn_count", "complaint": "complaint_count", "approval": "approval_rate",
           "atm": "atm_withdrawals", "withdrawal": "atm_withdrawals"}
    # "SME customers" or "retail customers" names a segment, not the customer-count measure
    qm = re.sub(r"\b(retail|sme|non-retail|non-sme)\s+customers\b", r"\1", q)
    found = [(qm.index(k), -len(k), m) for k, m in syn.items() if k in qm]
    metric = min(found)[2] if found else None   # the measure named first in the question
    dims = [d for d in ["branch", "city", "region", "segment", "product", "channel", "month", "topic"]
            if re.search(rf"\b(by|per|across|each)\s+(\w+\s+)?{d}", q) or f"by {d}" in q]
    filters = {}
    for key, values in (("city", CITY_VALUES), ("segment", ["Retail", "SME"]), ("region", REGION_VALUES)):
        inc, exc = [], []
        for v in values:
            if not re.search(rf"\b{re.escape(v.lower())}\b", q):
                continue
            (exc if re.search(NEG + re.escape(v.lower()) + r"\b", q) else inc).append(v)
        # "outside Lagos and Abuja": a value joined to an excluded one is excluded too
        changed = True
        while changed:
            changed = False
            for v in list(inc):
                if any(re.search(rf"\b{re.escape(e.lower())}\s*(?:,|and|or|nor)\s*(?:the\s+)?{re.escape(v.lower())}\b", q) for e in exc):
                    inc.remove(v); exc.append(v); changed = True
        f = filter_value(inc, exc)
        if f is not None:
            filters[key] = f
    if "branch" in q and not dims:
        dims = ["branch"]
    period = None
    for p in ["last quarter", "this quarter", "q3", "q2", "last month", "september", "august"]:
        if p in q:
            period = p
            break
    m = re.search(r"last (\d+) days", q)
    if m:
        period = m.group(0)
    if "trend" in q or "monthly" in q or "over time" in q:
        dims = dims + ["month"] if "month" not in dims else dims
    out = {"metric": metric, "dimensions": dims, "filters": filters, "period": period, "source": "rules"}
    if not metric:
        out["clarify"] = ("Which measure do you need? For example: delinquency rate, NPL ratio, loan book, deposit inflow, "
                          "complaints or ATM withdrawals.")
    return out


# ═════════════════════════ L5.5 CONNECTION (external effects) ═════════════════════════

def _los_env(ctx, app_id, decision, amount, **_):
    e = P.body("loan_policy")["auto_decision_envelope"]
    return (e["enabled"] and decision == "approve" and amount <= e["max_amount"]
            and ctx.state.get("credit", {}).get("pd", 1) <= e["max_pd"])


def _los_pre(ctx, app_id, decision, amount, decided_by=None, **_):
    if decision == "approve" and ctx.state.get("credit", {}).get("hard_fail") and not ctx.state.get("credit", {}).get("exception_approved"):
        return "Approval of an application that fails a hard policy rule needs an approved credit exception."
    return None


@tool("N7", "Loan origination system", "L5.5", "external", policy="loan_policy", envelope=_los_env, precheck=_los_pre,
      preview=lambda ctx, app_id, decision, amount, **k: {"system": "Loan origination (simulated)", "app_id": app_id,
                                                         "write": decision, "amount": money(amount)},
      key=lambda a: f"{a['app_id']}:{a['decision']}", route="credit_decision", ikey=lambda a: a["app_id"])
def write_decision(ctx, app_id, decision, amount, reason_codes, decided_by):
    """Writes the final credit decision and reason codes."""
    db.execute("INSERT INTO ext_loan_decisions (ts, run_id, app_id, decision, amount, reason_codes, decided_by, policy_version) "
               "VALUES (?,?,?,?,?,?,?,?)", (db.now(), ctx.run_id, app_id, decision, amount, json.dumps(reason_codes),
                                            decided_by, ctx.policy_versions.get("loan_policy")))
    db.execute("UPDATE loan_applications SET status=? WHERE app_id=?", (decision + "d" if decision != "refer" else "referred", app_id))
    return {"written": True}


@tool("N2", "Case management", "L5.5", "internal_write", route="aml_case",
      ikey=lambda a: f"{a['customer_id']}:{a['typology']}")
def open_case(ctx, customer_id, typology, score, narrative):
    """Creates an alert case in the case queue."""
    cid = f"CASE-{customer_id}-{typology[:4].upper()}"
    db.execute("INSERT INTO ext_cases VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (case_id) DO UPDATE SET ts=EXCLUDED.ts, "
               "run_id=EXCLUDED.run_id, typology=EXCLUDED.typology, risk_score=EXCLUDED.risk_score, status=EXCLUDED.status, "
               "narrative=EXCLUDED.narrative",
               (cid, db.now(), ctx.run_id, customer_id, typology, score, "open", narrative))
    return {"case_id": cid}


def _hold_env(ctx, customer_id, txn_id, amount, hours):
    e = P.body("envelopes")["fraud_hold"]
    return e["enabled"] and amount <= e["max_amount"] and hours <= e["max_hours"]


@tool("N1", "Payment hold", "L5.5", "external", policy="envelopes", envelope=_hold_env, route="fraud_hold",
      preview=lambda ctx, customer_id, txn_id, amount, hours: {"system": "Payment switch (simulated)",
                                                                 "hold": txn_id, "amount": money(amount), "hours": hours},
      key=lambda a: a["txn_id"], ikey=lambda a: a["txn_id"])
def hold(ctx, customer_id, txn_id, amount, hours):
    """Places a temporary hold on a transaction."""
    db.execute("INSERT INTO ext_holds (ts, run_id, customer_id, txn_id, amount, hours, status, authorised_by) "
               "VALUES (?,?,?,?,?,?,?,?)", (db.now(), ctx.run_id, customer_id, txn_id, amount, hours, "active",
                                             ctx.state.get("last_approver", "envelope")))
    return {"held": txn_id}


@tool("N3", "Regulatory filing gateway (NFIU)", "L5.5", "external", route="str_filing",
      preview=lambda ctx, case_id, customer_id, body: {"system": "NFIU goAML gateway (simulated)", "case": case_id,
                                                       "report_excerpt": body[:400]},
      key=lambda a: a["case_id"], ikey=lambda a: a["case_id"])
def file_str(ctx, case_id, customer_id, body):
    """Files a suspicious transaction report. Always needs a named compliance officer (A1)."""
    db.execute("INSERT INTO ext_str_filings (ts, run_id, case_id, customer_id, filed_by, body) VALUES (?,?,?,?,?,?)",
               (db.now(), ctx.run_id, case_id, customer_id, ctx.state.get("last_approver", "?"), body))
    return {"filed": case_id}


def _campaign_pre(ctx, customer_id, offer, channel, cost):
    why = suppression(ctx, [customer_id])[customer_id]
    return f"Contact suppressed: {', '.join(why)}." if why else None


def _campaign_env(ctx, customer_id, offer, channel, cost):
    e = P.body("envelopes")["retention_campaign"]
    return e["enabled"] and ctx.state.get("campaign_spend", 0) + cost <= e["max_cost"]


@tool("N4", "CRM and campaign engine", "L5.5", "external", policy="churn_policy", precheck=_campaign_pre,
      envelope=_campaign_env, route="campaign",
      preview=lambda ctx, customer_id, offer, channel, cost: {
          "system": "CRM (simulated)", "offers_in_batch": len(ctx.get("strategy", {}).get("chosen", [])),
          "total_cost": money(ctx.get("strategy", {}).get("spent", 0)), "first_customer": customer_id,
          "first_offer": offer, "channel": channel},
      key=lambda a: "batch", ikey=lambda a: a["customer_id"], ikey_scope="run")
def send_offer(ctx, customer_id, offer, channel, cost):
    """Schedules a retention offer to a customer."""
    db.execute("INSERT INTO ext_campaigns (ts, run_id, customer_id, offer, channel, cost, authorised_by) VALUES (?,?,?,?,?,?,?)",
               (db.now(), ctx.run_id, customer_id, offer, channel, cost, ctx.state.get("last_approver", "envelope")))
    ctx.state["campaign_spend"] = ctx.state.get("campaign_spend", 0) + cost
    return {"scheduled": customer_id}


def _cash_env(ctx, atm_id, delivery_date, amount):
    e = P.body("envelopes")["cash_orders"]
    return e["enabled"] and amount <= e["max_order_per_atm"] and ctx.state.get("cash_total", 0) + amount <= e["max_total"]


@tool("N9", "Cash-in-transit ordering", "L5.5", "external", policy="envelopes", envelope=_cash_env, route="cash_order",
      preview=lambda ctx, atm_id, delivery_date, amount: {"system": "CIT vendor (simulated)", "atm": atm_id,
                                                          "delivery": delivery_date, "amount": money(amount)},
      key=lambda a: f"{a['atm_id']}:{a['delivery_date']}", ikey=lambda a: f"{a['atm_id']}:{a['delivery_date']}")
def cash_order(ctx, atm_id, delivery_date, amount):
    """Issues a replenishment order to the cash-in-transit vendor."""
    env = _cash_env(ctx, atm_id, delivery_date, amount)
    db.execute("INSERT INTO ext_cash_orders (ts, run_id, atm_id, delivery_date, amount, authorised_by, within_envelope) "
               "VALUES (?,?,?,?,?,?,?)", (db.now(), ctx.run_id, atm_id, delivery_date, amount,
                                           "envelope" if env else ctx.state.get("last_approver", "?"), int(env)))
    ctx.state["cash_total"] = ctx.state.get("cash_total", 0) + amount
    return {"ordered": atm_id, "amount": amount}


@tool("N12", "Report publishing", "L5.5", "internal_write", route="release")
def publish(ctx, title, body):
    """Publishes a released report to the BI portal (simulated)."""
    return {"published": title}


def catalogue():
    return [{k: v for k, v in t.items() if k in ("id", "name", "layer", "class", "policy", "model", "doc", "route")}
            | {"has_envelope": bool(t["envelope"]), "has_precheck": bool(t["precheck"])} for t in REG.values()]
