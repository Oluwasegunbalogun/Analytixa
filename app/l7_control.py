"""L7 Oversight control plane.

Deterministic services only. No component here is an LLM agent, so no agent
output can talk its way past a gate.

  Autonomy governor    level(): computes A0 to A4 from policy tier, envelope and confidence
  Checkpoint manager   require(): creates and enforces review gates in a run
  Action guard         Guard.call(): the single door between agents (L6) and tools (L5)
  Steering bus         steer(): pause, edit parameters and rewind a run
  Uncertainty estimator confidence(): combines independent signals into one score
  Escalation router    route(): picks the human role for each gate from policy
"""
import json
import os
import uuid
from datetime import datetime, timezone

from . import l1_data as db
from . import l2_policy as P

LEVELS = ["A0", "A1", "A2", "A3", "A4"]          # A0 most restrictive, A4 fully autonomous
ALWAYS_A1 = {"N3"}                               # regulatory filing: never relaxable
RELAXABLE_KINDS = {"plan", "data_review", "release"}


class NeedsApproval(Exception):
    def __init__(self, kind, title, evidence, role, key, level="A1"):
        super().__init__(title)
        self.kind, self.title, self.evidence, self.role, self.key, self.level = kind, title, evidence, role, key, level


_CRASHED = set()


class SimulatedCrash(BaseException):
    """Evaluation only: stops a run the way a power cut would, without saving state."""


class Blocked(Exception):
    """A policy-forbidden action. The guard refuses it and the attempt is logged."""


# ── escalation router ───────────────────────────────────────────

def route(kind: str, requester_role: str) -> str:
    r = P.body("routing").get(kind, "requester")
    return requester_role if r == "requester" else r


# ── autonomy governor ───────────────────────────────────────────

def level(action_class: str, tool_id: str = None, envelope_ok: bool = False,
          confidence: float = 1.0, mode: str = "haara") -> str:
    if mode == "autonomous":
        return "A4"
    if mode == "guarded":
        # evaluation baseline: every deterministic check stays, every human gate goes.
        # Actions inside a bank limit run at A3 as in HAARA; the rest run unattended at A4.
        return "A3" if action_class == "external" and envelope_ok else "A4"
    if mode == "uniform":
        return "A1" if action_class in ("external", "internal_write") else "A2"
    pol = P.body("autonomy")
    tier = pol["risk_tier_by_class"].get(action_class, "high")
    lvl = pol["level_by_tier"][tier]
    if confidence < pol["confidence_floor"][tier] and lvl in ("A3", "A4"):
        lvl = "A2"
    if action_class == "external":
        # invariant: external effects never exceed A1 unless an approved envelope covers them
        lvl = "A3" if envelope_ok else "A1"
    if tool_id in ALWAYS_A1:
        lvl = "A1"
    return lvl


# ── uncertainty estimator ───────────────────────────────────────

def confidence(signals: dict) -> dict:
    """signals: name -> value in [0, 1]. Missing signals are skipped. The minimum is
    reported alongside the mean so a single weak signal is visible."""
    vals = {k: round(float(v), 3) for k, v in signals.items() if v is not None}
    if not vals:
        return {"score": 0.5, "signals": {}, "weakest": None}
    mean = sum(vals.values()) / len(vals)
    weakest = min(vals, key=vals.get)
    score = round(0.6 * mean + 0.4 * vals[weakest], 3)
    return {"score": score, "signals": vals, "weakest": weakest}


def floor(tier: str) -> float:
    return P.body("autonomy")["confidence_floor"][tier]


# ── checkpoint manager ──────────────────────────────────────────

def require(ctx, kind: str, title: str, evidence: dict, key: str = None, role: str = None, level_: str = "A1"):
    """Stop the run until a human decides, unless this gate was already decided.
    Returns the decision record (approve/edit) when it exists."""
    key = key or f"{kind}:{ctx.node_id}"
    if ctx.mode in ("autonomous", "guarded"):
        ctx.state.setdefault("skipped_gates", []).append(key)
        return {"decision": "auto", "edits": {}}
    d = ctx.state.get("decisions", {}).get(key)
    if d:
        if d["decision"] == "reject":
            raise Blocked(f"Rejected by {d['by']}: {d.get('comment') or title}")
        problem = approval_problem(ctx, d)
        if not problem:
            return d
        # the approval no longer holds: record why and ask again
        ctx.state["decisions"].pop(key, None)
        db.ledger("approval_invalidated", actor="L7.checkpoint_manager", run_id=ctx.run_id, node_id=ctx.node_id,
                  payload={"key": key, "reason": problem, "approved_by": d.get("by")})
        evidence = {**evidence, "reapproval_reason": problem}
        title = f"{title} (approval needed again)"
    raise NeedsApproval(kind, title, evidence, role or route(kind, ctx.role), key, level_)


def approval_problem(ctx, d) -> str:
    """Re-check an approval at the moment it is used. Returns a reason when it is no longer valid."""
    hours = P.body("autonomy").get("approval_valid_hours", 24)
    if d.get("ts") and hours:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(d["ts"].replace("Z", "+00:00"))).total_seconds() / 3600
        if age > hours:
            return f"The approval is {age:.0f} hours old. Approvals are valid for {hours} hours."
    if d.get("by_email"):
        u = db.one("SELECT role, active FROM app_users WHERE lower(email)=lower(?)", (d["by_email"],))
        if not u or not u["active"]:
            return f"The approver's account ({d['by_email']}) is no longer active."
        if u["role"] != d.get("by_role"):
            return f"The approver's role changed from {d.get('by_role')} to {u['role']}."
    pv = d.get("policy_versions")
    if pv:
        now_v = P.versions()
        changed = [k for k in RELEVANT_POLICIES.get(ctx.use_case, []) if pv.get(k) != now_v.get(k)]
        if changed:
            return "Policy changed after the approval: " + ", ".join(f"{k} v{pv.get(k)} to v{now_v.get(k)}" for k in changed) + "."
    return ""


RELEVANT_POLICIES = {
    "credit": ["loan_policy", "envelopes", "autonomy", "routing", "fairness"],
    "bi": ["semantic_layer", "entitlements", "autonomy"],
    "churn": ["churn_policy", "envelopes", "autonomy", "routing", "fairness"],
    "aml": ["aml_policy", "envelopes", "autonomy", "routing"],
    "cash": ["cash_policy", "envelopes", "autonomy", "routing"],
}


def create_checkpoint(ctx, na: NeedsApproval) -> str:
    cp_id = "CP-" + uuid.uuid4().hex[:8]
    # "requester" in the routing policy means the person who asked, not anyone with the same role
    who = getattr(ctx, "created_by", None)
    personal = who and P.body("routing").get(na.kind, "requester") == "requester" and na.role == ctx.role
    db.execute("INSERT INTO checkpoints (cp_id, run_id, node_id, kind, level, assigned_role, assigned_user, title, "
               "evidence, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
               (cp_id, ctx.run_id, ctx.node_id, na.kind, na.level, na.role, who if personal else None, na.title,
                json.dumps({"key": na.key, **na.evidence}, default=str), "open", db.now()))
    db.ledger("checkpoint_opened", actor="L7.checkpoint_manager", run_id=ctx.run_id, node_id=ctx.node_id,
              level=na.level, payload={"cp_id": cp_id, "kind": na.kind, "role": na.role, "title": na.title})
    return cp_id


def notify(ctx, role: str, title: str, body: str):
    db.execute("INSERT INTO notifications (ts, run_id, role, title, body) VALUES (?,?,?,?,?)",
               (db.now(), ctx.run_id, role, title, body))


def record_decision(cp: dict, decision: str, role: str, comment: str, edits: dict, use_case: str, by=None, actor=None):
    db.execute("UPDATE checkpoints SET status='closed', decided_at=?, decided_by=?, decision=?, comment=?, edits=? "
               "WHERE cp_id=?", (db.now(), by or role, decision, comment, json.dumps(edits or {}), cp["cp_id"]))
    db.execute("INSERT INTO oversight_memory (ts, run_id, use_case, kind, decision, changed, role, comment) "
               "VALUES (?,?,?,?,?,?,?,?)",
               (db.now(), cp["run_id"], use_case, cp["kind"], decision, int(bool(edits)), role, comment))
    db.ledger("human_decision", actor=actor or f"human:{role}", run_id=cp["run_id"], node_id=cp["node_id"],
              level=cp["level"], payload={"cp_id": cp["cp_id"], "kind": cp["kind"], "decision": decision,
                                          "comment": comment, "edits": edits})


def calibration_suggestions() -> list:
    """Propose relaxing gates people approve unchanged again and again. Never for external
    actions or credit/AML decisions (safety-monotone rule). A human admin applies any change."""
    n = P.body("autonomy")["calibration_min_unchanged"]
    rows = db.query("SELECT use_case, kind, decision, changed FROM oversight_memory ORDER BY id")
    streak = {}
    for r in rows:
        k = (r["use_case"], r["kind"])
        streak[k] = streak.get(k, 0) + 1 if (r["decision"] == "approve" and not r["changed"]) else 0
    out = []
    for (uc, kind), s in streak.items():
        if kind in RELAXABLE_KINDS and s >= n:
            out.append({"use_case": uc, "gate": kind, "unchanged_approvals": s,
                        "suggestion": f"Relax the {kind} gate for {uc} to review-after (A2)."})
    return out


# ── action guard ────────────────────────────────────────────────

class Guard:
    """Every agent-to-tool call passes here. It enforces agent scopes and policy rules,
    sets the autonomy level, demands approval when needed, and writes provenance."""

    def __init__(self, registry: dict, scopes: dict):
        self.registry = registry
        self.scopes = scopes

    def call(self, ctx, agent: str, tool_id: str, **args):
        tool = self.registry.get(tool_id)
        if tool is None:
            self._deny(ctx, agent, tool_id, "unknown tool", "external")
        cls = tool["class"]
        if tool_id not in self.scopes.get(agent, set()) and ctx.mode != "autonomous":
            self._deny(ctx, agent, tool_id, f"{agent} is not scoped to use {tool_id}", cls)
        reason = tool.get("precheck") and tool["precheck"](ctx, **args)
        if reason and ctx.mode != "autonomous":
            self._deny(ctx, agent, tool_id, reason, cls)
        env_ok = bool(tool.get("envelope") and tool["envelope"](ctx, **args))
        lvl = level(cls, tool_id, env_ok, ctx.state.get("confidence", 1.0), ctx.mode)
        key = f"action:{tool_id}:{tool['key'](args) if tool.get('key') else ctx.node_id}"
        if lvl in ("A0", "A1") or (ctx.mode == "uniform" and cls in ("external", "internal_write")):
            preview = tool["preview"](ctx, **args) if tool.get("preview") else {"args": args}
            require(ctx, "action", f"{tool['name']}: authorise this action", {
                "tool": tool_id, "class": cls, "level": lvl, "dry_run": preview,
                "envelope": "outside envelope" if tool.get("envelope") else "no envelope applies"},
                key=key, role=route(tool.get("route", "escalation"), ctx.role), level_=lvl)
            ctx.state.setdefault("authorised", []).append(key)
        ik = f"{tool_id}:{tool['ikey'](args)}" if tool.get("ikey") else None
        if ik and tool.get("ikey_scope") == "run":
            ik = f"{ctx.run_id}:{ik}"
        done = ctx.state.setdefault("executed", {})
        if ik and ik in done:
            return done[ik]          # idempotent replay after a pause: never repeat an effect
        if ik and cls in ("external", "internal_write") and not os.environ.get("HAARA_DISABLE_EFFECT_LOG"):
            prior = db.one("SELECT run_id, result FROM action_effects WHERE effect_key=?", (ik,))
            if prior:                # the effect already happened, here or in another run
                db.ledger("duplicate_suppressed", actor="L7.action_guard", run_id=ctx.run_id, node_id=ctx.node_id,
                          tool_id=tool_id, action_class=cls, payload={"effect_key": ik, "first_run": prior["run_id"]})
                result = json.loads(prior["result"]) if prior["result"] else None
                done[ik] = result
                return result
            # the effect and its record commit together, so a crash cannot leave one without the other
            result = db.atomic(lambda: self._effect(ctx, tool, args, ik))
        else:
            result = tool["fn"](ctx, **args)
        if ik:
            done[ik] = result
        crash = ctx.state.get("params", {}).get("_crash_after_tool")
        if crash == tool_id and os.environ.get("HAARA_FAULTS") == "1" and (ctx.run_id, tool_id) not in _CRASHED:
            _CRASHED.add((ctx.run_id, tool_id))     # one crash per run and tool, as a real outage would be
            raise SimulatedCrash(f"simulated crash after {tool_id}")
        ctx.calls.append({"agent": agent, "tool": tool_id, "class": cls, "level": lvl})
        db.ledger("tool_call", actor=agent, run_id=ctx.run_id, node_id=ctx.node_id, tool_id=tool_id,
                  action_class=cls, level=lvl, policy_version=ctx.policy_versions.get(tool.get("policy", "")),
                  model_version=tool.get("model"),
                  payload={"args": _short(args), "result": _short(result)})
        if lvl == "A3" and cls in ("external", "internal_write"):
            notify(ctx, route(tool.get("route", "escalation"), ctx.role), f"{tool['name']} executed",
                   json.dumps(_short(args)))
        if ctx.mode == "uniform":
            ctx.state["uniform_gates"] = ctx.state.get("uniform_gates", 0) + 1
        return result

    @staticmethod
    def _effect(ctx, tool, args, ik):
        result = tool["fn"](ctx, **args)
        db.execute("INSERT INTO action_effects (effect_key, tool_id, run_id, result) VALUES (?,?,?,?)",
                   (ik, tool["id"], ctx.run_id, json.dumps(result, default=str)))
        return result

    def _deny(self, ctx, agent, tool_id, reason, cls):
        ctx.blocked.append({"agent": agent, "tool": tool_id, "reason": reason})
        db.ledger("tool_blocked", actor="L7.action_guard", run_id=ctx.run_id, node_id=ctx.node_id,
                  tool_id=tool_id, action_class=cls, payload={"agent": agent, "reason": reason})
        raise Blocked(reason)


def _short(x, n=600):
    s = json.dumps(x, default=str)
    return json.loads(s) if len(s) <= n else s[:n] + "...(truncated)"
