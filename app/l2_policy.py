"""L2 Semantic and policy.

Versioned policy-as-code that the bank edits in Policy Studio. Agents read it
through tools but have no tool that writes it. Every edit creates a new version
and a ledger entry, and every run records the versions it used.
"""
import copy
import json

from . import l1_data as db

ROLES = {
    "analyst": "Data analyst. Asks questions and runs analyses. Approves plans and releases for own runs.",
    "loan_officer": "Loan officer. Reviews credit recommendations within delegated authority.",
    "credit_manager": "Credit manager. Approves policy exceptions and loans above the officer limit.",
    "compliance_officer": "Compliance officer. Reviews AML cases, authorises holds and files STRs.",
    "retail_manager": "Retail manager. Approves retention strategies and customer outreach.",
    "branch_ops": "Branch operations. Approves cash replenishment outside the envelope.",
    "admin": "Policy administrator. Edits policies in Policy Studio.",
}

DEFAULTS = {
    "loan_policy": {
        "description": "Retail and SME loan policy (fictional).",
        "rules": [
            {"id": "R1", "name": "Age at application", "kind": "hard", "enabled": True,
             "params": {"min_age": 21, "max_age_at_maturity": 65},
             "text": "Applicant is at least 21 and no older than 65 at loan maturity."},
            {"id": "R2", "name": "Identity verified", "kind": "hard", "enabled": True, "params": {},
             "text": "Applicant's BVN is verified."},
            {"id": "R3", "name": "Relationship tenure", "kind": "hard", "enabled": True,
             "params": {"min_tenure_months": 6},
             "text": "Applicant has banked with us for at least 6 months."},
            {"id": "R4", "name": "Debt-to-income", "kind": "hard", "enabled": True,
             "params": {"max_dti": 0.40},
             "text": "Existing plus new monthly repayments do not exceed 40% of monthly income."},
            {"id": "R5", "name": "Amount versus inflow", "kind": "soft", "enabled": True,
             "params": {"max_multiple_of_inflow": 4.0, "collateral_waiver_coverage": 1.2},
             "text": "Unsecured amount is at most 4 times average monthly inflow. Collateral of at least 120% of the amount waives this rule."},
            {"id": "R6", "name": "No arrears or default", "kind": "hard", "enabled": True,
             "params": {"max_dpd": 29},
             "text": "No existing loan is 30 or more days past due."},
            {"id": "R7", "name": "Maximum tenor", "kind": "hard", "enabled": True,
             "params": {"max_tenor_months": 36},
             "text": "Tenor does not exceed 36 months."},
            {"id": "R8", "name": "Thin-file alternative data", "kind": "hard", "enabled": True,
             "params": {"min_utility_ontime": 0.75},
             "text": "Applicants without a bureau score need consent to alternative data and at least 75% on-time utility payments."},
            {"id": "R9", "name": "Probability of default", "kind": "soft", "enabled": True,
             "params": {"approve_below": 0.12, "decline_above": 0.30},
             "text": "Model PD below 12% supports approval. 12% to 30% needs referral. Above 30% supports decline."},
        ],
        "interest_rate_annual": 0.28,
        "officer_delegated_limit": 3_000_000,
        "auto_decision_envelope": {"enabled": False, "max_amount": 500_000, "max_pd": 0.06,
                                   "note": "When enabled, clean approvals inside this envelope are written without waiting for a human (A3)."},
    },
    "aml_policy": {
        "cash_reporting_threshold": 5_000_000,
        "structuring": {"window_days": 7, "min_deposits": 3, "band_low_ratio": 0.80},
        "mule": {"min_inbound": 8, "window_hours": 6, "passthrough_ratio": 0.85},
        "card_fraud": {"multiple_of_income": 1.2, "night_start": 0, "night_end": 5},
        "anomaly_contamination": 0.01,
        "alert_score": 0.60, "hold_score": 0.80,
        "anti_tipping_off": True,
    },
    "churn_policy": {
        "definition": "A customer churns when salary or business inflows stop and the balance falls by half, or the account closes, within the horizon.",
        "horizon_days": 90, "risk_threshold": 0.50, "max_contacts_per_customer_30d": 2,
        "offers": [
            {"id": "O1", "name": "Fee waiver for 3 months", "cost": 3000, "drivers": ["charges and fees"], "segments": ["Retail", "SME"]},
            {"id": "O2", "name": "Service recovery call", "cost": 1500, "drivers": ["failed transfer", "card declined", "app login issue"], "segments": ["Retail", "SME"]},
            {"id": "O3", "name": "Relationship manager visit", "cost": 12000, "drivers": ["salary stopped", "outflow to other banks"], "segments": ["SME"]},
            {"id": "O4", "name": "Savings rate bonus", "cost": 5000, "drivers": ["outflow to other banks", "salary stopped"], "segments": ["Retail"]},
        ],
        "campaign_budget": 1_500_000,
    },
    "cash_policy": {"horizon_days": 7, "safety_days": 1.0, "min_fill_ratio": 0.25, "cost_per_trip": 45000},
    "fairness": {"protected_attributes": ["gender"], "monitored_attributes": ["gender", "region"],
                 "min_disparate_impact": 0.80,
                 "note": "Protected attributes are never model inputs. They are used only to monitor outcomes."},
    "envelopes": {
        "fraud_hold": {"enabled": True, "max_amount": 1_000_000, "max_hours": 24, "expires": "2026-12-31"},
        "cash_orders": {"enabled": True, "max_order_per_atm": 12_000_000, "max_total": 120_000_000, "expires": "2026-12-31"},
        "retention_campaign": {"enabled": False, "max_cost": 500_000, "expires": "2026-12-31"},
    },
    "autonomy": {
        "risk_tier_by_class": {"read": "low", "compute": "low", "internal_write": "medium", "external": "high"},
        "level_by_tier": {"low": "A4", "medium": "A3", "high": "A1"},
        "confidence_floor": {"low": 0.50, "medium": 0.70, "high": 0.85},
        "plan_approval": {"credit": False, "bi": False, "churn": True, "aml": True, "cash": False},
        "data_review": "on_issue",
        "bi_interpretation_check": "when_unsure",
        "approval_valid_hours": 24,
        "calibration_min_unchanged": 5,
    },
    "routing": {
        "plan": "requester", "data_review": "requester", "release": "requester",
        "credit_decision": "loan_officer", "credit_exception": "credit_manager",
        "aml_case": "compliance_officer", "fraud_hold": "compliance_officer", "str_filing": "compliance_officer",
        "churn_strategy": "retail_manager", "campaign": "retail_manager",
        "cash_order": "branch_ops", "escalation": "requester", "interpretation": "requester",
    },
    "uploads": {
        "description": "People may upload their own spreadsheets and ask questions about them. Only the uploader can "
                       "analyse a dataset. Personal columns are replaced by tokens before storage.",
        "enabled": True,
        "roles": ["analyst", "loan_officer", "credit_manager", "compliance_officer", "retail_manager", "branch_ops", "admin"],
        "max_file_mb": 25, "max_rows": 200000, "max_columns": 80,
        "mask_personal_columns": True,
    },
    "entitlements": {
        "analyst": {"metrics": "*", "pii": False}, "loan_officer": {"metrics": ["loan_book", "loan_count", "delinquency_rate", "npl_ratio", "approval_rate"], "pii": True},
        "credit_manager": {"metrics": "*", "pii": True}, "compliance_officer": {"metrics": "*", "pii": True},
        "retail_manager": {"metrics": "*", "pii": False}, "branch_ops": {"metrics": ["atm_withdrawals", "txn_count", "deposit_inflow"], "pii": False},
        "admin": {"metrics": "*", "pii": False},
    },
    "semantic_layer": {
        "as_of": "2026-09-30",
        "metrics": {
            "delinquency_rate": {"label": "Delinquency rate", "unit": "%", "fact": "loans",
                                 "sql": "100.0*SUM(CASE WHEN l.dpd>=30 THEN 1 ELSE 0 END)/COUNT(*)",
                                 "definition": "Share of loans 30 or more days past due."},
            "npl_ratio": {"label": "Non-performing loan ratio", "unit": "%", "fact": "loans",
                          "sql": "100.0*SUM(CASE WHEN l.dpd>=90 THEN l.outstanding ELSE 0 END)/SUM(l.outstanding)",
                          "definition": "Outstanding balance 90 or more days past due over total outstanding."},
            "loan_book": {"label": "Loan book", "unit": "NGN", "fact": "loans", "sql": "SUM(l.outstanding)",
                          "definition": "Total outstanding loan balance."},
            "loan_count": {"label": "Number of loans", "unit": "", "fact": "loans", "sql": "COUNT(*)",
                           "definition": "Number of loans on book."},
            "customer_count": {"label": "Customers", "unit": "", "fact": "customers", "sql": "COUNT(DISTINCT c.customer_id)",
                               "definition": "Number of customers."},
            "avg_income": {"label": "Average monthly income", "unit": "NGN", "fact": "customers", "sql": "AVG(c.monthly_income)",
                           "definition": "Mean declared monthly income."},
            "deposit_inflow": {"label": "Deposit inflow", "unit": "NGN", "fact": "transactions",
                               "sql": "SUM(CASE WHEN t.direction='credit' THEN t.amount ELSE 0 END)",
                               "definition": "Total credits into customer accounts."},
            "txn_count": {"label": "Transactions", "unit": "", "fact": "transactions", "sql": "COUNT(DISTINCT t.txn_id)",
                          "definition": "Number of distinct transactions."},
            "complaint_count": {"label": "Complaints", "unit": "", "fact": "interactions",
                                "sql": "SUM(CASE WHEN i.type='complaint' THEN 1 ELSE 0 END)",
                                "definition": "Number of complaints logged."},
            "approval_rate": {"label": "Approval rate", "unit": "%", "fact": "decisions",
                              "sql": "100.0*SUM(CASE WHEN d.decision='approve' THEN 1 ELSE 0 END)/COUNT(*)",
                              "definition": "Share of final loan decisions that were approvals."},
            "atm_withdrawals": {"label": "ATM withdrawals", "unit": "NGN", "fact": "atm_cash", "sql": "SUM(a.withdrawals)",
                                "definition": "Cash withdrawn from ATMs."},
        },
        "dimensions": {
            "branch": {"loans": "b.name", "customers": "b.name", "transactions": "b.name", "interactions": "b.name", "atm_cash": "b.name"},
            "city": {"loans": "b.city", "customers": "b.city", "transactions": "b.city", "interactions": "b.city", "atm_cash": "b.city"},
            "region": {"loans": "b.region", "customers": "b.region", "transactions": "b.region", "interactions": "b.region", "atm_cash": "b.region"},
            "segment": {"loans": "c.segment", "customers": "c.segment", "transactions": "c.segment", "interactions": "c.segment"},
            "product": {"loans": "l.product"},
            "channel": {"transactions": "t.channel", "interactions": "i.channel"},
            "month": {"transactions": "to_char(t.ts,'YYYY-MM')", "interactions": "to_char(i.date,'YYYY-MM')", "atm_cash": "to_char(a.date,'YYYY-MM')"},
            "topic": {"interactions": "i.topic"},
        },
        "facts": {
            "loans": "loans l JOIN customers c ON c.customer_id=l.customer_id JOIN branches b ON b.branch_id=l.branch_id",
            "customers": "customers c JOIN branches b ON b.branch_id=c.branch_id",
            "transactions": "transactions_clean t JOIN customers c ON c.customer_id=t.customer_id JOIN branches b ON b.branch_id=c.branch_id",
            "interactions": "interactions i JOIN customers c ON c.customer_id=i.customer_id JOIN branches b ON b.branch_id=c.branch_id",
            "atm_cash": "atm_cash a JOIN atms m ON m.atm_id=a.atm_id JOIN branches b ON b.branch_id=m.branch_id",
            "decisions": "ext_loan_decisions d",
        },
        "time_columns": {"transactions": "t.ts", "interactions": "i.date", "atm_cash": "a.date"},
    },
}


def ensure_defaults() -> None:
    for key, body in DEFAULTS.items():
        if not db.one("SELECT 1 FROM policies WHERE key=?", (key,)):
            db.execute("INSERT INTO policies VALUES (?,?,?,?,?,?)",
                       (key, 1, json.dumps(body), db.now(), "system", "Initial policy"))
            db.ledger("policy_created", actor="system", payload={"key": key, "version": 1})
            continue
        # a newer release added settings: add them with default values as a new version
        cur, ver = get(key)
        if isinstance(cur, dict) and isinstance(body, dict) and any(k not in cur for k in body):
            added = [k for k in body if k not in cur]
            new = {**cur, **{k: copy.deepcopy(body[k]) for k in added}}
            db.execute("INSERT INTO policies VALUES (?,?,?,?,?,?)",
                       (key, ver + 1, json.dumps(new), db.now(), "system", "Added new settings: " + ", ".join(added)))
            db.ledger("policy_updated", actor="system", policy_version=ver + 1, payload={"key": key, "added": added})


def get(key: str) -> tuple:
    r = db.one("SELECT body, version FROM policies WHERE key=? ORDER BY version DESC LIMIT 1", (key,))
    return json.loads(r["body"]), r["version"]


def body(key: str) -> dict:
    return get(key)[0]


def versions() -> dict:
    rows = db.query("SELECT key, MAX(version) v FROM policies GROUP BY key")
    return {r["key"]: r["v"] for r in rows}


def history(key: str) -> list:
    return db.query("SELECT version, created_at, created_by, note FROM policies WHERE key=? ORDER BY version DESC", (key,))


def _same_shape(a, b, path="") -> list:
    """Edits may change values, not structure, except list lengths of rules and offers."""
    errs = []
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [f"{path} must be an object"]
        for k in a:
            if k not in b:
                errs.append(f"{path}.{k} is missing")
            else:
                errs += _same_shape(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, bool):
        if not isinstance(b, bool):
            errs.append(f"{path} must be true or false")
    elif isinstance(a, (int, float)):
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            errs.append(f"{path} must be a number")
        elif b < 0:
            errs.append(f"{path} must not be negative")
    return errs


def update(key: str, new_body: dict, role: str, note: str, who=None) -> dict:
    """Policy edits come only from a human admin through Policy Studio."""
    if role != "admin":
        db.ledger("policy_edit_denied", actor=who or f"human:{role}", payload={"key": key})
        raise PermissionError("Only the admin role may edit policies.")
    current, ver = get(key)
    errs = _same_shape(DEFAULTS.get(key, current), new_body, key)
    if key == "autonomy" and new_body.get("bi_interpretation_check") not in ("always", "when_unsure", "after_only"):
        errs.append("autonomy.bi_interpretation_check must be always, when_unsure or after_only")
    if errs:
        raise ValueError("; ".join(errs[:5]))
    db.execute("INSERT INTO policies VALUES (?,?,?,?,?,?)",
               (key, ver + 1, json.dumps(new_body), db.now(), who or role, note or ""))
    db.ledger("policy_updated", actor=who or f"human:{role}", policy_version=ver + 1,
              payload={"key": key, "from": ver, "to": ver + 1, "note": note})
    return {"key": key, "version": ver + 1}


def reset(key: str, role: str, who=None) -> dict:
    return update(key, copy.deepcopy(DEFAULTS[key]), role, "Reset to default", who)
