"""The bank's own data as datasets, so full analysis and dashboards work on it the same way as on uploads.

Each bank dataset is a read-only SQL view over the core tables. It leaves out personal columns (names, phone numbers,
BVN status, counterparties) and joins branch, product and customer segment so results can be broken down by them.
The views are live: a question reads the data as it is when the question runs. Access is by role, set in the
uploads policy (bank_data_roles) with the defaults below. Bank datasets cannot be deleted.
"""
import hashlib
import json

import pandas as pd

from . import datasets as DS
from . import l1_data as db
from . import l2_policy as P

OWNER = "bank"
M, C, I, D = "measure", "category", "identifier", "date"

VIEWS = {
    "loans": {
        "name": "Loan portfolio", "roles": ["analyst", "loan_officer", "credit_manager", "admin"],
        "about": "Every loan on the book with branch, product, customer segment, balance and arrears.",
        "sql": """SELECT l.loan_id, b.name AS branch, b.city, b.region, COALESCE(p.name, l.product) AS product,
                  c.segment, c.employer_type, l.status, l.principal::float8 AS principal, l.outstanding::float8 AS outstanding,
                  l.monthly_installment::float8 AS monthly_installment, l.tenor_months::float8 AS tenor_months,
                  l.dpd::float8 AS days_past_due, 1::float8 AS loans,
                  CASE WHEN l.status = 'default' THEN 1 ELSE 0 END::float8 AS in_default,
                  CASE WHEN l.status = 'arrears' THEN 1 ELSE 0 END::float8 AS in_arrears,
                  CASE WHEN l.status IN ('arrears', 'default') THEN l.outstanding ELSE 0 END::float8 AS outstanding_at_risk
                  FROM loans l LEFT JOIN branches b ON b.branch_id = l.branch_id
                  LEFT JOIN products p ON p.product_code = l.product LEFT JOIN customers c ON c.customer_id = l.customer_id""",
        "cols": [("loan_id", "Loan ID", I), ("branch", "Branch", C), ("city", "City", C), ("region", "Region", C),
                 ("product", "Product", C), ("segment", "Segment", C), ("employer_type", "Employer type", C),
                 ("status", "Status", C), ("principal", "Principal", M), ("outstanding", "Outstanding", M),
                 ("monthly_installment", "Monthly installment", M), ("tenor_months", "Tenor (months)", M),
                 ("days_past_due", "Days past due", M), ("loans", "Loans", M), ("in_default", "In default", M),
                 ("in_arrears", "In arrears", M), ("outstanding_at_risk", "Outstanding at risk", M)]},
    "applications": {
        "name": "Loan applications", "roles": ["analyst", "loan_officer", "credit_manager", "admin"],
        "about": "Loan applications with requested amount, collateral, declared income, bureau score and decision status.",
        "sql": """SELECT a.app_id, a.applied_on, COALESCE(p.name, a.product) AS product, a.purpose, a.status, b.name AS branch,
                  b.region, c.segment, a.amount::float8 AS amount, a.tenor_months::float8 AS tenor_months,
                  a.collateral_value::float8 AS collateral_value, a.declared_income::float8 AS declared_income,
                  a.existing_installments::float8 AS existing_installments, a.bureau_score::float8 AS bureau_score, 1::float8 AS applications
                  FROM loan_applications a LEFT JOIN products p ON p.product_code = a.product
                  LEFT JOIN customers c ON c.customer_id = a.customer_id LEFT JOIN branches b ON b.branch_id = c.branch_id""",
        "cols": [("app_id", "Application ID", I), ("applied_on", "Applied on", D), ("product", "Product", C),
                 ("purpose", "Purpose", C), ("status", "Status", C), ("branch", "Branch", C), ("region", "Region", C),
                 ("segment", "Segment", C), ("amount", "Amount", M), ("tenor_months", "Tenor (months)", M),
                 ("collateral_value", "Collateral value", M), ("declared_income", "Declared income", M),
                 ("existing_installments", "Existing installments", M), ("bureau_score", "Bureau score", M), ("applications", "Applications", M)]},
    "transactions": {
        "name": "Transactions", "roles": ["analyst", "compliance_officer", "retail_manager", "admin"],
        "about": "Cleaned account transactions with channel, direction, category, branch and customer segment.",
        "sql": """SELECT t.txn_id, t.ts::date AS txn_date, t.channel, t.direction, t.category,
                  CASE WHEN t.is_salary = 1 THEN 'Yes' ELSE 'No' END AS salary, b.name AS branch, b.region, c.segment,
                  t.customer_id, t.amount::float8 AS amount,
                  CASE WHEN t.direction = 'credit' THEN t.amount ELSE 0 END::float8 AS inflow,
                  CASE WHEN t.direction = 'debit' THEN t.amount ELSE 0 END::float8 AS outflow, 1::float8 AS transactions
                  FROM transactions_clean t LEFT JOIN customers c ON c.customer_id = t.customer_id
                  LEFT JOIN branches b ON b.branch_id = c.branch_id""",
        "cols": [("txn_id", "Transaction ID", I), ("txn_date", "Date", D), ("channel", "Channel", C),
                 ("direction", "Direction", C), ("category", "Category", C), ("salary", "Salary payment", C),
                 ("branch", "Branch", C), ("region", "Region", C), ("segment", "Segment", C),
                 ("customer_id", "Customer ID", I), ("amount", "Amount", M), ("inflow", "Inflow", M),
                 ("outflow", "Outflow", M), ("transactions", "Transactions", M)]},
    "atm_cash": {
        "name": "ATM cash", "roles": ["analyst", "branch_ops", "admin"],
        "about": "Daily withdrawals, replenishment and closing balance for each ATM.",
        "sql": """SELECT a.atm_id, a.date, m.location, m.city, b.name AS branch, a.withdrawals::float8 AS withdrawals,
                  a.replenished::float8 AS replenished, a.closing_balance::float8 AS closing_balance,
                  m.capacity::float8 AS capacity
                  FROM atm_cash a LEFT JOIN atms m ON m.atm_id = a.atm_id LEFT JOIN branches b ON b.branch_id = m.branch_id""",
        "cols": [("atm_id", "ATM", C), ("date", "Date", D), ("location", "Location", C), ("city", "City", C),
                 ("branch", "Branch", C), ("withdrawals", "Withdrawals", M), ("replenished", "Replenished", M),
                 ("closing_balance", "Closing balance", M), ("capacity", "Capacity", M)]},
    "interactions": {
        "name": "Customer interactions", "roles": ["analyst", "retail_manager", "admin"],
        "about": "Contacts with customers by channel, type and topic, with sentiment and whether each was resolved.",
        "sql": """SELECT i.interaction_id, i.date, i.channel, i.type, i.topic,
                  CASE WHEN i.resolved = 1 THEN 'Yes' ELSE 'No' END AS resolved, b.name AS branch, c.segment,
                  i.sentiment::float8 AS sentiment, i.resolved::float8 AS resolved_count, 1::float8 AS contacts
                  FROM interactions i LEFT JOIN customers c ON c.customer_id = i.customer_id
                  LEFT JOIN branches b ON b.branch_id = c.branch_id""",
        "cols": [("interaction_id", "Interaction ID", I), ("date", "Date", D), ("channel", "Channel", C),
                 ("type", "Type", C), ("topic", "Topic", C), ("resolved", "Resolved", C), ("branch", "Branch", C),
                 ("segment", "Segment", C), ("sentiment", "Sentiment", M), ("resolved_count", "Resolved count", M), ("contacts", "Contacts", M)]},
}


def _id(key):
    return f"BANK-{key}"


def roles_for(key):
    over = (P.body("uploads") or {}).get("bank_data_roles") or {}
    return over.get(key) or VIEWS[key]["roles"]


def is_bank(dataset_id):
    return str(dataset_id or "").startswith("BANK-")


def allowed(dataset_id, role):
    key = str(dataset_id)[5:]
    return key in VIEWS and (role == "admin" or role in roles_for(key))


def sync(max_age_hours=0):
    """Creates or refreshes the views and their dataset records. Runs at start-up. Safe to run again.

    With max_age_hours, a view whose definition is unchanged and whose profile is newer than that is left alone,
    so a serverless instance starts in a second instead of re-reading 126,000 transactions."""
    done = []
    for key, v in VIEWS.items():
        view = f"bank_{key}"
        sha = hashlib.sha256(v["sql"].encode()).hexdigest()
        if max_age_hours and db.table_exists(view):
            r = db.one("SELECT sha256, uploaded_at FROM datasets WHERE dataset_id=? AND status='ready'", (_id(key),))
            when = pd.Timestamp(r["uploaded_at"]) if r and r["uploaded_at"] else None
            if when is not None and when.tzinfo is None:
                when = when.tz_localize("UTC")
            if r and r["sha256"] == sha and when is not None and \
                    pd.Timestamp.now(tz="UTC") - when < pd.Timedelta(hours=max_age_hours):
                done.append((key, None))
                continue
        db.script(f"CREATE OR REPLACE VIEW {view} AS {v['sql']}")
        frame = pd.DataFrame(db.query(f"SELECT * FROM {view}"))
        cols = []
        for name, label, role in v["cols"]:
            typ = "number" if role == M else "date" if role == D else "text"
            cols.append({"name": name, "label": label, "type": typ, "role": role})
            if typ == "date":
                frame[name] = pd.to_datetime(frame[name], errors="coerce").dt.date
            elif typ == "number":
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
        profile = DS._profile(frame, cols)
        args = (v["name"], f"Bank data: {v['about']}", OWNER, "bank", db.now(), len(frame), json.dumps(cols),
                json.dumps(profile, default=str), view, sha)
        if db.one("SELECT 1 FROM datasets WHERE dataset_id=?", (_id(key),)):
            db.execute("UPDATE datasets SET name=?, file_name=?, owner_email=?, owner_role=?, uploaded_at=?, row_count=?, "
                       "columns=?, profile=?, table_name=?, sha256=?, status='ready' WHERE dataset_id=?", args + (_id(key),))
        else:
            db.execute("INSERT INTO datasets (name, file_name, owner_email, owner_role, uploaded_at, row_count, columns, "
                       "profile, table_name, sha256, status, dataset_id) VALUES (?,?,?,?,?,?,?,?,?,?,'ready',?)",
                       args + (_id(key),))
        done.append((key, len(frame)))
    return done


def listing(who) -> list:
    out = []
    for key, v in VIEWS.items():
        if not allowed(_id(key), who["role"]):
            continue
        r = db.one("SELECT row_count, profile FROM datasets WHERE dataset_id=? AND status='ready'", (_id(key),))
        if not r:
            continue
        p = json.loads(r["profile"])
        out.append({"dataset_id": _id(key), "name": v["name"], "about": v["about"], "row_count": r["row_count"],
                    "columns": len(p["columns"]), "score": p["score"],
                    "warnings": sum(1 for i in p["issues"] if i["severity"] == "warning"), "bank": True})
    return out


CUES = {
    "loans": r"loan portfolio|loan book|\bloans?\b(?! applications?)|outstanding|principal|arrears|in default|default rate|\bpar\b|\bnpl\b|days past due",
    "applications": r"applications?\b|applicants?|bureau score|collateral|declared income",
    "transactions": r"transactions?\b|inflow|outflow|channel|\bpos\b|ussd|transfers?\b|average ticket",
    "atm_cash": r"\batms?\b|withdrawals?|replenish|closing balance|cash position",
    "interactions": r"interactions?\b|complaints?|sentiment|contacts?\b|call cent|resolution",
}


def wants_dataset(prompt: str) -> bool:
    from . import dataset_dashboard as DD
    from . import dataset_report as RP
    return DD.is_dashboard(prompt) or RP.is_report(prompt)


def pick(prompt: str, role: str):
    """For a full analysis or dashboard typed on Ask: the bank dataset it is about, or None.

    Each dataset scores its topic words and the columns the request names. The best score wins, among the datasets
    the person's role may use. A request with no clear topic stays a normal question."""
    import re
    from . import dataset_dashboard as DD
    from . import dataset_report as RP
    if not (DD.is_dashboard(prompt) or RP.is_report(prompt)):
        return None
    q = " " + prompt.lower() + " "
    best, score = None, 0
    for key in VIEWS:
        if not allowed(_id(key), role):
            continue
        try:
            cols = DS._row(_id(key))["columns"]
        except ValueError:
            continue
        s = 3 * len(re.findall(CUES[key], q)) + len({n for *_, n in DS._mentions(q, [c for c in cols if c["role"] != I or "id" in c["name"]])})
        if s > score:
            best, score = key, s
    return _id(best) if best and score >= 3 else None


def unmatched_message(role: str) -> str:
    names = [VIEWS[k]["name"] for k in VIEWS if allowed(_id(k), role)]
    if not names:
        return "Your role cannot use the bank's data for a full analysis or dashboard. Upload a file in Your data instead."
    listed = ", ".join(names[:-1]) + (" or " if len(names) > 1 else "") + names[-1]
    return (f"Analytixa could not tell which bank data this is for, or your role cannot use it. Your role can use {listed}. "
            "Name one in the request, or open it in Your data and ask there.")
