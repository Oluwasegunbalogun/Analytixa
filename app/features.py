"""Feature builders (the D6 feature store). Used by model training and by L5 tools."""
import numpy as np
import pandas as pd

from . import l1_data as db

AS_OF = pd.Timestamp("2026-09-30 23:59:59")

CREDIT_FEATURES = ["log_income", "dti", "tenure_months", "utility_ontime_ratio", "log_airtime",
                   "mm_inflow_ratio", "has_bureau", "bureau_centered", "prior_default", "informal_income"]

CREDIT_LABELS = {
    "log_income": "monthly income", "dti": "debt-to-income ratio", "tenure_months": "relationship tenure",
    "utility_ontime_ratio": "on-time utility payments", "log_airtime": "airtime spend",
    "mm_inflow_ratio": "mobile money inflow relative to income", "has_bureau": "credit bureau file",
    "bureau_centered": "credit bureau score", "prior_default": "previous default",
    "informal_income": "informal income source",
}


def credit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df needs monthly_income, dti, tenure_months, utility_ontime_ratio, airtime_monthly_avg,
    mobile_money_inflow_avg, bureau_score, prior_default, informal_income."""
    inc = df["monthly_income"].clip(lower=1000)
    out = pd.DataFrame({
        "log_income": np.log(inc),
        "dti": df["dti"].clip(0, 2),
        "tenure_months": df["tenure_months"],
        "utility_ontime_ratio": df["utility_ontime_ratio"].clip(0, 1),
        "log_airtime": np.log(df["airtime_monthly_avg"].clip(lower=100)),
        "mm_inflow_ratio": (df["mobile_money_inflow_avg"] / inc).clip(0, 3),
        "has_bureau": df["bureau_score"].notna().astype(int),
        "bureau_centered": (df["bureau_score"].fillna(640) - 640) / 100,
        "prior_default": df["prior_default"],
        "informal_income": df["informal_income"],
    })
    return out[CREDIT_FEATURES]


CHURN_FEATURES = ["tenure_months", "salary_stopped", "inflow_change", "other_bank_share",
                  "complaints_90d", "unresolved", "login_trend", "negative_sentiment"]

CHURN_LABELS = {
    "tenure_months": "short relationship tenure", "salary_stopped": "salary stopped",
    "inflow_change": "falling inflows", "other_bank_share": "outflow to other banks",
    "complaints_90d": "recent complaints", "unresolved": "unresolved complaints",
    "login_trend": "falling app logins", "negative_sentiment": "negative service sentiment",
}

DRIVER_TOPICS = {"salary_stopped": "salary stopped", "other_bank_share": "outflow to other banks",
                 "inflow_change": "salary stopped"}


def churn_frame(customer_ids=None) -> pd.DataFrame:
    cust = pd.DataFrame(db.query("SELECT customer_id, tenure_months, employer_type, segment, city, region, gender, "
                                 "branch_id, consent_marketing, dnd, monthly_income FROM customers ORDER BY customer_id"))
    if customer_ids is not None:
        cust = cust[cust.customer_id.isin(customer_ids)]
    tx = pd.DataFrame(db.query(
        "SELECT customer_id, to_char(ts,'YYYY-MM') m, direction, is_salary, counterparty, amount FROM transactions ORDER BY entry_id"))
    tx = tx[tx.customer_id.isin(cust.customer_id)]
    recent = tx.m.isin(["2026-08", "2026-09"])
    inflow = tx[tx.direction == "credit"]
    inc_recent = inflow[recent[inflow.index]].groupby("customer_id").amount.sum() / 2
    inc_prior = inflow[~recent[inflow.index]].groupby("customer_id").amount.sum() / 4
    sal_recent = tx[(tx.is_salary == 1) & recent].groupby("customer_id").size()
    debit = tx[tx.direction == "debit"]
    other = debit[debit.counterparty == "OTHER BANK TRANSFER"].groupby("customer_id").amount.sum()
    total_debit = debit.groupby("customer_id").amount.sum()
    inter = pd.DataFrame(db.query("SELECT customer_id, date, type, resolved, sentiment, topic FROM interactions ORDER BY interaction_id"))
    inter = inter[inter.customer_id.isin(cust.customer_id)]
    last90 = inter[pd.to_datetime(inter.date) >= AS_OF - pd.Timedelta(days=90)]
    comp = last90[last90.type == "complaint"].groupby("customer_id").size()
    unres = inter[(inter.type == "complaint") & (inter.resolved == 0)].groupby("customer_id").size()
    sent = last90.groupby("customer_id").sentiment.mean()
    logins = pd.DataFrame(db.query("SELECT customer_id, month, app_logins FROM app_logins ORDER BY customer_id, month"))
    l_recent = logins[logins.month >= "2026-07"].groupby("customer_id").app_logins.mean()
    l_prior = logins[logins.month < "2026-07"].groupby("customer_id").app_logins.mean()
    f = cust.set_index("customer_id")
    salaried = f.employer_type.isin(["salaried_private", "public"])
    out = pd.DataFrame(index=f.index)
    out["tenure_months"] = f.tenure_months
    out["salary_stopped"] = (salaried & (sal_recent.reindex(f.index).fillna(0) < 2)).astype(int)
    out["inflow_change"] = (inc_recent.reindex(f.index).fillna(0) / inc_prior.reindex(f.index).replace(0, np.nan)).fillna(1).clip(0, 3)
    out["other_bank_share"] = (other.reindex(f.index).fillna(0) / total_debit.reindex(f.index).replace(0, np.nan)).fillna(0)
    out["complaints_90d"] = comp.reindex(f.index).fillna(0)
    out["unresolved"] = unres.reindex(f.index).fillna(0)
    out["login_trend"] = (l_recent.reindex(f.index) / l_prior.reindex(f.index).replace(0, np.nan)).fillna(1).clip(0, 3)
    out["negative_sentiment"] = (-sent.reindex(f.index).fillna(0)).clip(lower=0)
    meta = f[["segment", "city", "region", "gender", "branch_id", "consent_marketing", "dnd", "monthly_income", "employer_type"]]
    top_topic = (inter[(inter.type == "complaint")].groupby("customer_id").topic
                 .agg(lambda s: s.value_counts().index[0]))
    meta = meta.assign(top_complaint=top_topic.reindex(f.index))
    return out[CHURN_FEATURES].join(meta)


def applicant_record(app_id: str) -> dict:
    """Join a loan application with the customer's profile and history."""
    a = db.one("SELECT * FROM loan_applications WHERE app_id=?", (app_id,))
    if not a:
        return None
    c = db.one("SELECT c.*, b.name AS branch FROM customers c JOIN branches b ON b.branch_id=c.branch_id "
               "WHERE customer_id=?", (a["customer_id"],))
    loans = db.query("SELECT loan_id, product, outstanding, monthly_installment, dpd, status FROM loans WHERE customer_id=?",
                     (a["customer_id"],))
    inflow = db.one("SELECT SUM(amount)/6.0 v FROM (SELECT DISTINCT txn_id, amount FROM transactions "
                    "WHERE customer_id=? AND direction='credit') x", (a["customer_id"],))["v"] or 0
    return {"application": a, "customer": c, "existing_loans": loans, "avg_monthly_inflow": round(inflow, 2)}
