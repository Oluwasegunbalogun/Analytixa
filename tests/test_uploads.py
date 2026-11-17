"""Regression test for "Your data" (uploaded datasets).

Run from the project folder:  python tests/test_uploads.py
It uploads samples/sample_microfinance_loans.csv as the first active analyst, checks how the
columns were read, that personal columns were masked, how 24 questions are interpreted and
that another person cannot reach the data. It deletes the dataset at the end.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import auth, datasets as DS, l1_data as db, l2_policy as P  # noqa: E402

EXPECT = {
    "Total amount by branch": "Total Amount, by Branch, all rows.",
    "Amount by month": "Total Amount, by month of Disbursed On, all rows.",
    "Average outstanding by state": "Average Outstanding, by State, all rows.",
    "How many rows by branch": "Number of rows, by Branch, all rows.",
    "Top 5 branch by total amount": "Total Amount, by Branch, top 5, all rows.",
    "What is the total outstanding for branches outside Lagos?": "Total Outstanding, State is not Lagos, all rows.",
    "Total amount by product except Trader loan": "Total Amount, by Product, Product is not Trader loan, all rows.",
    "Average days past due by product in 2025": "Average Days Past Due, by Product, Disbursed On in 2025.",
    "Number of loans with amount above 1 million by state":
        "Number of distinct Loan ID, by State, Amount above 1,000,000, all rows.",
    "Total amount disbursed in the last 3 months": "Total Amount, Disbursed On in the last 3 months.",
    "Outstanding by state and product": "Total Outstanding, by State and Product, all rows.",
    "How many distinct branches are there?": "Number of distinct Branch, all rows.",
    "Total amount in Lagos and Abuja": "Total Amount, State is Lagos or Abuja, all rows.",
    "Median amount for salary advance": "Median Amount, Product is Salary advance, all rows.",
    "Monthly disbursement trend for Kano": "Total Amount, by month of Disbursed On, State is Kano, all rows.",
    "Bottom 3 products by average amount": "Average Amount, by Product, bottom 3, all rows.",
    "Amount by status excluding Closed": "Total Amount, by Status, Status is not Closed, all rows.",
    "What is the largest loan amount?": "Largest Amount, all rows.",
    "Total outstanding where days past due is at least 30 by branch":
        "Total Outstanding, by Branch, Days Past Due at least 30, all rows.",
    "Count of loans by quarter of disbursed on": "Number of distinct Loan ID, by quarter of Disbursed On, all rows.",
    "Total amount between March and June 2026": "Total Amount, Disbursed On in March to June 2026.",
    "Total outstanding for Ikeja, Lekki and Yaba": "Total Outstanding, Branch is Ikeja or Lekki or Yaba, all rows.",
    "Loans not in Lagos or Abuja by product": "Number of rows, by Product, State is neither Lagos nor Abuja, all rows.",
    "How healthy is the portfolio?": None,           # no measure fits, so the agent must ask
}


def main():
    db.init_core(); auth.init(); DS.init(); P.ensure_defaults()
    owner = db.one("SELECT email, role FROM app_users WHERE role='analyst' AND active ORDER BY user_id LIMIT 1")
    other = db.one("SELECT email, role FROM app_users WHERE role<>'analyst' AND active ORDER BY user_id LIMIT 1")
    data = (ROOT / "samples" / "sample_microfinance_loans.csv").read_bytes()
    d = DS.create("sample_microfinance_loans.csv", data, owner, "Regression test")
    fails = []
    try:
        cols = {c["label"]: c for c in d["columns"]}
        for label, role in (("Customer Name", "personal"), ("Phone", "personal"), ("Amount", "measure"),
                            ("Disbursed On", "date"), ("Branch", "category"), ("Loan ID", "identifier")):
            if cols[label]["role"] != role:
                fails.append(f"{label} read as {cols[label]['role']}, expected {role}")
        if not all(str(r["customer_name"]).startswith("P-") for r in d["preview"]):
            fails.append("customer names were stored unmasked")
        r = DS._row(d["dataset_id"])
        vals = DS.category_values(r)
        md = max(c.get("max", "") for c in r["profile"]["columns"] if c["type"] == "date")
        for q, want in EXPECT.items():
            i = DS.interpret(q, r["columns"], vals, md)
            got = DS.describe(DS.validate(i, r["columns"]), r["columns"]) if i.get("metrics") else None
            if got != want:
                fails.append(f"{q!r}: got {got!r}, expected {want!r}")
            elif got:
                plan = DS.validate(i, r["columns"])
                sql, params, _, _, _ = DS.build_sql(r, plan)
                if DS.check_sql(sql, plan, r["table_name"]):
                    fails.append(f"{q!r}: compiled SQL does not match its plan")
                db.query(sql, params)
        if other and not DS.access_problem(d["dataset_id"], other["email"]):
            fails.append("another person could reach the dataset")
        try:
            DS.validate({"metrics": [{"agg": "count"}], "dimensions": ["customer_name"]}, r["columns"])
            fails.append("a personal column was accepted as a breakdown")
        except ValueError:
            pass
    finally:
        DS.delete(d["dataset_id"], owner)
    print(f"{len(EXPECT) + 9 - len(fails)} of {len(EXPECT) + 9} checks passed")
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
