"""Writes semantic.json: the natural-language question benchmark.

Each item fixes, before any run, what a careful analyst means by the question
(the gold interpretation) or the safe response when no valid interpretation exists.
The gold values were written by hand from the wording of each question, not from
the system's output.

expect:
  answer    the question maps to an approved metric; gold gives metric, breakdown, filters, period
  clarify   no approved metric fits, the wording is contradictory, or it needs a condition the
            semantic layer cannot express (a threshold). Answering anyway is a semantic error.
  blocked   the asking role is not entitled to the metric
"""
import json
from pathlib import Path

A = lambda m, d=(), f=None, p=None: {"metric": m, "dimensions": list(d), "filters": f or {}, "period": p}
Q = []


def add(cat, q, gold=None, expect="answer", role="analyst", note=""):
    Q.append({"id": f"S{len(Q) + 1:03d}", "category": cat, "question": q, "role": role, "expect": expect,
              "gold": gold, "note": note})


# 1. simple, single metric (16)
add("simple", "What is our total loan book?", A("loan_book"))
add("simple", "How many loans do we have?", A("loan_count"))
add("simple", "What is the delinquency rate?", A("delinquency_rate"))
add("simple", "Give me the NPL ratio", A("npl_ratio"))
add("simple", "How many customers does the bank have?", A("customer_count"))
add("simple", "What is the average monthly income of our customers?", A("avg_income"))
add("simple", "Total deposit inflow", A("deposit_inflow"))
add("simple", "How many transactions were processed?", A("txn_count"))
add("simple", "How many complaints have we received?", A("complaint_count"))
add("simple", "What is the loan approval rate?", A("approval_rate"))
add("simple", "How much cash was withdrawn from our ATMs?", A("atm_withdrawals"))
add("simple", "Show the non-performing loan ratio", A("npl_ratio"))
add("simple", "What is the outstanding loan balance?", A("loan_book"))
add("simple", "Number of customers", A("customer_count"))
add("simple", "Total complaints", A("complaint_count"))
add("simple", "Total ATM withdrawals", A("atm_withdrawals"))

# 2. grouping (16)
add("grouping", "Delinquency rate by branch", A("delinquency_rate", ["branch"]))
add("grouping", "Loan book by product", A("loan_book", ["product"]))
add("grouping", "NPL ratio by region", A("npl_ratio", ["region"]))
add("grouping", "Customers by city", A("customer_count", ["city"]))
add("grouping", "Deposit inflow by month", A("deposit_inflow", ["month"]))
add("grouping", "Complaints by topic", A("complaint_count", ["topic"]))
add("grouping", "Transactions by channel", A("txn_count", ["channel"]))
add("grouping", "ATM withdrawals by branch", A("atm_withdrawals", ["branch"]))
add("grouping", "Average income per segment", A("avg_income", ["segment"]))
add("grouping", "Show the loan count for each region", A("loan_count", ["region"]))
add("grouping", "Complaints by channel", A("complaint_count", ["channel"]))
add("grouping", "Delinquency rate by segment", A("delinquency_rate", ["segment"]))
add("grouping", "Deposit inflow by branch and month", A("deposit_inflow", ["branch", "month"]))
add("grouping", "Monthly trend of ATM withdrawals", A("atm_withdrawals", ["month"]))
add("grouping", "Loan book by region and product", A("loan_book", ["region", "product"]))
add("grouping", "Customers across each segment", A("customer_count", ["segment"]))

# 3. inclusion filters (12)
add("filter", "What is the delinquency rate in Lagos?", A("delinquency_rate", f={"city": "Lagos"}))
add("filter", "Delinquency rate by branch in Lagos", A("delinquency_rate", ["branch"], {"city": "Lagos"}))
add("filter", "Loan book for SME customers", A("loan_book", f={"segment": "SME"}))
add("filter", "Deposit inflow from retail customers by month", A("deposit_inflow", ["month"], {"segment": "Retail"}))
add("filter", "Complaints in Abuja by topic", A("complaint_count", ["topic"], {"city": "Abuja"}))
add("filter", "NPL ratio for the South West region", A("npl_ratio", f={"region": "South West"}))
add("filter", "Number of customers in Kano", A("customer_count", f={"city": "Kano"}))
add("filter", "ATM withdrawals in Port Harcourt by month", A("atm_withdrawals", ["month"], {"city": "Port Harcourt"}))
add("filter", "Average income of SME customers in Lagos", A("avg_income", f={"city": "Lagos", "segment": "SME"}))
add("filter", "Transactions in Ibadan by channel", A("txn_count", ["channel"], {"city": "Ibadan"}))
add("filter", "Loan count in the North West region by product", A("loan_count", ["product"], {"region": "North West"}))
add("filter", "Delinquency rate for retail customers in Enugu", A("delinquency_rate", f={"city": "Enugu", "segment": "Retail"}))

# 4. negation and exclusion (16)
add("negation", "What is the deposit inflow by month when the city is not Lagos?", A("deposit_inflow", ["month"], {"city": {"not": "Lagos"}}))
add("negation", "Deposit inflow by month except Lagos", A("deposit_inflow", ["month"], {"city": {"not": "Lagos"}}))
add("negation", "Delinquency rate by branch outside Lagos", A("delinquency_rate", ["branch"], {"city": {"not": "Lagos"}}))
add("negation", "Loan book excluding SME customers", A("loan_book", f={"segment": {"not": "SME"}}))
add("negation", "Complaints by topic for customers who are not retail", A("complaint_count", ["topic"], {"segment": {"not": "Retail"}}))
add("negation", "NPL ratio by region other than the South West", A("npl_ratio", ["region"], {"region": {"not": "South West"}}))
add("negation", "Customers by city apart from Abuja", A("customer_count", ["city"], {"city": {"not": "Abuja"}}))
add("negation", "ATM withdrawals by branch outside Lagos and Abuja", A("atm_withdrawals", ["branch"], {"city": {"not": ["Lagos", "Abuja"]}}))
add("negation", "Deposit inflow for all cities except Kano and Kaduna", A("deposit_inflow", f={"city": {"not": ["Kano", "Kaduna"]}}))
add("negation", "Delinquency rate for non-Lagos branches", A("delinquency_rate", f={"city": {"not": "Lagos"}}))
add("negation", "Transactions by channel besides Port Harcourt", A("txn_count", ["channel"], {"city": {"not": "Port Harcourt"}}))
add("negation", "Average income excluding the North West region", A("avg_income", f={"region": {"not": "North West"}}))
add("negation", "Loan count by product without SME customers", A("loan_count", ["product"], {"segment": {"not": "SME"}}))
add("negation", "Complaints in Lagos but not from SME customers", A("complaint_count", f={"city": "Lagos", "segment": {"not": "SME"}}))
add("negation", "Show deposit inflow by month, leaving out Lagos", A("deposit_inflow", ["month"], {"city": {"not": "Lagos"}}),
    note="'leaving out' is a negation phrase outside the parser's word list")
add("negation", "Delinquency rate for every city but Lagos", A("delinquency_rate", f={"city": {"not": "Lagos"}}),
    note="'but' as exclusion")

# 5. several values (and / or) (10)
add("and_or", "Loan book in Lagos and Abuja", A("loan_book", f={"city": ["Lagos", "Abuja"]}))
add("and_or", "Customers in Kano or Kaduna", A("customer_count", f={"city": ["Kano", "Kaduna"]}))
add("and_or", "Deposit inflow by month for Lagos and Ibadan", A("deposit_inflow", ["month"], {"city": ["Lagos", "Ibadan"]}))
add("and_or", "Delinquency rate in Enugu or Port Harcourt", A("delinquency_rate", f={"city": ["Enugu", "Port Harcourt"]}))
add("and_or", "Complaints by topic in the South West and South East regions", A("complaint_count", ["topic"], {"region": ["South West", "South East"]}))
add("and_or", "NPL ratio for retail and SME customers by region", A("npl_ratio", ["region"]),
    note="retail and SME together is every segment, so no segment filter")
add("and_or", "ATM withdrawals in Abuja, Kano and Kaduna", A("atm_withdrawals", f={"city": ["Abuja", "Kano", "Kaduna"]}))
add("and_or", "Transactions in Lagos or Abuja by channel", A("txn_count", ["channel"], {"city": ["Lagos", "Abuja"]}))
add("and_or", "Average income in North West or North Central", A("avg_income", f={"region": ["North West", "North Central"]}))
add("and_or", "Loan count for SME customers in Lagos and Abuja", A("loan_count", f={"city": ["Lagos", "Abuja"], "segment": "SME"}))

# 6. dates and periods (12)
add("period", "Deposit inflow last quarter", A("deposit_inflow", p="last quarter"))
add("period", "Complaints by topic last quarter", A("complaint_count", ["topic"], p="last quarter"))
add("period", "Transactions last month", A("txn_count", p="last month"))
add("period", "ATM withdrawals in August", A("atm_withdrawals", p="august"))
add("period", "Deposit inflow in September by branch", A("deposit_inflow", ["branch"], p="september"))
add("period", "Complaints in the last 30 days", A("complaint_count", p="last 30 days"))
add("period", "Transactions by channel in Q2", A("txn_count", ["channel"], p="q2"))
add("period", "ATM withdrawals by branch over the last 90 days", A("atm_withdrawals", ["branch"], p="last 90 days"))
add("period", "Deposit inflow in Lagos last month", A("deposit_inflow", f={"city": "Lagos"}, p="last month"))
add("period", "Complaints this quarter by channel", A("complaint_count", ["channel"], p="last quarter"),
    note="data ends 30 September, so this quarter is Q3, the same range as last quarter in the parser")
add("period", "Deposit inflow between July and September", A("deposit_inflow", p="last quarter"),
    note="July to September is Q3; a range phrase the parser does not read")
add("period", "Transactions in the past week", A("txn_count", p="last 7 days"),
    note="'past week' is a period phrase outside the parser's list")

# 7. ambiguous terms or no approved metric: the safe response is a clarifying question (10)
add("ambiguous", "Which branches are performing best?", expect="clarify", note="performance is not a defined metric")
add("ambiguous", "How healthy is our loan portfolio?", expect="clarify")
add("ambiguous", "What is our profit by branch?", expect="clarify", note="profit is not in the semantic layer")
add("ambiguous", "Show me customer satisfaction", expect="clarify")
add("ambiguous", "What is the average account balance?", expect="clarify", note="balances are not a defined metric")
add("ambiguous", "How is the weather in Lagos?", expect="clarify")
add("ambiguous", "What is the cost of funds?", expect="clarify")
add("ambiguous", "Show me risk by region", expect="clarify", note="risk could mean delinquency, NPL or AML risk")
add("ambiguous", "What is the market share in Abuja?", expect="clarify")
add("ambiguous", "Top 10 customers by revenue", expect="clarify")

# 8. thresholds and conditions the semantic layer cannot express (6)
add("threshold", "Loan book for loans above 1 million naira", expect="clarify", note="amount thresholds are not supported")
add("threshold", "How many customers earn more than 500,000 a month?", expect="clarify")
add("threshold", "Delinquency rate for loans 60 or more days past due", expect="clarify",
    note="the metric fixes the 30-day definition; a 60-day variant is not defined")
add("threshold", "Transactions over 5 million naira by channel", expect="clarify")
add("threshold", "Complaints from customers with more than two loans", expect="clarify")
add("threshold", "ATM withdrawals on weekends only", expect="clarify")

# 9. contradictory instructions (4)
add("contradiction", "Deposit inflow in Lagos excluding Lagos", expect="clarify")
add("contradiction", "Loan book for SME customers but not SME customers", expect="clarify")
add("contradiction", "Complaints in Abuja only, outside Abuja", expect="clarify")
add("contradiction", "Customers in Kano and not in Kano by city", expect="clarify")

# 10. permissions: the role is not entitled to the metric (8)
add("permission", "What is the deposit inflow by branch?", expect="blocked", role="loan_officer")
add("permission", "Complaints by topic", expect="blocked", role="loan_officer")
add("permission", "Average income by city", expect="blocked", role="loan_officer")
add("permission", "Customers by segment", expect="blocked", role="branch_ops")
add("permission", "Loan book by product", expect="blocked", role="branch_ops")
add("permission", "Delinquency rate by branch", expect="blocked", role="branch_ops")
add("permission", "NPL ratio by region", expect="blocked", role="branch_ops")
add("permission", "Approval rate", expect="blocked", role="branch_ops")

# entitled controls for the same roles (4)
add("permission", "Loan book by product", A("loan_book", ["product"]), role="loan_officer")
add("permission", "Delinquency rate by branch", A("delinquency_rate", ["branch"]), role="loan_officer")
add("permission", "ATM withdrawals by branch", A("atm_withdrawals", ["branch"]), role="branch_ops")
add("permission", "Transactions by channel", A("txn_count", ["channel"]), role="branch_ops")

if __name__ == "__main__":
    out = {"benchmark": "HAARA semantic interpretation benchmark", "version": "1.0", "items": Q,
           "counts": {c: sum(1 for q in Q if q["category"] == c) for c in dict.fromkeys(q["category"] for q in Q)}}
    Path(__file__).with_name("semantic.json").write_text(json.dumps(out, indent=1))
    print(len(Q), out["counts"])
