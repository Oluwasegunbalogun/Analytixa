"""Synthetic data and relational model for a fictional Nigerian bank.

All people, accounts and transactions are generated. Ground truth used only by
the evaluation (true churn, injected fraud, seeded data errors) is kept in
tables whose names start with an underscore. Agents never read those tables.
"""
import json
import numpy as np
import pandas as pd

from . import l1_data as db

RNG = np.random.default_rng(42)
PRODUCTS = [("SAV", "Savings", "deposit"), ("CUR", "Current", "deposit"), ("DOM", "Domiciliary", "deposit"),
            ("PL", "Personal Loan", "loan"), ("SME", "SME Working Capital", "loan")]
TODAY = pd.Timestamp("2026-09-30")
START = pd.Timestamp("2026-04-01")

FIRST_M = ["Adebayo", "Chinedu", "Emeka", "Ibrahim", "Musa", "Tunde", "Segun", "Kelechi", "Yusuf",
           "Olumide", "Obinna", "Femi", "Uche", "Abdullahi", "Kunle", "Dapo", "Ifeanyi", "Sani",
           "Tobi", "Bayo", "Chukwuma", "Gbenga", "Nnamdi", "Aliyu", "Seun"]
FIRST_F = ["Aisha", "Ngozi", "Funmi", "Chioma", "Amina", "Bisi", "Ada", "Halima", "Yetunde",
           "Kemi", "Nkechi", "Zainab", "Folake", "Ifeoma", "Hauwa", "Tolu", "Ebere", "Bukola",
           "Fatima", "Ronke", "Uju", "Sade", "Blessing", "Grace", "Hadiza"]
LAST = ["Okafor", "Adeyemi", "Bello", "Eze", "Ogunleye", "Abubakar", "Nwosu", "Balogun", "Okonkwo",
        "Lawal", "Adewale", "Umar", "Obi", "Afolabi", "Ibekwe", "Danjuma", "Oyelaran", "Chukwu",
        "Salami", "Ogundipe", "Nwachukwu", "Garba", "Akinola", "Onyeka", "Usman", "Fashola"]

BRANCHES = [
    ("BR01", "Ikeja", "Lagos", "South West"), ("BR02", "Lekki", "Lagos", "South West"),
    ("BR03", "Yaba", "Lagos", "South West"), ("BR04", "Surulere", "Lagos", "South West"),
    ("BR05", "Victoria Island", "Lagos", "South West"), ("BR06", "Wuse", "Abuja", "North Central"),
    ("BR07", "Garki", "Abuja", "North Central"), ("BR08", "Trans Amadi", "Port Harcourt", "South South"),
    ("BR09", "Sabon Gari", "Kano", "North West"), ("BR10", "Dugbe", "Ibadan", "South West"),
    ("BR11", "Ogui", "Enugu", "South East"), ("BR12", "Ahmadu Bello Way", "Kaduna", "North West"),
]

N_CUST = 1200


def _names(n, genders):
    out = []
    for g in genders:
        first = RNG.choice(FIRST_M if g == "M" else FIRST_F)
        out.append(f"{first} {RNG.choice(LAST)}")
    return out


def build_customers():
    gender = RNG.choice(["M", "F"], N_CUST)
    branch = RNG.choice([b[0] for b in BRANCHES], N_CUST,
                        p=[.13, .12, .1, .09, .1, .08, .06, .07, .06, .07, .06, .06])
    bmap = {b[0]: b for b in BRANCHES}
    segment = RNG.choice(["Retail", "SME"], N_CUST, p=[.8, .2])
    employer = np.where(segment == "SME", "self_employed",
                        RNG.choice(["salaried_private", "public", "self_employed", "informal"],
                                   N_CUST, p=[.4, .25, .2, .15]))
    income = np.round(RNG.lognormal(np.log(260000), .55, N_CUST) * np.where(segment == "SME", 2.2, 1), -3)
    age = RNG.integers(21, 66, N_CUST)
    tenure = RNG.integers(2, 120, N_CUST)
    churn_p = 1 / (1 + np.exp(-(-2.6 + 0.9 * (tenure < 12) + 0.6 * (employer == "informal")
                                + RNG.normal(0, 1, N_CUST))))
    churn = RNG.random(N_CUST) < churn_p
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUST + 1)],
        "name": _names(N_CUST, gender),
        "gender": gender, "age": age,
        "branch_id": branch,
        "city": [bmap[b][2] for b in branch],
        "region": [bmap[b][3] for b in branch],
        "segment": segment, "employer_type": employer,
        "monthly_income": income, "tenure_months": tenure,
        "bvn_verified": (RNG.random(N_CUST) < .95).astype(int),
        "consent_alt_data": (RNG.random(N_CUST) < .72).astype(int),
        "consent_marketing": (RNG.random(N_CUST) < .82).astype(int),
        "dnd": (RNG.random(N_CUST) < .08).astype(int),
        "phone": [f"080{RNG.integers(10000000, 99999999)}" for _ in range(N_CUST)],
    })
    # behavioural warning signs: most churners show them, a few loyal customers do too
    signal = (churn & (RNG.random(N_CUST) < .65)) | (~churn & (RNG.random(N_CUST) < .07))
    truth = pd.DataFrame({"customer_id": df.customer_id, "churned_next_90d": churn.astype(int),
                          "signal": signal.astype(int)})
    return df, truth


def build_accounts(cust):
    rows = []
    for c in cust.itertuples():
        rows.append((f"A{c.customer_id[1:]}1", c.customer_id,
                     "Current" if c.segment == "SME" else "Savings",
                     str((TODAY - pd.Timedelta(days=int(c.tenure_months * 30))).date()), "Active"))
        if RNG.random() < .25:
            rows.append((f"A{c.customer_id[1:]}2", c.customer_id, "Domiciliary",
                         str((TODAY - pd.Timedelta(days=int(RNG.integers(30, 900)))).date()), "Active"))
    return pd.DataFrame(rows, columns=["account_id", "customer_id", "product", "open_date", "status"])


def build_transactions(cust, truth):
    churn = dict(zip(truth.customer_id, truth.signal))
    days = pd.date_range(START, TODAY, freq="D")
    recs = []
    tid = 0
    for c in cust.itertuples():
        acct = f"A{c.customer_id[1:]}1"
        churner = churn[c.customer_id]
        inc = c.monthly_income
        for m_start in pd.date_range(START, TODAY, freq="MS"):
            late = churner and m_start >= pd.Timestamp("2026-08-01")
            # salary or business inflow
            if c.employer_type in ("salaried_private", "public"):
                if not (late and RNG.random() < .7):
                    tid += 1
                    d = m_start + pd.Timedelta(days=24, hours=9)
                    recs.append((f"T{tid:07d}", acct, c.customer_id, d, inc, "credit", "transfer",
                                 "EMPLOYER PAYROLL", "salary", 1))
            else:
                for _ in range(RNG.integers(3, 9)):
                    tid += 1
                    d = m_start + pd.Timedelta(days=int(RNG.integers(0, 28)), hours=int(RNG.integers(8, 19)))
                    amt = round(inc / 5 * RNG.uniform(.5, 1.5) * (.5 if late else 1), -2)
                    recs.append((f"T{tid:07d}", acct, c.customer_id, d, amt, "credit",
                                 RNG.choice(["transfer", "mobile_money", "branch_cash", "POS_settlement"]),
                                 "CUSTOMER PAYMENT", "business_inflow", 0))
            # spending
            n_debits = RNG.integers(8, 22)
            for _ in range(n_debits):
                tid += 1
                d = m_start + pd.Timedelta(days=int(RNG.integers(0, 28)), hours=int(RNG.integers(7, 22)),
                                           minutes=int(RNG.integers(0, 60)))
                ch = RNG.choice(["POS", "transfer", "ATM", "USSD", "bill_payment"], p=[.35, .3, .15, .1, .1])
                amt = round(inc * RNG.uniform(.01, .09), -2)
                cp = "OTHER BANK TRANSFER" if (ch == "transfer" and (late or RNG.random() < .2)) else \
                    {"POS": "MERCHANT", "transfer": "TRANSFER", "ATM": "ATM WITHDRAWAL",
                     "USSD": "AIRTIME/DATA", "bill_payment": "UTILITY"}[ch]
                if late and ch == "transfer":
                    amt = round(amt * 2.5, -2)
                recs.append((f"T{tid:07d}", acct, c.customer_id, d, amt, "debit", ch, cp, ch.lower(), 0))
    tx = pd.DataFrame(recs, columns=["txn_id", "account_id", "customer_id", "ts", "amount", "direction",
                                     "channel", "counterparty", "category", "is_salary"])
    tx = tx[tx.ts <= TODAY + pd.Timedelta(hours=23)]
    return tx, tid


def inject_financial_crime(cust, tx, tid):
    """Inject known structuring, mule and card-fraud patterns for the AML/fraud use case."""
    truth = []
    ids = list(cust.customer_id.sample(45, random_state=_rs(7)))
    recs = []
    # structuring: several cash deposits just below the reporting threshold within days
    for cid in ids[:8]:
        base = TODAY - pd.Timedelta(days=int(RNG.integers(1, 6)))
        for k in range(int(RNG.integers(3, 6))):
            tid += 1
            recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(hours=int(k * 7 + RNG.integers(0, 4))),
                         round(RNG.uniform(4.2e6, 4.95e6), -3), "credit", "branch_cash", "CASH DEPOSIT",
                         "cash_deposit", 0))
        truth.append((cid, "structuring"))
    # mule accounts: many inbound transfers, then fast outbound pass-through
    for cid in ids[8:13]:
        base = TODAY - pd.Timedelta(days=int(RNG.integers(1, 5)))
        total = 0
        for k in range(int(RNG.integers(8, 14))):
            tid += 1
            a = round(RNG.uniform(1.5e5, 6e5), -3)
            total += a
            recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(minutes=int(k * 11)),
                         a, "credit", "transfer", f"SENDER {RNG.integers(100, 999)}", "transfer_in", 0))
        tid += 1
        recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(hours=3),
                     round(total * .97, -3), "debit", "transfer", "OTHER BANK TRANSFER", "transfer", 0))
        truth.append((cid, "mule_pass_through"))
    # card fraud: large night-time POS spends far above the customer's norm
    for cid in ids[13:25]:
        inc = float(cust.set_index("customer_id").loc[cid, "monthly_income"])
        tid += 1
        recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid,
                     TODAY - pd.Timedelta(days=int(RNG.integers(0, 3))) + pd.Timedelta(hours=int(RNG.integers(1, 4))),
                     round(inc * RNG.uniform(1.5, 3.5), -2), "debit", "POS", "FOREIGN MERCHANT ONLINE", "pos", 0))
        truth.append((cid, "card_fraud"))
    # negative controls: benign look-alikes that a sound detector must NOT alert on
    benign = []
    for cid in ids[25:30]:      # two cash deposits in the band (the rule needs three)
        base = TODAY - pd.Timedelta(days=int(RNG.integers(1, 5)))
        for k in range(2):
            tid += 1
            recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(hours=k * 9 + 2),
                         round(RNG.uniform(4.2e6, 4.95e6), -3), "credit", "branch_cash", "CASH DEPOSIT", "cash_deposit", 0))
        benign.append((cid, "two_band_deposits"))
    for cid in ids[30:35]:      # many inbound transfers that stay in the account
        base = TODAY - pd.Timedelta(days=int(RNG.integers(1, 5)))
        total = 0
        for k in range(int(RNG.integers(9, 13))):
            tid += 1
            a = round(RNG.uniform(1.5e5, 4e5), -3)
            total += a
            recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(minutes=int(k * 13)),
                         a, "credit", "transfer", f"CUSTOMER {RNG.integers(100, 999)}", "transfer_in", 0))
        tid += 1
        recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(hours=5),
                     round(total * .2, -3), "debit", "transfer", "OTHER BANK TRANSFER", "transfer", 0))
        benign.append((cid, "inbound_burst_retained"))
    for cid in ids[35:40]:      # large daytime POS spend at a local merchant
        inc = float(cust.set_index("customer_id").loc[cid, "monthly_income"])
        tid += 1
        recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid,
                     TODAY - pd.Timedelta(days=int(RNG.integers(0, 3))) + pd.Timedelta(hours=int(RNG.integers(11, 17))),
                     round(inc * RNG.uniform(1.5, 3.0), -2), "debit", "POS", "LOCAL APPLIANCE STORE", "pos", 0))
        benign.append((cid, "daytime_large_pos"))
    for cid in ids[40:45]:      # real structuring, but 20 to 30 days ago: outside a 7-day window
        base = TODAY - pd.Timedelta(days=int(RNG.integers(20, 30)))
        for k in range(4):
            tid += 1
            recs.append((f"T{tid:07d}", f"A{cid[1:]}1", cid, base + pd.Timedelta(hours=k * 8 + 1),
                         round(RNG.uniform(4.2e6, 4.95e6), -3), "credit", "branch_cash", "CASH DEPOSIT", "cash_deposit", 0))
        benign.append((cid, "structuring_outside_window"))
    extra = pd.DataFrame(recs, columns=tx.columns)
    return (pd.concat([tx, extra], ignore_index=True), pd.DataFrame(truth, columns=["customer_id", "pattern"]),
            pd.DataFrame(benign, columns=["customer_id", "pattern"]))


def build_loans(cust):
    rows = []
    lid = 0
    for c in cust.itertuples():
        if RNG.random() < .4:
            lid += 1
            amt = round(c.monthly_income * RNG.uniform(2, 8), -3)
            tenor = int(RNG.choice([6, 12, 18, 24, 36]))
            inst = round(amt * (1 + .28 * tenor / 12) / tenor, -2)
            dpd = int(RNG.choice([0, 0, 0, 0, 0, 0, 0, 15, 35, 95]))
            rows.append((f"L{lid:05d}", c.customer_id, "Personal Loan" if c.segment == "Retail" else "SME Working Capital",
                         amt, tenor, inst, round(amt * RNG.uniform(.2, .9), -3), dpd,
                         "default" if dpd >= 90 else ("arrears" if dpd >= 30 else "current"),
                         c.branch_id))
    return pd.DataFrame(rows, columns=["loan_id", "customer_id", "product", "principal", "tenor_months",
                                       "monthly_installment", "outstanding", "dpd", "status", "branch_id"])


def build_history(n=5000, rng=None, shift=None):
    """Historical repaid/defaulted loans used to train the credit model.
    shift changes the population for robustness tests, for example a downturn."""
    R = rng or RNG
    sh = {"income_scale": 1.0, "dti_shift": 0.0, "informal_p": .2, "bureau_p": .65, "intercept": 0.0, **(shift or {})}
    income = R.lognormal(np.log(260000 * sh["income_scale"]), .55, n)
    dti = np.clip(R.normal(.33 + sh["dti_shift"], .14, n), .02, .95)
    tenure = R.integers(2, 120, n)
    util = np.clip(R.beta(6, 2, n), 0, 1)
    airtime = np.round(R.lognormal(np.log(6000), .6, n), -2)
    mm = np.round(income * R.uniform(.1, 1.2, n), -2)
    has_bureau = R.random(n) < sh["bureau_p"]
    bureau = np.where(has_bureau, np.clip(R.normal(640, 70, n), 300, 850), np.nan)
    prior = (R.random(n) < .08).astype(int)
    informal = (R.random(n) < sh["informal_p"]).astype(int)
    z = (-2.2 + sh["intercept"] + 3.2 * (dti - .33) - .9 * (np.log(income) - np.log(260000)) - .012 * (tenure - 40)
         - 2.4 * (util - .75) + 1.3 * prior + .45 * informal
         - np.where(has_bureau, (np.nan_to_num(bureau) - 640) / 90, 0) + R.normal(0, .6, n))
    default = (R.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"monthly_income": income.round(-3), "dti": dti.round(3), "tenure_months": tenure,
                         "utility_ontime_ratio": util.round(3), "airtime_monthly_avg": airtime,
                         "mobile_money_inflow_avg": mm, "bureau_score": np.round(bureau),
                         "prior_default": prior, "informal_income": informal, "defaulted": default})


def build_applications(cust, loans):
    pick = cust.sample(200, random_state=_rs(11)).reset_index(drop=True)
    inst = loans.groupby("customer_id").monthly_installment.sum()
    rows, errors = [], []
    for i, c in enumerate(pick.itertuples(), 1):
        amt = round(c.monthly_income * RNG.uniform(.5, 4), -3)
        tenor = int(RNG.choice([6, 12, 18, 24, 36, 48], p=[.05, .25, .2, .25, .2, .05]))
        has_bureau = RNG.random() < .6
        rows.append({
            "app_id": f"APP-{1000 + i}", "customer_id": c.customer_id,
            "applied_on": str((TODAY - pd.Timedelta(days=int(RNG.integers(0, 20)))).date()),
            "product": "Personal Loan" if c.segment == "Retail" else "SME Working Capital",
            "amount": amt, "tenor_months": tenor,
            "purpose": RNG.choice(["School fees", "Rent", "Business stock", "Medical", "Car repair", "Home improvement"]),
            "collateral_value": round(amt * RNG.uniform(.8, 1.8), -3) if RNG.random() < .3 else 0,
            "declared_income": c.monthly_income,
            "existing_installments": float(inst.get(c.customer_id, 0)),
            "utility_ontime_ratio": round(float(np.clip(RNG.beta(6, 2), 0, 1)), 3),
            "airtime_monthly_avg": float(round(RNG.lognormal(np.log(6000), .6), -2)),
            "mobile_money_inflow_avg": float(round(c.monthly_income * RNG.uniform(.1, 1.2), -2)),
            "bureau_score": float(round(np.clip(RNG.normal(640, 70), 300, 850))) if has_bureau else None,
            "status": "pending",
        })
    apps = pd.DataFrame(rows)
    # seeded data errors, recorded for evaluation
    for idx in [5, 23, 61, 97, 140, 188]:
        apps.loc[idx, "declared_income"] = None
        errors.append((apps.loc[idx, "app_id"], "missing_income"))
    for idx in [12, 77, 150]:
        apps.loc[idx, "amount"] = -abs(apps.loc[idx, "amount"])
        errors.append((apps.loc[idx, "app_id"], "negative_amount"))
    for idx in [33, 120]:
        apps.loc[idx, "utility_ontime_ratio"] = 1.7
        errors.append((apps.loc[idx, "app_id"], "ratio_out_of_range"))
    return apps, pd.DataFrame(errors, columns=["record_id", "error"])


def build_interactions(cust, truth):
    churn = dict(zip(truth.customer_id, truth.signal))
    rows = []
    iid = 0
    for c in cust.itertuples():
        n = RNG.poisson(2.6 if churn[c.customer_id] else 1.2)
        for _ in range(n):
            iid += 1
            topic = RNG.choice(["failed transfer", "card declined", "charges and fees", "app login issue",
                                "loan enquiry", "account statement"],
                               p=[.25, .15, .25, .15, .1, .1] if churn[c.customer_id] else [.1, .1, .1, .1, .3, .3])
            neg = topic in ("failed transfer", "card declined", "charges and fees", "app login issue")
            rows.append((f"I{iid:06d}", c.customer_id,
                         str((TODAY - pd.Timedelta(days=int(RNG.integers(0, 150)))).date()),
                         RNG.choice(["call centre", "chat", "branch", "email"]),
                         "complaint" if neg else "enquiry", topic,
                         round(float(RNG.uniform(-.9, -.2) if neg else RNG.uniform(0, .8)), 2),
                         int(RNG.random() < (.4 if churn[c.customer_id] else .8))))
    # app logins per month, falling for churners
    logins = []
    for c in cust.itertuples():
        base = RNG.poisson(14)
        for m in pd.date_range(START, TODAY, freq="MS"):
            drop = churn[c.customer_id] and m >= pd.Timestamp("2026-07-01") and RNG.random() < .8
            logins.append((c.customer_id, str(m.date())[:7], int(max(0, base * (.3 if drop else 1) + RNG.normal(0, 2)))))
    return (pd.DataFrame(rows, columns=["interaction_id", "customer_id", "date", "channel", "type", "topic",
                                        "sentiment", "resolved"]),
            pd.DataFrame(logins, columns=["customer_id", "month", "app_logins"]))


def build_atms():
    atms, hist = [], []
    days = pd.date_range(TODAY - pd.Timedelta(days=179), TODAY, freq="D")
    for i in range(20):
        b = BRANCHES[i % len(BRANCHES)]
        atm_id = f"ATM{i + 1:02d}"
        cap = int(RNG.choice([12, 16, 20])) * 1_000_000
        base = RNG.uniform(1.8e6, 4.5e6)
        atms.append((atm_id, b[0], f"{b[1]} {'Branch' if i < 12 else 'Mall'}", b[2], cap))
        bal = cap * .7
        for d in days:
            dow = [1.0, .9, .9, .95, 1.25, 1.35, .7][d.dayofweek]
            pay = 1.45 if d.day >= 25 or d.day <= 2 else 1.0
            w = max(0, base * dow * pay * RNG.normal(1, .12))
            bal -= w
            refill = 0
            if bal < cap * .25:
                refill = cap - bal
                bal = cap
            hist.append((atm_id, str(d.date()), round(w, -3), round(refill, -3), round(bal, -3)))
    return (pd.DataFrame(atms, columns=["atm_id", "branch_id", "location", "city", "capacity"]),
            pd.DataFrame(hist, columns=["atm_id", "date", "withdrawals", "replenished", "closing_balance"]))


# ── relational model ──────────────────────────────────────────────
# Money is numeric(18,2), dates are date, event times are timestamp.
# Foreign keys tie every record to its customer, branch, product or ATM.

BANK_DDL = """
CREATE TABLE company (
  company_id  smallint PRIMARY KEY DEFAULT 1 CHECK (company_id = 1),
  name        text NOT NULL,
  short_name  text NOT NULL,
  country     text NOT NULL,
  currency    char(3) NOT NULL,
  data_as_of  date NOT NULL,
  synthetic   boolean NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE branches (
  branch_id text PRIMARY KEY,
  name      text NOT NULL,
  city      text NOT NULL,
  region    text NOT NULL
);

CREATE TABLE products (
  product_code text PRIMARY KEY,
  name         text NOT NULL UNIQUE,
  category     text NOT NULL CHECK (category IN ('deposit','loan'))
);

CREATE TABLE customers (
  customer_id       text PRIMARY KEY,
  name              text NOT NULL,
  gender            char(1) NOT NULL CHECK (gender IN ('M','F')),
  age               smallint NOT NULL CHECK (age BETWEEN 18 AND 100),
  branch_id         text NOT NULL REFERENCES branches(branch_id),
  city              text NOT NULL,
  region            text NOT NULL,
  segment           text NOT NULL CHECK (segment IN ('Retail','SME')),
  employer_type     text NOT NULL,
  monthly_income    numeric(18,2) NOT NULL,
  tenure_months     integer NOT NULL,
  bvn_verified      smallint NOT NULL CHECK (bvn_verified IN (0,1)),
  consent_alt_data  smallint NOT NULL CHECK (consent_alt_data IN (0,1)),
  consent_marketing smallint NOT NULL CHECK (consent_marketing IN (0,1)),
  dnd               smallint NOT NULL CHECK (dnd IN (0,1)),
  phone             text
);
CREATE INDEX ix_cust_branch ON customers(branch_id);

CREATE TABLE accounts (
  account_id  text PRIMARY KEY,
  customer_id text NOT NULL REFERENCES customers(customer_id),
  product     text NOT NULL REFERENCES products(name),
  open_date   date NOT NULL,
  status      text NOT NULL
);
CREATE INDEX ix_acct_cust ON accounts(customer_id);

-- entry_id is the ledger row. txn_id is the switch reference, which can arrive
-- twice when a feed is replayed. The data quality tool detects those repeats.
CREATE TABLE transactions (
  entry_id     bigserial PRIMARY KEY,
  txn_id       text NOT NULL,
  account_id   text NOT NULL REFERENCES accounts(account_id),
  customer_id  text NOT NULL REFERENCES customers(customer_id),
  ts           timestamp NOT NULL,
  amount       numeric(18,2) NOT NULL,
  direction    text NOT NULL CHECK (direction IN ('credit','debit')),
  channel      text NOT NULL,
  counterparty text,
  category     text,
  is_salary    smallint NOT NULL DEFAULT 0
);
CREATE INDEX ix_tx_cust ON transactions(customer_id);
CREATE INDEX ix_tx_ts ON transactions(ts);
CREATE INDEX ix_tx_txn ON transactions(txn_id);

CREATE VIEW transactions_clean AS
  SELECT DISTINCT ON (txn_id) txn_id, account_id, customer_id, ts, amount, direction, channel,
         counterparty, category, is_salary
  FROM transactions ORDER BY txn_id, entry_id;

CREATE TABLE loans (
  loan_id             text PRIMARY KEY,
  customer_id         text NOT NULL REFERENCES customers(customer_id),
  product             text NOT NULL REFERENCES products(name),
  principal           numeric(18,2) NOT NULL,
  tenor_months        smallint NOT NULL,
  monthly_installment numeric(18,2) NOT NULL,
  outstanding         numeric(18,2) NOT NULL,
  dpd                 integer NOT NULL DEFAULT 0,
  status              text NOT NULL CHECK (status IN ('current','arrears','default')),
  branch_id           text NOT NULL REFERENCES branches(branch_id)
);
CREATE INDEX ix_loans_cust ON loans(customer_id);

-- closed loans with known outcomes, used to train the credit model
CREATE TABLE loan_history (
  history_id              serial PRIMARY KEY,
  monthly_income          numeric(18,2),
  dti                     numeric(6,3),
  tenure_months           integer,
  utility_ontime_ratio    numeric(6,3),
  airtime_monthly_avg     numeric(18,2),
  mobile_money_inflow_avg numeric(18,2),
  bureau_score            numeric(5,0),
  prior_default           smallint,
  informal_income         smallint,
  defaulted               smallint NOT NULL
);

-- applications arrive from the origination channel as captured, errors included
CREATE TABLE loan_applications (
  app_id                  text PRIMARY KEY,
  customer_id             text NOT NULL REFERENCES customers(customer_id),
  applied_on              date NOT NULL,
  product                 text NOT NULL REFERENCES products(name),
  amount                  numeric(18,2),
  tenor_months            smallint,
  purpose                 text,
  collateral_value        numeric(18,2) DEFAULT 0,
  declared_income         numeric(18,2),
  existing_installments   numeric(18,2) DEFAULT 0,
  utility_ontime_ratio    numeric(6,3),
  airtime_monthly_avg     numeric(18,2),
  mobile_money_inflow_avg numeric(18,2),
  bureau_score            numeric(5,0),
  status                  text NOT NULL DEFAULT 'pending'
);
CREATE INDEX ix_app_cust ON loan_applications(customer_id);

CREATE TABLE interactions (
  interaction_id text PRIMARY KEY,
  customer_id    text NOT NULL REFERENCES customers(customer_id),
  date           date NOT NULL,
  channel        text,
  type           text CHECK (type IN ('complaint','enquiry')),
  topic          text,
  sentiment      numeric(4,2),
  resolved       smallint
);
CREATE INDEX ix_int_cust ON interactions(customer_id);

CREATE TABLE app_logins (
  customer_id text NOT NULL REFERENCES customers(customer_id),
  month       char(7) NOT NULL,
  app_logins  integer NOT NULL,
  PRIMARY KEY (customer_id, month)
);

CREATE TABLE atms (
  atm_id    text PRIMARY KEY,
  branch_id text NOT NULL REFERENCES branches(branch_id),
  location  text NOT NULL,
  city      text NOT NULL,
  capacity  numeric(18,2) NOT NULL
);

CREATE TABLE atm_cash (
  atm_id          text NOT NULL REFERENCES atms(atm_id),
  date            date NOT NULL,
  withdrawals     numeric(18,2) NOT NULL,
  replenished     numeric(18,2) NOT NULL,
  closing_balance numeric(18,2) NOT NULL,
  PRIMARY KEY (atm_id, date)
);

-- ground truth for the evaluation only; agents never read these
CREATE TABLE _truth_churn (customer_id text PRIMARY KEY REFERENCES customers(customer_id), churned_next_90d smallint);
CREATE TABLE _truth_crime (customer_id text REFERENCES customers(customer_id), pattern text);
CREATE TABLE _truth_seeded_errors (record_id text, error text);
CREATE TABLE _truth_benign (customer_id text REFERENCES customers(customer_id), pattern text);

-- systems outside the analytics platform, simulated as tables
CREATE TABLE ext_loan_decisions (id bigserial PRIMARY KEY, ts timestamptz, run_id text,
  app_id text REFERENCES loan_applications(app_id), decision text, amount numeric(18,2),
  reason_codes text, decided_by text, policy_version integer);
CREATE TABLE ext_cases (case_id text PRIMARY KEY, ts timestamptz, run_id text, customer_id text,
  typology text, risk_score numeric(6,3), status text, narrative text);
CREATE TABLE ext_holds (id bigserial PRIMARY KEY, ts timestamptz, run_id text, customer_id text,
  txn_id text, amount numeric(18,2), hours integer, status text, authorised_by text);
CREATE TABLE ext_str_filings (id bigserial PRIMARY KEY, ts timestamptz, run_id text,
  case_id text, customer_id text, filed_by text, body text);
CREATE TABLE ext_campaigns (id bigserial PRIMARY KEY, ts timestamptz, run_id text,
  customer_id text, offer text, channel text, cost numeric(18,2), authorised_by text);
CREATE TABLE ext_cash_orders (id bigserial PRIMARY KEY, ts timestamptz, run_id text,
  atm_id text REFERENCES atms(atm_id), delivery_date date, amount numeric(18,2),
  authorised_by text, within_envelope integer);
"""

BANK_TABLES = ["ext_cash_orders", "ext_campaigns", "ext_str_filings", "ext_holds", "ext_cases",
               "ext_loan_decisions", "_truth_benign", "_truth_seeded_errors", "_truth_crime", "_truth_churn",
               "atm_cash", "atms", "app_logins", "interactions", "loan_applications", "loan_history",
               "loans", "transactions", "accounts", "customers", "products", "branches", "company"]


def _rows(df: pd.DataFrame):
    """DataFrame rows as plain Python values with NaN turned into NULL."""
    cols = [df[c].tolist() for c in df.columns]
    for row in zip(*cols):
        yield tuple(None if isinstance(v, float) and v != v else v for v in row)


def _load(table, df):
    db.copy_rows(table, list(df.columns), _rows(df))


def company_name() -> str:
    from . import config
    return config.get("COMPANY_NAME", "Lagoon Trust Bank")


SEED = 42


def _rs(k):
    """Pandas sampling seeds: the original values for the default bank, derived values otherwise."""
    return k if SEED == 42 else SEED * 1000 + k


def seed(force: bool = False, progress=None, seed_value: int = 42) -> dict:
    """Build the bank. seed_value fixes every random draw, so the same value gives the same bank."""
    global RNG, SEED
    db.init_core()
    if not force and db.table_exists("customers"):
        return {"seeded": False}
    say = progress or (lambda m: None)
    SEED = int(seed_value)
    RNG = np.random.default_rng(SEED)     # same seed, same bank, in every schema
    db.script("DROP VIEW IF EXISTS transactions_clean; " +
              "; ".join(f"DROP TABLE IF EXISTS {t} CASCADE" for t in BANK_TABLES))
    db.script(BANK_DDL)
    say("Generating customers, accounts and transactions")
    cust, truth = build_customers()
    accts = build_accounts(cust)
    tx, tid = build_transactions(cust, truth)
    tx, crime, benign = inject_financial_crime(cust, tx, tid)
    tx["ts"] = tx.ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    # seeded transaction errors: the same switch reference delivered twice
    dups = tx.sample(25, random_state=_rs(3))
    tx = pd.concat([tx, dups], ignore_index=True)
    loans = build_loans(cust)
    hist = build_history()
    apps, app_err = build_applications(cust, loans)
    inter, logins = build_interactions(cust, truth)
    atms, cash = build_atms()
    seeded = pd.concat([app_err, pd.DataFrame({"record_id": dups.txn_id, "error": "duplicate_transaction"})])
    say("Writing the bank to the database")
    db.execute("INSERT INTO company (name, short_name, country, currency, data_as_of) VALUES (?,?,?,?,?)",
               (company_name(), "".join(w[0] for w in company_name().split()[:3]).upper(), "Nigeria", "NGN",
                str(TODAY.date())))
    _load("branches", pd.DataFrame(BRANCHES, columns=["branch_id", "name", "city", "region"]))
    _load("products", pd.DataFrame(PRODUCTS, columns=["product_code", "name", "category"]))
    _load("customers", cust)
    _load("accounts", accts)
    _load("transactions", tx)
    _load("loans", loans)
    _load("loan_history", hist)
    _load("loan_applications", apps)
    _load("interactions", inter)
    _load("app_logins", logins)
    _load("atms", atms)
    _load("atm_cash", cash)
    _load("_truth_churn", truth[["customer_id", "churned_next_90d"]])
    _load("_truth_crime", crime)
    _load("_truth_seeded_errors", seeded)
    _load("_truth_benign", benign)
    db.script("ANALYZE")
    counts = {t: db.one(f"SELECT COUNT(*) n FROM {t}")["n"] for t in
              ["customers", "accounts", "transactions", "loans", "loan_applications", "interactions", "atms", "atm_cash"]}
    db.execute("COMMENT ON TABLE company IS 'synthetic bank, seed " + str(SEED) + "'")
    db.ledger("data_seeded", payload={**counts, "seed": SEED})
    return {"seeded": True, **counts}
