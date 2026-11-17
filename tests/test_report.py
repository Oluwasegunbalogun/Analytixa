"""Tests for the full analysis report on an uploaded dataset.

Run from the project folder:   python -m unittest tests.test_report -v
The fixture tests/fixtures/ads_synthetic.csv is invented data (fixed seed). It has blank clicks, rows with zero
impressions and one duplicate row, so the edge cases are exercised.
Unit tests need no database. Integration tests need the database in config.env and are skipped without it.
"""
import json
import sys
import time
import unittest
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import dataset_report as RP  # noqa: E402
from app import datasets as DS  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ads_synthetic.csv"
QUESTION = """Comprehensive advertising performance analysis of ad_id, account_id, ad_spend, ad_impressions, ad_clicks and ad_date.

1. Overall performance: total spend, total impressions, total clicks, average daily spend.
2. Efficiency:
- Click-through rate (CTR) = total clicks divided by total impressions, multiplied by 100.
- Cost per click (CPC) = total ad spend divided by total clicks.
- Cost per thousand impressions (CPM) = total ad spend divided by total impressions, multiplied by 1,000.
- Spend per ad and spend per account.
Handle zero denominators explicitly. Do not report undefined ratios as zero.
3. Account-level performance across accounts.
4. Ad-level performance for each ad_id.
5. Trends over time by day, week and month.
6. Relationship between spend and engagement, with correlation.
7. Anomaly and data quality detection: missing values, duplicates.
8. Concentration and risk.
9. Five actionable recommendations. Do not invent conversion rates or revenue.
10. Executive summary."""
COLS = [{"name": "ad_id", "label": "ad_id", "role": "identifier", "type": "text"},
        {"name": "account_id", "label": "account_id", "role": "identifier", "type": "text"},
        {"name": "ad_spend", "label": "ad_spend", "role": "measure", "type": "number"},
        {"name": "ad_impressions", "label": "ad_impressions", "role": "measure", "type": "number"},
        {"name": "ad_clicks", "label": "ad_clicks", "role": "measure", "type": "number"},
        {"name": "ad_date", "label": "ad_date", "role": "date", "type": "date"}]
DISTINCT = {"ad_id": 80, "account_id": 3, "ad_date": 120}


def frame():
    return pd.read_csv(FIXTURE, dtype={"ad_id": str, "account_id": str})


def plan():
    return RP.validate(RP.build_plan(QUESTION, COLS, {}, DISTINCT), COLS)


def section(rep, sid):
    return next(s for s in rep["sections"] if s["id"] == sid)


class UnitReport(unittest.TestCase):
    """Detection, plan building, formulas and edge cases. No database."""

    def test_detects_a_full_analysis(self):
        self.assertTrue(RP.is_report(QUESTION))
        for q in ("Total ad_spend by account_id", "What is the CTR by month?", "Top 10 ads by clicks"):
            self.assertFalse(RP.is_report(q), q)

    def test_ratios_parsed_from_the_formulas(self):
        got = {r["name"]: (r["num"], r["den"], r["mult"], r["better"]) for r in plan()["ratios"]}
        self.assertEqual(got, {"CTR": ("ad_clicks", "ad_impressions", 100.0, "higher"),
                               "CPC": ("ad_spend", "ad_clicks", 1.0, "lower"),
                               "CPM": ("ad_spend", "ad_impressions", 1000.0, "lower")})

    def test_plan_structure(self):
        p = plan()
        self.assertEqual(p["measures"], ["ad_spend", "ad_impressions", "ad_clicks"])
        self.assertEqual((p["groups"], p["items"], p["date"]), (["account_id"], ["ad_id"], "ad_date"))
        self.assertEqual(p["filters"], {})                               # no silent filters
        self.assertEqual(p["sections"][0], "overall")
        self.assertEqual(p["sections"][-1], "summary")
        self.assertIn("revenue", p["unavailable"])                     # named, not estimated
        self.assertEqual({(x["measure"], x["per"]) for x in p["per"]}, {("ad_spend", "ad_id"), ("ad_spend", "account_id")})

    def test_plan_hash_ignores_version_and_notes(self):
        p = plan()
        self.assertEqual(RP.plan_hash(p), RP.plan_hash({**p, "version": 4, "notes": ["x"]}))
        self.assertNotEqual(RP.plan_hash(p), RP.plan_hash({**p, "groups": []}))

    def test_validate_drops_unknown_columns(self):
        p = RP.validate({**plan(), "measures": ["ad_spend", "nope"], "groups": ["ad_spend"]}, COLS)
        self.assertEqual((p["measures"], p["groups"]), (["ad_spend"], []))
        self.assertEqual(p["ratios"], [])                              # ratios need both of their measures
        with self.assertRaises(ValueError):
            RP.validate({**plan(), "measures": ["ad_date"]}, COLS)

    def test_zero_denominator_is_undefined_not_zero(self):
        self.assertIsNone(RP._ratio(10, 0, 100))
        self.assertIsNone(RP._ratio(10, float("nan"), 100))
        self.assertEqual(RP._ratio(5, 1000, 100), 0.5)
        df = frame()
        df.loc[df["account_id"] == "9000000000000003", "ad_impressions"] = 0   # one account with no impressions
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)              # no division warnings either
            rep = RP.compute(df, plan(), COLS, {"issues": []})
        t = section(rep, "group:account_id")["tables"][0]
        row = next(r for r in t["rows"] if r[0] == "9000000000000003")
        self.assertIsNone(row[t["columns"].index("CTR")])
        self.assertIsNone(row[t["columns"].index("CPM")])
        self.assertIsNotNone(row[t["columns"].index("CPC")])           # clicks are still there

    def test_totals_and_ratios_match_the_raw_file(self):
        df = frame()
        rep = RP.compute(df.copy(), plan(), COLS, {"issues": []})
        for m in ("ad_spend", "ad_impressions", "ad_clicks"):
            self.assertAlmostEqual(rep["totals"][m], round(float(df[m].sum()), 2), places=2)
        both = df.dropna(subset=["ad_clicks", "ad_impressions"])
        ctr = both["ad_clicks"].sum() / both["ad_impressions"].sum() * 100     # ratio of sums, not mean of ratios
        eff = section(rep, "efficiency")["tables"][0]
        got = next(r for r in eff["rows"] if r[0].startswith("Click-through"))[1]
        self.assertAlmostEqual(got, round(ctr, 2), places=2)
        self.assertEqual(rep["rows"], len(df))

    def test_sections_present_and_quality_finds_the_edge_cases(self):
        rep = RP.compute(frame(), plan(), COLS, {"issues": []})
        self.assertEqual([s["id"] for s in rep["sections"]], plan()["sections"])
        q = json.dumps(section(rep, "quality"), default=str).lower()
        self.assertIn("duplicate", q)
        self.assertIn("missing ad_clicks", q)
        recs = section(rep, "recommendations")
        text = json.dumps(recs, default=str).lower()
        for word in ("conversion rate of", "roas of", "revenue of"):
            self.assertNotIn(word, text)                                # nothing invented


def _db():
    try:
        from app import config
        if not config.configured():
            return None
        from app import auth, l1_data as db, l2_policy as P
        db.init_core(); auth.init(); DS.init(); P.ensure_defaults()
        return db
    except Exception:
        return None


DB = _db()


@unittest.skipIf(DB is None, "No database configured (config.env). Integration tests need PostgreSQL.")
class IntegrationReport(unittest.TestCase):
    """The report through the orchestrator: plan gate, routing to the requester, SQL checks and stale plans."""

    @classmethod
    def setUpClass(cls):
        from app import l6_agents as A
        cls.A, cls.db = A, DB
        cls.who = DB.one("SELECT user_id, email, role, full_name FROM app_users WHERE role='admin' AND active "
                         "ORDER BY user_id LIMIT 1")
        cls.ds = DS.create("ads_synthetic.csv", FIXTURE.read_bytes(), cls.who, "Report tests")

    @classmethod
    def tearDownClass(cls):
        DS.delete(cls.ds["dataset_id"], cls.who)

    def run_q(self):
        rid = self.A.create_run(QUESTION, self.who["role"], "haara", "upload",
                                {"dataset_id": self.ds["dataset_id"], "question": QUESTION}, who=self.who)
        self.A.execute(rid)
        return rid

    def wait(self, rid):
        for _ in range(300):
            r = self.db.one("SELECT status, state FROM runs WHERE run_id=?", (rid,))
            if r["status"] != "running":
                return r["status"], json.loads(r["state"])
            time.sleep(0.1)
        self.fail("run did not settle")

    def gate(self, rid):
        cp = self.db.one("SELECT * FROM checkpoints WHERE run_id=? AND status='open'", (rid,))
        if cp:
            cp["evidence"] = json.loads(cp["evidence"])
        return cp

    def test_plan_gate_goes_to_the_person_who_asked(self):
        rid = self.run_q()
        st, s = self.wait(rid)
        self.assertEqual(st, "awaiting_approval")
        cp = self.gate(rid)
        self.assertEqual(cp["kind"], "plan")
        self.assertEqual((cp["assigned_user"] or "").lower(), self.who["email"].lower())   # an admin asking keeps the decision
        self.assertNotIn("data", s.get("out", {}))                     # nothing is computed before approval
        self.assertTrue(cp["evidence"]["plan_hash"])
        self.assertIn("CTR", json.dumps(cp["evidence"]["plan"]))

    def test_approved_plan_runs_and_totals_match_sql(self):
        rid = self.run_q()
        self.wait(rid)
        cp = self.gate(rid)
        self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        res = s["result"]
        self.assertEqual(res["type"], "report")
        checks = s["out"]["checks"]["checks"]
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(c["pass"] for c in checks), checks)
        df = frame()
        self.assertAlmostEqual(res["report"]["totals"]["ad_spend"], round(float(df["ad_spend"].sum()), 2), places=2)
        led = self.db.one("SELECT payload FROM ledger WHERE run_id=? AND event='plan_executed' ORDER BY seq DESC LIMIT 1", (rid,))
        self.assertIsNotNone(led)

    def test_approval_for_a_changed_plan_is_refused(self):
        rid = self.run_q()
        self.wait(rid)
        cp = self.gate(rid)
        ctx = self.A.load(rid)                                          # the plan changes under the open gate
        ctx.params["intent"] = RP.validate({**ctx.params["intent"], "groups": [], "version": 2}, self.ds["columns"])
        self.A._save(ctx, "awaiting_approval")
        with self.assertRaises(PermissionError):
            self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)

    def test_deleting_the_dataset_withdraws_open_decisions(self):
        ds = DS.create("ads_synthetic.csv", FIXTURE.read_bytes(), self.who, "Report tests, deleted")
        rid = self.A.create_run(QUESTION, self.who["role"], "haara", "upload",
                                {"dataset_id": ds["dataset_id"], "question": QUESTION}, who=self.who)
        self.A.execute(rid)
        self.wait(rid)
        self.assertIsNotNone(self.gate(rid))
        DS.delete(ds["dataset_id"], self.who)
        self.assertIsNone(self.gate(rid))
        self.assertEqual(self.db.one("SELECT status FROM runs WHERE run_id=?", (rid,))["status"], "cancelled")

    def test_stale_report_result_is_refused(self):
        rid = self.run_q()
        self.wait(rid)
        ctx = self.A.load(rid)
        ctx.state["out"]["data"] = {"plan_hash": "stale", "report": {}}
        with self.assertRaises(ValueError):
            self.A._current(ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
