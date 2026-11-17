"""Tests for dashboards on uploaded datasets.

Run from the project folder:   python -m unittest tests.test_dashboard -v
Uses invented data only: samples/sample_microfinance_loans.csv and tests/fixtures/ads_synthetic.csv.
Unit tests need no database. Integration tests need the database in config.env and are skipped without it.
"""
import json
import sys
import time
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import dataset_dashboard as DD  # noqa: E402
from app import datasets as DS  # noqa: E402

ADS = ROOT / "tests" / "fixtures" / "ads_synthetic.csv"
LOANS = ROOT / "samples" / "sample_microfinance_loans.csv"
COLS = [{"name": "ad_id", "label": "ad_id", "role": "identifier", "type": "text"},
        {"name": "account_id", "label": "account_id", "role": "identifier", "type": "text"},
        {"name": "ad_spend", "label": "ad_spend", "role": "measure", "type": "number"},
        {"name": "ad_impressions", "label": "ad_impressions", "role": "measure", "type": "number"},
        {"name": "ad_clicks", "label": "ad_clicks", "role": "measure", "type": "number"},
        {"name": "ad_date", "label": "ad_date", "role": "date", "type": "date"}]
DISTINCT = {"ad_id": 80, "account_id": 3, "ad_date": 120}
Q = ("Create a dashboard of ad_spend, ad_impressions and ad_clicks by account_id, weekly, with the top ad_id. "
     "CTR = total clicks divided by total impressions, multiplied by 100.")


def frame():
    return pd.read_csv(ADS, dtype={"ad_id": str, "account_id": str})


def plan(q=Q):
    return DD.validate(DD.build_plan(q, COLS, {}, DISTINCT), COLS)


class UnitDashboard(unittest.TestCase):
    def test_detection(self):
        for q in ("Create a dashboard", "build me a KPI board? no, a dashboard of sales", "Make a scorecard by branch"):
            self.assertTrue(DD.is_dashboard(q), q)
        for q in ("Total ad_spend by account_id", "Comprehensive analysis of the data"):
            self.assertFalse(DD.is_dashboard(q), q)

    def test_plan_reads_the_request(self):
        p = plan()
        self.assertEqual(p["measures"], ["ad_spend", "ad_impressions", "ad_clicks"])
        self.assertEqual((p["dims"], p["item"], p["date"], p["grain"]), (["account_id"], "ad_id", "ad_date", "week"))
        self.assertEqual([(r["name"], r["num"], r["den"], r["mult"]) for r in p["ratios"]],
                         [("CTR", "ad_clicks", "ad_impressions", 100.0)])
        self.assertEqual(p["filters"], {})

    def test_plain_request_uses_defaults(self):
        p = plan("Create a dashboard")
        self.assertEqual(p["measures"], ["ad_spend", "ad_impressions", "ad_clicks"])
        self.assertEqual((p["date"], p["grain"]), ("ad_date", "auto"))

    def test_corrections_add_remove_and_regrain(self):
        p = DD.validate(DD.revise(plan(), "remove ad_impressions and show it monthly", COLS, DISTINCT), COLS)
        self.assertNotIn("ad_impressions", p["measures"])
        self.assertEqual(p["ratios"], [])                         # CTR needs impressions, so it goes too
        self.assertEqual((p["grain"], p["version"]), ("month", 2))
        p2 = DD.validate(DD.revise(plan(), "only ad_spend", COLS, DISTINCT), COLS)
        self.assertEqual(p2["measures"], ["ad_spend"])
        p3 = DD.validate(DD.revise(plan("Create a dashboard of ad_spend"), "CPC = total ad_spend divided by total ad_clicks",
                                   COLS, DISTINCT), COLS)
        self.assertIn("ad_clicks", p3["measures"])
        self.assertEqual([r["name"] for r in p3["ratios"]], ["CPC"])

    def test_hash_changes_with_the_plan(self):
        self.assertEqual(DD.plan_hash(plan()), DD.plan_hash({**plan(), "version": 7}))
        self.assertNotEqual(DD.plan_hash(plan()), DD.plan_hash({**plan(), "grain": "month"}))

    def test_cube_adds_up_and_ratios_are_ratios_of_sums(self):
        df = frame()
        d = DD.compute(df.copy(), plan(), COLS, {"issues": []})
        self.assertTrue(all(d["cube_matches_totals"].values()))
        self.assertEqual(sum(r[-1] for r in d["cube"]["rows"]), len(df))
        self.assertAlmostEqual(d["totals"]["ad_spend"], round(df["ad_spend"].sum(), 6), places=2)
        # recompute CTR for one account from the cube, as the page does, and compare with the raw rows
        cols = d["cube"]["columns"]
        acct = d["dims"][0]["values"].index("9000000000000002")
        sub = [r for r in d["cube"]["rows"] if r[cols.index("account_id")] == acct]
        n, den = sum(r[cols.index("__n_CTR")] or 0 for r in sub), sum(r[cols.index("__d_CTR")] or 0 for r in sub)
        raw = df[(df["account_id"] == "9000000000000002") & df["ad_clicks"].notna() & df["ad_impressions"].notna()]
        self.assertAlmostEqual(n / den * 100, raw["ad_clicks"].sum() / raw["ad_impressions"].sum() * 100, places=9)
        self.assertEqual(d["date"]["grain"], "week")
        self.assertEqual(d["items"]["covered"], 80)

    def test_zero_denominator_stays_out_of_the_ratio(self):
        df = frame()
        df.loc[df["account_id"] == "9000000000000003", "ad_impressions"] = 0
        d = DD.compute(df, plan(), COLS, {"issues": []})
        cols = d["cube"]["columns"]
        acct = d["dims"][0]["values"].index("9000000000000003")
        den = sum(r[cols.index("__d_CTR")] or 0 for r in d["cube"]["rows"] if r[cols.index("account_id")] == acct)
        self.assertEqual(den, 0)                                   # the page shows Undefined, never 0

    def test_auto_grain_follows_the_date_range(self):
        self.assertEqual(DD._grain(30, "auto"), "day")
        self.assertEqual(DD._grain(120, "auto"), "week")
        self.assertEqual(DD._grain(500, "auto"), "month")
        self.assertEqual(DD._grain(500, "quarter"), "quarter")


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
class IntegrationDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import l6_agents as A
        cls.A, cls.db = A, DB
        cls.who = DB.one("SELECT user_id, email, role, full_name FROM app_users WHERE role='analyst' AND active "
                         "ORDER BY user_id LIMIT 1")
        cls.ds = DS.create("sample_microfinance_loans.csv", LOANS.read_bytes(), cls.who, "Dashboard tests")

    @classmethod
    def tearDownClass(cls):
        DS.delete(cls.ds["dataset_id"], cls.who)

    def run_q(self, q):
        rid = self.A.create_run(q, self.who["role"], "haara", "upload",
                                {"dataset_id": self.ds["dataset_id"], "question": q}, who=self.who)
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
        return self.db.one("SELECT * FROM checkpoints WHERE run_id=? AND status='open'", (rid,))

    def test_dashboard_runs_and_checks_totals(self):
        rid = self.run_q("Create a dashboard of Amount and Outstanding by Branch and Product, monthly. "
                         "OSR = total Outstanding divided by total Amount, multiplied by 100. Lower is better.")
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        res = s["result"]
        self.assertEqual(res["type"], "dashboard")
        d = res["dashboard"]
        self.assertEqual([x["name"] for x in d["dims"]], ["branch", "product"])
        self.assertEqual(d["ratios"][0]["better"], "lower")
        self.assertEqual(d["date"]["grain"], "month")
        self.assertTrue(all(c["pass"] for c in s["out"]["checks"]["checks"]), s["out"]["checks"])
        df = pd.read_csv(LOANS)
        self.assertEqual(d["totals"]["amount"], int(df["Amount"].sum()))

    def test_correction_makes_a_new_version_and_asks_to_confirm(self):
        rid = self.run_q("Create a dashboard of Amount by Branch")
        self.wait(rid)
        self.A.correct_interpretation(rid, "add State and remove Branch, show it weekly", self.who["role"], None, who=self.who)
        st, s = self.wait(rid)
        cp = self.gate(rid)
        self.assertEqual((st, cp["kind"]), ("awaiting_approval", "interpretation"))
        self.assertEqual((cp["assigned_user"] or "").lower(), self.who["email"].lower())
        self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        p = s["params"]["intent"]
        self.assertEqual((p["dims"], p["grain"], p["version"]), (["state"], "week", 2))
        self.assertEqual([x["name"] for x in s["result"]["dashboard"]["dims"]], ["state"])

    def test_stale_dashboard_result_is_refused(self):
        rid = self.run_q("Create a dashboard of Amount by Branch")
        self.wait(rid)
        ctx = self.A.load(rid)
        ctx.state["out"]["data"] = {"plan_hash": "stale", "dashboard": {}}
        with self.assertRaises(ValueError):
            self.A._current(ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
