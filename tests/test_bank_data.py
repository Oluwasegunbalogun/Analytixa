"""Tests for the bank's own data as datasets (Your data, Bank data).

Run from the project folder:   python -m unittest tests.test_bank_data -v
Needs the database in config.env. Skipped without it.
"""
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import datasets as DS  # noqa: E402


def _db():
    try:
        from app import config
        if not config.configured():
            return None
        from app import auth, l1_data as db, l2_policy as P
        db.init_core(); auth.init(); DS.init(); P.ensure_defaults()
        return db if db.table_exists("loans") else None
    except Exception:
        return None


DB = _db()


@unittest.skipIf(DB is None, "No database with the bank seeded (config.env).")
class BankData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import bank_data as BK, l6_agents as A
        cls.BK, cls.A, cls.db = BK, A, DB
        cls.synced = dict(BK.sync())
        user = lambda role: DB.one("SELECT user_id, email, role, full_name FROM app_users WHERE role=? AND active LIMIT 1", (role,))
        cls.analyst, cls.ops, cls.credit = user("analyst"), user("branch_ops"), user("credit_manager")

    def wait(self, rid):
        for _ in range(600):
            r = self.db.one("SELECT status, state FROM runs WHERE run_id=?", (rid,))
            if r["status"] != "running":
                return r["status"], json.loads(r["state"])
            time.sleep(0.1)
        self.fail("run did not settle")

    def run_q(self, did, q, who):
        rid = self.A.create_run(q, who["role"], "haara", "upload", {"dataset_id": did, "question": q}, who=who)
        self.A.execute(rid)
        return rid

    def test_views_match_the_core_tables(self):
        self.assertEqual(self.synced["loans"], self.db.one("SELECT COUNT(*) n FROM loans")["n"])
        self.assertEqual(self.synced["transactions"], self.db.one("SELECT COUNT(*) n FROM transactions_clean")["n"])

    def test_no_personal_columns(self):
        for key in self.BK.VIEWS:
            names = {c["name"] for c in DS._row(f"BANK-{key}")["columns"]}
            for bad in ("name", "phone", "bvn_verified", "counterparty", "monthly_income"):
                self.assertNotIn(bad, names, key)

    def test_access_follows_role(self):
        self.assertIn("ATM cash", [d["name"] for d in self.BK.listing(self.ops)])
        self.assertNotIn("Loan portfolio", [d["name"] for d in self.BK.listing(self.ops)])
        with self.assertRaises(PermissionError):
            DS.get("BANK-loans", self.ops)
        self.assertEqual(DS.get("BANK-loans", self.credit, preview=False)["dataset_id"], "BANK-loans")

    def test_ask_routes_analysis_and_dashboards_to_bank_data(self):
        flat = ("Analyse the bank's loan portfolio to evaluate exposure, credit risk and data quality. 1. Overall performance: "
                "total Principal and total Outstanding. 2. Efficiency ratios: Portfolio at risk (PAR) = total Outstanding at risk "
                "divided by total Outstanding, multiplied by 100. 3. Compare performance across Branch and Product. "
                "4. Data quality: missing values. 5. Concentration. 6. Recommendations. 7. Executive summary.")
        self.assertEqual(self.BK.pick(flat, "analyst"), "BANK-loans")
        self.assertIsNone(self.BK.pick(flat, "branch_ops"))                     # not allowed, so not picked
        self.assertEqual(self.BK.pick("Create a dashboard of withdrawals by ATM", "branch_ops"), "BANK-atm_cash")
        self.assertEqual(self.BK.pick("Create a dashboard of inflow and outflow by channel", "analyst"), "BANK-transactions")
        self.assertEqual(self.BK.pick("Create a dashboard of loan applications by product", "analyst"), "BANK-applications")
        self.assertIsNone(self.BK.pick("What is the delinquency rate by branch in Lagos?", "analyst"))   # a normal question
        self.assertIsNone(self.BK.pick("Create a dashboard", "analyst"))
        self.assertTrue(self.BK.wants_dataset("Create a dashboard"))

    def test_cannot_be_deleted(self):
        with self.assertRaises(PermissionError):
            DS.delete("BANK-loans", self.analyst)

    def test_role_without_access_is_stopped_by_the_guard(self):
        rid = self.run_q("BANK-loans", "Total Outstanding by Branch", self.ops)
        st, s = self.wait(rid)
        self.assertNotEqual(st, "completed")
        self.assertIsNone((s.get("result") or {}).get("result"))

    def test_question_dashboard_and_analysis_match_sql(self):
        out = self.db.one("SELECT SUM(outstanding)::float8 s FROM loans")["s"]
        rid = self.run_q("BANK-loans", "Total Outstanding by Branch", self.analyst)
        st, s = self.wait(rid)
        if st == "awaiting_approval":
            cp = self.db.one("SELECT cp_id FROM checkpoints WHERE run_id=? AND status='open'", (rid,))
            self.A.decide(cp["cp_id"], "approve", self.analyst["role"], who=self.analyst, background=False)
            st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        self.assertAlmostEqual(sum(r["total_outstanding"] for r in s["result"]["result"]["rows"]), out, places=2)
        rid = self.run_q("BANK-loans", "Create a dashboard of Outstanding by Branch and Product. PAR = total Outstanding at "
                                       "risk divided by total Outstanding, multiplied by 100. Lower is better.", self.analyst)
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        self.assertAlmostEqual(s["result"]["dashboard"]["totals"]["outstanding"], out, places=2)
        self.assertTrue(all(c["pass"] for c in s["out"]["checks"]["checks"]))
        q = ("Comprehensive loan portfolio analysis. 1. Overall totals of Principal and Outstanding. 2. Efficiency ratios: "
             "Portfolio at risk (PAR) = total Outstanding at risk divided by total Outstanding, multiplied by 100. Lower is better. "
             "3. Compare across Branch and Product. 4. Data quality: missing values. 5. Concentration. 6. Recommendations. 7. Summary.")
        rid = self.run_q("BANK-loans", q, self.analyst)
        st, _ = self.wait(rid)
        cp = self.db.one("SELECT cp_id, kind, assigned_user FROM checkpoints WHERE run_id=? AND status='open'", (rid,))
        self.assertEqual((st, cp["kind"]), ("awaiting_approval", "plan"))
        self.assertEqual(cp["assigned_user"].lower(), self.analyst["email"].lower())
        self.A.decide(cp["cp_id"], "approve", self.analyst["role"], who=self.analyst, background=False)
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        self.assertAlmostEqual(s["result"]["report"]["totals"]["outstanding"], out, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
