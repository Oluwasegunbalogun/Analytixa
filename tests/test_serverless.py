"""Tests for Vercel mode: tasks advance on status checks under a lease, and the public showcase stays read-only.

Run from the project folder:   python -m unittest tests.test_serverless -v
Needs the database in config.env. Skipped without it.
"""
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _db():
    try:
        from app import config
        if not config.configured():
            return None
        from app import auth, datasets as DS, l1_data as db, l2_policy as P
        db.init_core(); auth.init(); DS.init(); P.ensure_defaults()
        return db if db.table_exists("customers") else None
    except Exception:
        return None


DB = _db()


@unittest.skipIf(DB is None, "No database with the bank seeded (config.env).")
class Serverless(unittest.TestCase):
    def setUp(self):
        os.environ["HAARA_SERVERLESS"] = "1"
        from app import l6_agents as A
        self.A = A
        self.who = DB.one("SELECT user_id, email, role, full_name FROM app_users WHERE role='analyst' AND active LIMIT 1")

    def tearDown(self):
        os.environ.pop("HAARA_SERVERLESS", None)

    def test_no_thread_and_progress_by_polling(self):
        rid = self.A.create_run("What is the delinquency rate by branch in Lagos?", "analyst", "haara", "bi", {}, who=self.who)
        self.assertIsNone(self.A.start(rid))                       # nothing runs in the background
        self.assertEqual(DB.one("SELECT status FROM runs WHERE run_id=?", (rid,))["status"], "running")
        for _ in range(20):
            self.A.advance(rid, 0.0)                               # one step per call when the budget is spent
            if DB.one("SELECT status FROM runs WHERE run_id=?", (rid,))["status"] != "running":
                break
        st = DB.one("SELECT status, lease_until FROM runs WHERE run_id=?", (rid,))
        self.assertIn(st["status"], ("completed", "awaiting_approval"))
        self.assertIsNone(st["lease_until"])                       # the lease is released

    def test_lease_stops_a_second_worker(self):
        rid = self.A.create_run("Show deposit inflow by month", "analyst", "haara", "bi", {}, who=self.who)
        DB.execute("UPDATE runs SET lease_until = now() + interval '60 seconds' WHERE run_id=?", (rid,))
        self.assertFalse(self.A.advance(rid, 1.0))                 # another request holds it
        DB.execute("UPDATE runs SET lease_until = now() - interval '1 second' WHERE run_id=?", (rid,))
        self.assertTrue(self.A.advance(rid, 30.0))                 # an expired lease is taken over
        self.assertNotEqual(DB.one("SELECT status FROM runs WHERE run_id=?", (rid,))["status"], "running")


@unittest.skipIf(DB is None, "No database with the bank seeded (config.env).")
class Showcase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.update(HAARA_SHOWCASE="1", CRON_SECRET="test-secret")
        from fastapi.testclient import TestClient
        import server
        server.boot.done = True                                     # the database is ready; skip start-up work
        cls.c = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        for k in ("HAARA_SHOWCASE", "CRON_SECRET"):
            os.environ.pop(k, None)

    def test_demo_buttons_and_sign_in(self):
        p = self.c.get("/api/public").json()
        self.assertTrue(p["showcase"])
        self.assertIn("analyst", [d["role"] for d in p["demo"]])
        self.assertNotIn("email", json.dumps(p["demo"]))           # no addresses or passwords on the public page
        r = self.c.post("/api/demo-login", json={"role": "admin"})
        self.assertEqual(r.json()["role"], "admin")

    def test_real_accounts_are_never_demo_accounts(self):
        from app import auth, l1_data as db
        a = db.one("SELECT user_id, email FROM app_users WHERE role='admin' AND active ORDER BY user_id LIMIT 1")
        db.execute("UPDATE app_users SET email='owner@example.com' WHERE user_id=?", (a["user_id"],))
        try:
            self.assertNotIn("admin", [d["role"] for d in auth.demo_accounts()])
            with self.assertRaises(PermissionError):
                auth.demo_login("admin")
        finally:
            db.execute("UPDATE app_users SET email=? WHERE user_id=?", (a["email"], a["user_id"]))

    def test_settings_policies_and_staff_are_read_only(self):
        self.c.post("/api/demo-login", json={"role": "admin"})
        self.assertEqual(self.c.put("/api/policies/uploads", json={"body": {}, "note": "x"}).status_code, 403)
        self.assertEqual(self.c.post("/api/settings", json={"api_key": "x"}).status_code, 403)
        self.assertEqual(self.c.post("/api/users", json={"full_name": "X", "email": "x@x.test", "role": "analyst",
                                                         "password": "Abcdefgh1!"}).status_code, 403)

    def test_daily_maintenance_runs_once_a_day(self):
        import server
        out = server.daily_maintenance("system:test")
        self.assertIn("loans", out["bank_datasets"])
        self.assertFalse(server.maintenance_due())                  # just ran, so the hourly check skips it
        self.assertTrue(server.maintenance_due(hours=0))

    def test_cron_needs_the_secret(self):
        self.assertEqual(self.c.get("/api/cron/daily").status_code, 401)
        self.assertEqual(self.c.get("/api/cron/daily", headers={"authorization": "Bearer wrong"}).status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
