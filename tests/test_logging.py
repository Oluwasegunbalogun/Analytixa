"""Tests for the diagnostic log: secrets are removed, errors carry a reference, and only admins can read the log.

Run from the project folder:   python -m unittest tests.test_logging -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import logs  # noqa: E402


class Redaction(unittest.TestCase):
    def test_secrets_removed(self):
        text = ("postgresql://postgres.abcd:FakePass0000.@aws-0-eu-west-2.pooler.supabase.com:5432/postgres "
                "key sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef "
                "sk-ant-api03-AbCdEf123456789 Authorization: Bearer abcdef123456789 "
                '{"password": "hunter22", "api_key": "xyz123"} haara_session=tok123456')
        out = logs.redact(text)
        for secret in ("FakePass0000", "0123456789abcdef", "AbCdEf123456789", "abcdef123456789", "hunter22", "xyz123", "tok123456"):
            self.assertNotIn(secret, out)
        self.assertIn("postgres.abcd:***@aws-0-eu-west-2.pooler.supabase.com", out)   # the host stays, for diagnosis


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            import server
            from app import config
            if not config.configured():
                raise unittest.SkipTest("No database configured")
            server.boot()
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"Server could not start here: {e}")

        def broken():
            raise RuntimeError("boom with password=hunter22 in it")
        server.app.add_api_route("/api/_test_broken", broken, methods=["GET"])
        cls.client = TestClient(server.app, raise_server_exceptions=False)
        cls.server = server

    def test_unexpected_error_gives_reference_and_is_logged(self):
        r = self.client.get("/api/_test_broken")
        self.assertEqual(r.status_code, 500)
        ref = r.json()["reference"]
        self.assertTrue(ref.startswith("E-"))
        self.assertNotIn("boom", r.json()["detail"])                  # details stay on the server
        log = logs.tail(200)
        self.assertIn(ref, log)
        self.assertIn("RuntimeError: boom", log)                      # the traceback is in the log
        self.assertNotIn("hunter22", log)

    def test_diagnostics_needs_admin(self):
        self.assertEqual(self.client.get("/api/diagnostics").status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
