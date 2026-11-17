"""Regression tests for analytical intent, corrections, confirmation and SQL generation on uploaded datasets.

Run from the project folder:   python -m unittest tests.test_intent_state -v
Unit tests need no database. Integration tests need the database in config.env and are skipped without it.
"""
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import datasets as DS  # noqa: E402

QUESTION = ("Analyze the uploaded order dataset. Calculate total sales, total profit, and average profit margin for each "
            "product category. Show the results in a table with one row per category. Explain what the figures reveal "
            "about business performance.")
CORRECTION = ("Calculate three metrics separately: total sales using SUM(sales), total profit using SUM(profit), and average "
              "profit margin using AVG(profit_margin). Group the results by category only. Return one row for each category. "
              "Do not group by product_name.")
REPLACE = ("Discard the previous query plan completely. Start again: total sales, total profit and average profit margin "
           "for Office Supplies, Technology and Furniture, by category only.")
WRONG_PLAN = {"metrics": [{"agg": "avg", "column": "sales"}], "dimensions": ["product_name", "category"],
              "filters": {"product_name": "Table"}}

# product_name comes before category on purpose: the old parser picked the first matching column in column order
COLS = [{"name": "order_id", "label": "order_id", "role": "identifier", "type": "text"},
        {"name": "order_date", "label": "order_date", "role": "date", "type": "date"},
        {"name": "region", "label": "region", "role": "category", "type": "text"},
        {"name": "product_name", "label": "product_name", "role": "category", "type": "text"},
        {"name": "category", "label": "category", "role": "category", "type": "text"},
        {"name": "quantity", "label": "quantity", "role": "measure", "type": "number"},
        {"name": "sales", "label": "sales", "role": "measure", "type": "number"},
        {"name": "profit", "label": "profit", "role": "measure", "type": "number"},
        {"name": "profit_margin", "label": "profit_margin", "role": "measure", "type": "number"},
        {"name": "customer", "label": "Customer Name", "role": "personal", "type": "text", "masked": True}]
VALUES = {"category": ["Office Supplies", "Technology", "Furniture"], "region": ["North", "South", "East", "West"],
          "product_name": ["Table", "Binder", "Phone", "Paper", "Laptop", "Office Chair"]}
TABLE = {"table_name": "uds_test", "columns": COLS}


def plan_for(q):
    return DS.validate(DS.interpret(q, COLS, VALUES, "2026-08-20"), COLS)


def metrics(p):
    return [(m["agg"], m["column"]) for m in p["metrics"]]


class UnitIntent(unittest.TestCase):
    """Parsing, validation, SQL compilation and the SQL check. No database."""

    def test_01_total_sales_is_sum(self):
        self.assertIn(("sum", "sales"), metrics(plan_for(QUESTION)))
        self.assertNotIn(("avg", "sales"), metrics(plan_for(QUESTION)))
        self.assertEqual(metrics(plan_for("What are total sales?")), [("sum", "sales")])
        sql = DS.build_sql(TABLE, plan_for("total sales by region"))[0]
        self.assertIn('SUM("sales")', sql)
        self.assertNotIn('AVG("sales")', sql)

    def test_02_total_profit_is_sum(self):
        self.assertIn(("sum", "profit"), metrics(plan_for(QUESTION)))
        self.assertEqual(metrics(plan_for("sum of profit by category")), [("sum", "profit")])

    def test_03_average_profit_margin_is_avg(self):
        self.assertIn(("avg", "profit_margin"), metrics(plan_for(QUESTION)))
        self.assertNotIn(("avg", "profit"), metrics(plan_for(QUESTION)))       # 'profit margin' is not 'profit'
        self.assertEqual(metrics(plan_for("AVG(profit_margin) by region")), [("avg", "profit_margin")])
        self.assertEqual(metrics(plan_for("profit margin by region")), [("avg", "profit_margin")])  # a ratio is averaged

    def test_03b_three_metrics_in_order(self):
        p = plan_for(QUESTION)
        self.assertEqual(metrics(p), [("sum", "sales"), ("sum", "profit"), ("avg", "profit_margin")])
        self.assertEqual([m["alias"] for m in p["metrics"]], ["total_sales", "total_profit", "average_profit_margin"])
        self.assertEqual(metrics(plan_for(CORRECTION)), [("sum", "sales"), ("sum", "profit"), ("avg", "profit_margin")])
        self.assertEqual(metrics(plan_for("total sales and profit by region")), [("sum", "sales"), ("sum", "profit")])

    def test_04_group_by_category_only(self):
        for q in (QUESTION, CORRECTION, REPLACE, "total sales for each product category"):
            p = plan_for(q)
            self.assertEqual(p["dimensions"], ["category"], q)
            self.assertEqual(p["grain"], ["category"], q)
            self.assertNotIn("product_name", DS.build_sql(TABLE, p)[0])
        self.assertEqual(plan_for(CORRECTION) and DS.interpret(CORRECTION, COLS, VALUES)["excluded_dimensions"], ["product_name"])
        self.assertEqual(plan_for("total sales by product")["dimensions"], ["product_name"])

    def test_04b_expected_sql_structure(self):
        sql, params, p, _, _ = DS.build_sql(TABLE, plan_for(QUESTION))
        self.assertTrue(sql.startswith('SELECT "category"::text AS "category", SUM("sales") AS "total_sales", '
                                       'SUM("profit") AS "total_profit", AVG("profit_margin") AS "average_profit_margin" '
                                       'FROM uds_test GROUP BY "category"::text'), sql)
        self.assertEqual(params, [])
        self.assertEqual(DS.check_sql(sql, p, "uds_test"), [])

    def test_10_no_unintended_filters(self):
        self.assertEqual(plan_for(QUESTION)["filters"], {})                  # 'in a table' is not the product 'Table'
        self.assertEqual(plan_for("Total sales by category in a table")["filters"], {})
        p = plan_for(REPLACE)                                                # listing every category is not a filter
        self.assertEqual(p["filters"], {})
        self.assertTrue(any("every category value" in n for n in p["notes"]))
        self.assertEqual(plan_for("Total sales for Table and Binder")["filters"], {"product_name": ["Binder", "Table"]})
        self.assertEqual(plan_for("Average profit margin by category except Furniture")["filters"], {"category": {"not": "Furniture"}})

    def test_presentation_fields(self):
        p = plan_for(QUESTION)
        self.assertEqual(p["output_format"], "table")
        self.assertTrue(p["explain"])
        self.assertEqual(plan_for("total sales by region")["output_format"], "auto")
        self.assertFalse(plan_for("total sales by region")["explain"])

    def test_sql_check_catches_mismatch(self):
        sql, _, p, _, _ = DS.build_sql(TABLE, plan_for(QUESTION))
        self.assertTrue(DS.check_sql(sql.replace('SUM("sales")', 'AVG("sales")'), p, "uds_test"))
        self.assertTrue(DS.check_sql(sql.replace('GROUP BY "category"::text', 'GROUP BY "category"::text, "product_name"::text'), p, "uds_test"))
        self.assertTrue(DS.check_sql(sql.replace(" GROUP BY", ' WHERE "product_name"::text IN (?) GROUP BY'), p, "uds_test"))
        self.assertTrue(DS.check_sql(sql.replace("uds_test", "other_table"), p, "uds_test"))

    def test_plan_hash_ignores_version_and_notes(self):
        a = plan_for(QUESTION)
        b = dict(a, version=7, notes=["x"], source="corrected_text")
        self.assertEqual(DS.plan_hash(a), DS.plan_hash(b))
        self.assertNotEqual(DS.plan_hash(a), DS.plan_hash(DS.validate(WRONG_PLAN, COLS)))

    def test_old_shape_still_reads(self):
        p = DS.validate({"measure": {"agg": "sum", "column": "sales"}, "dimensions": ["region"], "order": "desc", "limit": 3}, COLS)
        self.assertEqual(metrics(p), [("sum", "sales")])
        self.assertEqual(p["sort"], {"by": "total_sales", "direction": "desc"})

    def test_personal_columns_refused(self):
        with self.assertRaises(ValueError):
            DS.validate({"metrics": [{"agg": "count"}], "dimensions": ["customer"]}, COLS)

    def test_11_conflict_detection(self):
        cons = DS.constraints(CORRECTION, COLS)
        self.assertEqual(DS.conflicts(plan_for(CORRECTION), cons, COLS), [])
        bad = DS.validate(WRONG_PLAN, COLS)
        found = DS.conflicts(bad, cons, COLS)
        self.assertTrue(any("product_name" in f for f in found), found)
        self.assertTrue(any("total sales" in f for f in found), found)


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
class Integration(unittest.TestCase):
    """The orchestrator, corrections, checkpoints and confirmation against a real database."""

    @classmethod
    def setUpClass(cls):
        from app import l6_agents as A
        cls.A, cls.db = A, DB
        cls.who = DB.one("SELECT user_id, email, role, full_name FROM app_users WHERE role='analyst' AND active "
                         "ORDER BY user_id LIMIT 1")
        cls.ds = DS.create("orders.csv", (ROOT / "tests" / "fixtures" / "orders.csv").read_bytes(), cls.who, "Intent tests")
        cls.cols = cls.ds["columns"]
        cls.table = DS._row(cls.ds["dataset_id"])

    @classmethod
    def tearDownClass(cls):
        DS.delete(cls.ds["dataset_id"], cls.who)

    # helpers
    def run_q(self, q, plan=None):
        params = {"dataset_id": self.ds["dataset_id"], "question": q}
        if plan:
            params["intent"] = plan
        rid = self.A.create_run(q, self.who["role"], "haara", "upload", params, who=self.who)
        self.A.execute(rid)
        return rid

    def wait(self, rid):
        for _ in range(150):
            r = self.db.one("SELECT status, state FROM runs WHERE run_id=?", (rid,))
            if r["status"] != "running":
                return r["status"], json.loads(r["state"])
            time.sleep(0.1)
        self.fail("run did not settle")

    def plan(self, rid):
        return self.wait(rid)[1]["params"]["intent"]

    def gate(self, rid):
        cp = self.db.one("SELECT * FROM checkpoints WHERE run_id=? AND status='open'", (rid,))
        if cp:
            cp["evidence"] = json.loads(cp["evidence"])
        return cp

    def approve(self, rid):
        cp = self.gate(rid)
        self.assertIsNotNone(cp, "expected an open confirmation")
        self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)
        return cp

    def correct(self, rid, text, intent=None):
        out = self.A.correct_interpretation(rid, text, self.who["role"], intent, who=self.who)
        self.wait(rid)
        return out

    def finish(self, rid):
        st, s = self.wait(rid)
        if st == "awaiting_approval" and self.gate(rid)["kind"] == "interpretation":
            self.approve(rid)
            st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error") or s.get("result"))
        return s

    # tests
    def test_05_explicit_correction_replaces_previous_intent(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)                 # the reported state: avg sales by product_name and category
        self.assertEqual(metrics(self.plan(rid)), [("avg", "sales")])
        self.correct(rid, CORRECTION)
        p = self.plan(rid)
        self.assertEqual(metrics(p), [("sum", "sales"), ("sum", "profit"), ("avg", "profit_margin")])
        self.assertEqual(p["dimensions"], ["category"])
        self.assertEqual(p["filters"], {})                     # the stale 'product_name is Table' filter is gone
        self.assertEqual(p["version"], 2)
        rid2 = self.run_q(QUESTION, WRONG_PLAN)
        self.correct(rid2, REPLACE)
        p2 = self.plan(rid2)
        self.assertEqual(metrics(p2), metrics(p))
        self.assertEqual((p2["dimensions"], p2["filters"]), (["category"], {}))
        led = self.db.one("SELECT payload FROM ledger WHERE run_id=? AND event='interpretation_corrected' ORDER BY seq DESC LIMIT 1", (rid2,))
        self.assertEqual(json.loads(led["payload"])["mode"], "replace")

    def test_06_repeated_corrections_do_not_restore_stale_intent(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)
        self.correct(rid, CORRECTION)
        self.finish(rid)
        self.correct(rid, "Only for 2026")
        self.finish(rid)
        self.correct(rid, "Leave out Furniture too")
        s = self.finish(rid)
        p = s["params"]["intent"]
        self.assertEqual(metrics(p), [("sum", "sales"), ("sum", "profit"), ("avg", "profit_margin")])
        self.assertEqual(p["dimensions"], ["category"])
        self.assertNotIn("product_name", json.dumps(p))
        self.assertEqual(p["filters"], {"category": {"not": "Furniture"}})
        self.assertEqual(p["period"]["label"], "2026")
        self.assertEqual([h["version"] for h in s["params"]["plan_history"]], [1, 2, 3, 4])
        self.assertNotIn('"product_name"', s["result"]["result"]["sql"])

    def test_07_confirmation_applies_to_the_exact_plan(self):
        rid = self.run_q(QUESTION)
        s = self.finish(rid)
        v1 = s["params"]["intent"]["version"]
        self.correct(rid, "Use region instead of category")
        s = self.finish(rid)
        v2 = s["params"]["intent"]["version"]
        self.assertEqual(v2, v1 + 1)
        with self.assertRaises(ValueError):                    # a click on the page that still showed version 1
            self.A.confirm_interpretation(rid, self.who["role"], who=self.who, plan_version=v1)
        with self.assertRaises(ValueError):                    # a confirmation must say which plan it confirms
            self.A.confirm_interpretation(rid, self.who["role"], who=self.who)
        out = self.A.confirm_interpretation(rid, self.who["role"], who=self.who, plan_version=v2)
        self.assertEqual(out["plan_version"], v2)
        led = json.loads(self.db.one("SELECT payload FROM ledger WHERE run_id=? AND event='plan_confirmed' ORDER BY seq DESC LIMIT 1",
                                     (rid,))["payload"])
        self.assertEqual((led["plan_version"], led["plan_hash"]), (v2, DS.plan_hash(s["params"]["intent"])))

    def test_08_changed_plan_invalidates_stale_sql_and_approvals(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)
        self.correct(rid, CORRECTION)
        old_cp = self.gate(rid)
        self.assertIsNotNone(old_cp, "a reworded correction must be confirmed")
        self.assertEqual(old_cp["evidence"]["plan_version"], 2)
        with self.assertRaises(ValueError):                    # three measures, so the agent asks which one to change
            self.correct(rid, "Use the median instead")
        self.correct(rid, "By region only")                    # plan changes again before the person decides
        new_cp = self.gate(rid)
        self.assertNotEqual(old_cp["cp_id"], new_cp["cp_id"])
        status = self.db.one("SELECT status FROM checkpoints WHERE cp_id=?", (old_cp["cp_id"],))["status"]
        self.assertEqual(status, "superseded")
        with self.assertRaises(ValueError):                    # the old approval cannot be used
            self.A.decide(old_cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)
        _, s = self.wait(rid)
        self.assertNotIn("data", s["out"])                     # no SQL or result from an older plan survives
        self.assertIsNone(s.get("result"))
        # a result computed for another plan is refused downstream
        ctx = self.A.load(rid)
        ctx.state["out"]["data"] = {"plan_hash": "stale", "rows": []}
        with self.assertRaises(ValueError):
            self.A._current(ctx)

    def test_08b_old_gate_approval_after_plan_change_is_refused(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)
        self.correct(rid, CORRECTION)
        cp = self.gate(rid)
        ctx = self.A.load(rid)                                  # the plan changes underneath the open gate
        ctx.params["intent"] = DS.validate({**WRONG_PLAN, "version": 3}, self.cols)
        self.A._save(ctx, "awaiting_approval")
        with self.assertRaises(PermissionError):
            self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)

    def test_09_sql_uses_latest_approved_plan(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)
        self.correct(rid, CORRECTION)
        self.approve(rid)
        s = self.finish(rid)
        p = s["params"]["intent"]
        sql = DS.build_sql(self.table, p)[0]
        self.assertEqual(s["result"]["result"]["sql"], sql)
        self.assertEqual(s["result"]["plan_hash"], DS.plan_hash(p))
        self.assertEqual(len(s["result"]["result"]["rows"]), 3)  # one row for each category
        self.assertEqual(set(s["result"]["result"]["rows"][0]), {"category", "total_sales", "total_profit", "average_profit_margin"})
        ex = json.loads(self.db.one("SELECT payload FROM ledger WHERE run_id=? AND event='plan_executed' ORDER BY seq DESC LIMIT 1",
                                    (rid,))["payload"])
        self.assertEqual((ex["plan_version"], ex["sql"]), (p["version"], sql))
        decided = json.loads(self.db.one("SELECT payload FROM ledger WHERE run_id=? AND event='human_decision' ORDER BY seq DESC LIMIT 1",
                                         (rid,))["payload"])
        self.assertEqual(decided["kind"], "interpretation")
        st = self.wait(rid)[1]
        approved = [d for k, d in st["decisions"].items() if k.startswith("interpretation:")]
        self.assertEqual(approved[-1]["plan_hash"], DS.plan_hash(p))

    def test_11_contradiction_asks_for_clarification(self):
        rid = self.run_q(QUESTION, WRONG_PLAN)
        form = {"metrics": [{"agg": "sum", "column": "sales"}], "dimensions": ["category", "product_name"], "filters": {}}
        with self.assertRaises(ValueError) as e:              # words say no product_name, the form adds it
            self.A.correct_interpretation(rid, "Do not group by product_name", self.who["role"], form, who=self.who)
        self.assertIn("disagree", str(e.exception))
        self.assertEqual(metrics(self.plan(rid)), [("avg", "sales")])   # nothing was applied
        # a plan that contradicts the last correction can be shown, but it cannot be confirmed
        ctx = self.A.load(rid)
        self.A.steer(rid, "n2", {"intent": DS.validate({**WRONG_PLAN, "version": 2}, self.cols),
                                 "constraints": DS.constraints(CORRECTION, self.cols)}, self.who["role"], background=False, who=self.who)
        cp = self.gate(rid)
        self.assertTrue(cp["evidence"]["conflicts"])
        with self.assertRaises(PermissionError):
            self.A.decide(cp["cp_id"], "approve", self.who["role"], who=self.who, background=False)
        self.assertEqual(self.wait(rid)[0], "awaiting_approval")

    def test_12a_existing_dataset_questions_still_work(self):
        s = self.finish(self.run_q("Total sales by region"))
        self.assertEqual(len(s["result"]["result"]["rows"]), 4)
        s = self.finish(self.run_q("Top 2 product by total profit"))
        self.assertEqual(len(s["result"]["result"]["rows"]), 2)
        s = self.finish(self.run_q("Average profit margin by category except Furniture"))
        self.assertEqual(len(s["result"]["result"]["rows"]), 2)

    def test_12b_bank_question_still_works(self):
        rid = self.A.create_run("What is the delinquency rate by branch in Lagos?", "analyst", "haara", "bi", {}, who=self.who)
        self.A.execute(rid)
        st, s = self.wait(rid)
        self.assertEqual(st, "completed", s.get("error"))
        self.assertEqual(s["result"]["type"], "bi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
