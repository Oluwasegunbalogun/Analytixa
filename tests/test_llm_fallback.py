"""Tests for the language model fallback chain. No network: the providers are replaced by fakes.

Run from the project folder:   python -m unittest tests.test_llm_fallback -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app import datasets as DS  # noqa: E402


class FakeProviders:
    """Each model answers from a script: an exception, empty text or a string."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, prov):
        def go(model, system, user, max_tokens):
            self.calls.append(f"{prov}:{model}")
            out = self.script.get(f"{prov}:{model}", RuntimeError("HTTP 429 rate limited"))
            if isinstance(out, Exception):
                raise out
            return out
        return go


class Fallback(unittest.TestCase):
    def setUp(self):
        self.saved = (dict(llm.SETTINGS), llm._openrouter, llm._anthropic)
        llm.configure(api_key="test-key", model="first/model", enabled=True, fallbacks=["second/model", "third/model"],
                      anthropic_key="", anthropic_model="claude-test")

    def tearDown(self):
        llm.SETTINGS.clear(); llm.SETTINGS.update(self.saved[0])
        llm._openrouter, llm._anthropic = self.saved[1], self.saved[2]

    def use(self, script):
        fake = FakeProviders(script)
        llm._openrouter, llm._anthropic = fake("openrouter"), fake("anthropic")
        return fake

    def test_chain_order(self):
        self.assertEqual(llm.chain(), [("openrouter", "first/model"), ("openrouter", "second/model"), ("openrouter", "third/model")])
        llm.configure(anthropic_key="a-key")
        self.assertEqual(llm.chain()[-1], ("anthropic", "claude-test"))
        llm.configure(api_key="")
        self.assertEqual(llm.chain(), [("anthropic", "claude-test")])     # Anthropic alone still works

    def test_moves_on_when_a_model_fails(self):
        fake = self.use({"openrouter:second/model": "Hello"})
        self.assertEqual(llm.chat("s", "u"), "Hello")
        self.assertEqual(fake.calls, ["openrouter:first/model", "openrouter:second/model"])
        self.assertEqual((llm.last_call()["model"], llm.last_call()["fallbacks_used"]), ("second/model", 1))

    def test_moves_on_when_output_is_not_json(self):
        fake = self.use({"openrouter:first/model": "Sure! Here is the plan.", "openrouter:second/model": "",
                         "openrouter:third/model": '{"metrics": []}'})
        self.assertEqual(llm.chat_json("s", "u"), {"metrics": []})
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(llm.last_call()["model"], "third/model")

    def test_anthropic_is_the_last_resort(self):
        llm.configure(anthropic_key="a-key")
        fake = self.use({"anthropic:claude-test": "From Anthropic"})
        self.assertEqual(llm.chat("s", "u"), "From Anthropic")
        self.assertEqual(fake.calls[-1], "anthropic:claude-test")
        self.assertEqual(llm.last_call()["provider"], "anthropic")

    def test_every_model_fails_returns_none(self):
        self.use({})
        self.assertIsNone(llm.chat_json("s", "u"))
        self.assertIsNone(llm.last_call()["model"])
        self.assertEqual(len(llm.last_call()["tried"]), 3)

    def test_no_key_means_rules(self):
        llm.configure(api_key="", anthropic_key="")
        self.assertFalse(llm.available())
        self.assertIsNone(llm.chat("s", "u"))

    def test_public_settings_hide_keys(self):
        llm.configure(anthropic_key="secret-a")
        pub = json.dumps(llm.public())
        self.assertNotIn("test-key", pub)
        self.assertNotIn("secret-a", pub)

    def test_dataset_question_read_by_backup_model(self):
        cols = [{"name": "category", "label": "category", "role": "category", "type": "text"},
                {"name": "sales", "label": "sales", "role": "measure", "type": "number"},
                {"name": "profit", "label": "profit", "role": "measure", "type": "number"}]
        plan = {"metrics": [{"agg": "sum", "column": "sales"}, {"agg": "sum", "column": "profit"}],
                "dimensions": ["category"], "filters": {}}
        self.use({"openrouter:second/model": json.dumps(plan)})
        out = DS.interpret_llm("total sales and profit by category", cols, {})
        self.assertEqual([(m["agg"], m["column"]) for m in out["metrics"]], [("sum", "sales"), ("sum", "profit")])
        self.assertEqual(out["source"], "llm")
        self.assertTrue(any("second/model" in n and "1 unavailable" in n for n in out["notes"]), out["notes"])

    def test_invalid_model_plan_is_rejected(self):
        cols = [{"name": "sales", "label": "sales", "role": "measure", "type": "number"}]
        self.use({"openrouter:first/model": json.dumps({"metrics": [{"agg": "sum", "column": "no_such_column"}]})})
        self.assertIsNone(DS.interpret_llm("total sales", cols, {}))     # the agent then uses the rules reader


if __name__ == "__main__":
    unittest.main(verbosity=2)
