"""Live check of the language model chain against the real OpenRouter (and Anthropic) services.

Run from the project folder on your own computer:   python tests/live_llm_check.py
It asks for the key without showing it, keeps it in memory only, and prints:
  1. whether each model in the chain answers, and how fast
  2. how the language model reads the reported question about an orders file, compared with the expected plan
Nothing is written to disk or to the database.
"""
import getpass
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app import datasets as DS  # noqa: E402

QUESTION = ("Analyze the uploaded order dataset. Calculate total sales, total profit, and average profit margin for each "
            "product category. Show the results in a table with one row per category. Explain what the figures reveal "
            "about business performance.")
EXPECTED = [("sum", "sales"), ("sum", "profit"), ("avg", "profit_margin")]
COLS = [{"name": "order_id", "label": "order_id", "role": "identifier", "type": "text"},
        {"name": "order_date", "label": "order_date", "role": "date", "type": "date"},
        {"name": "region", "label": "region", "role": "category", "type": "text"},
        {"name": "product_name", "label": "product_name", "role": "category", "type": "text"},
        {"name": "category", "label": "category", "role": "category", "type": "text"},
        {"name": "sales", "label": "sales", "role": "measure", "type": "number"},
        {"name": "profit", "label": "profit", "role": "measure", "type": "number"},
        {"name": "profit_margin", "label": "profit_margin", "role": "measure", "type": "number"}]
VALUES = {"category": ["Office Supplies", "Technology", "Furniture"], "region": ["North", "South", "East", "West"],
          "product_name": ["Table", "Binder", "Phone", "Paper", "Laptop"]}


def main():
    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key (input hidden): ").strip()
    akey = os.environ["ANTHROPIC_API_KEY"] if "ANTHROPIC_API_KEY" in os.environ else \
        getpass.getpass("Anthropic API key, or press Enter to skip: ").strip()
    if not key and not akey:
        print("No key given. Nothing to test.")
        return
    llm.configure(api_key=key, anthropic_key=akey, enabled=True)

    print("\n1. Each model in the chain, on its own")
    ok = 0
    for prov, model in llm.chain():
        t0 = time.time()
        try:
            fn = llm._openrouter if prov == "openrouter" else llm._anthropic
            text = fn(model, "Reply with one JSON object only.", 'Return {"ok": true}', 50)
            good = llm._parse_json(text) is not None
            ok += good
            print(f"   {'OK  ' if good else 'BAD '} {prov}:{model}  {time.time() - t0:.1f}s  {text[:60]!r}")
        except Exception as e:
            print(f"   FAIL {prov}:{model}  {time.time() - t0:.1f}s  {str(e)[:90]}")
    print(f"   {ok} of {len(llm.chain())} models answered with readable output.")

    print("\n2. The chain as Analytixa uses it, on the reported question")
    t0 = time.time()
    plan = DS.interpret_llm(QUESTION, COLS, VALUES)
    used = llm.last_call()
    print(f"   Answered by: {used.get('provider')}:{used.get('model')} after {used.get('fallbacks_used')} switch(es), "
          f"{time.time() - t0:.1f}s")
    for t in used.get("tried", []):
        print(f"   skipped {t['model']}: {t['reason']}")
    if not plan:
        print("   No usable plan from any model. Analytixa would use the rules reader, which gives:")
        plan = DS.validate(DS.interpret(QUESTION, COLS, VALUES), COLS)
    print("   Plan:", DS.describe(plan, COLS))
    got = [(m["agg"], m["column"]) for m in plan["metrics"]]
    checks = [("metrics are SUM(sales), SUM(profit), AVG(profit_margin)", got == EXPECTED),
              ("grouped by category only", plan["dimensions"] == ["category"]),
              ("no filters", plan["filters"] == {}),
              ("asks for an explanation", plan.get("explain") is True)]
    for c, p in checks:
        print(f"   {'PASS' if p else 'FAIL'} {c}")
    print("\n   Plan JSON:", json.dumps({k: plan[k] for k in ("metrics", "dimensions", "filters", "output_format", "explain")}))
    print("\nDone. The key was not saved. Revoke it on openrouter.ai if you made it only for this test.")


if __name__ == "__main__":
    main()
