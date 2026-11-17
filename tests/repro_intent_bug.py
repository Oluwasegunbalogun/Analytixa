"""Reproduce the intent and correction bug on a given Analytixa code tree.
Usage (from the project folder): python tests/repro_intent_bug.py .
"""
import json
import sys
import time

sys.path.insert(0, sys.argv[1])
from app import auth, datasets as DS, l1_data as db, l2_policy as P, l6_agents as A  # noqa: E402

Q = ("Analyze the uploaded order dataset. Calculate total sales, total profit, and average profit margin for each "
     "product category. Show the results in a table with one row per category. Explain what the figures reveal "
     "about business performance.")
C1 = ("Calculate three metrics separately: total sales using SUM(sales), total profit using SUM(profit), and average "
      "profit margin using AVG(profit_margin). Group the results by category only. Return one row for each category. "
      "Do not group by product_name.")
C2 = ("Discard the previous query plan completely. Start again: total sales, total profit and average profit margin "
      "for Office Supplies, Technology and Furniture, by category only.")

db.init_core(); auth.init(); DS.init(); P.ensure_defaults()
who = db.one("SELECT user_id, email, role, full_name FROM app_users WHERE role='analyst' AND active ORDER BY user_id LIMIT 1")
ds = DS.create("orders.csv", open(str(__import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "orders.csv"), "rb").read(), who, "Orders (repro)")


def wait(rid):
    for _ in range(100):
        r = db.one("SELECT status, state FROM runs WHERE run_id=?", (rid,))
        if r["status"] != "running":
            return r["status"], json.loads(r["state"])
        time.sleep(0.2)


def show(label, rid):
    st, s = wait(rid)
    it = s["params"].get("intent", {})
    res = s.get("result") or {}
    interp = res.get("interpretation") or (s["out"].get("interpretation") or {}).get("text")
    print(f"--- {label}\n status: {st}\n interpretation: {interp}\n intent: {json.dumps({k: v for k, v in it.items() if k != 'source'})}")
    if res.get("result"):
        print(" sql:", res["result"]["sql"])
    cp = db.one("SELECT cp_id, kind, evidence FROM checkpoints WHERE run_id=? AND status='open'", (rid,))
    if cp:
        print(" open gate:", cp["kind"], json.loads(cp["evidence"]).get("interpretation"))
    return cp


try:
    rid = A.create_run(Q, who["role"], "haara", "upload", {"dataset_id": ds["dataset_id"], "question": Q}, who=who)
    A.execute(rid)
    cp = show("1. original question", rid)
    if cp:
        A.decide(cp["cp_id"], "approve", who["role"], who=who, background=False)
        show("1b. after confirming", rid)
    try:
        A.correct_interpretation(rid, C1, who["role"], who=who)
        cp = show("2. first correction", rid)
        if cp:
            A.decide(cp["cp_id"], "approve", who["role"], who=who, background=False)
            show("2b. after confirming", rid)
    except Exception as e:
        print("--- 2. first correction refused:", e)
    try:
        A.correct_interpretation(rid, C2, who["role"], who=who)
        cp = show("3. replace instruction", rid)
        if cp:
            A.decide(cp["cp_id"], "approve", who["role"], who=who, background=False)
            show("3b. after confirming", rid)
    except Exception as e:
        print("--- 3. replace instruction refused:", e)
    led = db.query("SELECT event, payload FROM ledger WHERE run_id=? AND event IN ('interpretation_corrected',"
                   "'interpretation_confirmed','human_decision','plan_created','plan_confirmed','correction_unchanged','plan_executed','decision_denied') ORDER BY seq", (rid,))
    print("--- ledger:", [l["event"] for l in led])
finally:
    DS.delete(ds["dataset_id"], who)
