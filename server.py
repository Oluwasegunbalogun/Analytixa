"""Analytixa HAARA server: FastAPI entry point (L8 application backend).

Run:  python setup.py   (once, to connect the database and create staff accounts)
      python server.py  then open http://127.0.0.1:4000
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app import datasets as DS
from app import auth
from app import l1_data as db
from app import l2_policy as P
from app import l3_models as M
from app import l5_tools as T
from app import l6_agents as A
from app import l7_control as C
from app import llm
from app import logs
from app import seed

ROOT = Path(__file__).parent
COOKIE = "haara_session"
app = FastAPI(title="Analytixa HAARA", version="4.0")
log = logs.setup()


def boot(verbose=False):
    def say(m):
        log.info(m)
        if verbose:
            print(m, flush=True)
    t0 = time.time()
    log.info("starting Analytixa %s, Python %s on %s", app.version, platform.python_version(), platform.system())
    db.init_core()
    auth.init()
    DS.init()
    light = config.serverless()
    if not db.table_exists("customers"):
        if light:                                         # building the bank takes minutes: do it once, from setup.py
            raise RuntimeError("This database has no bank yet. Run python setup.py on your computer against it first.")
        say("First run: building the bank in your database. This takes one to three minutes...")
    s = seed.seed(progress=say)
    P.ensure_defaults()
    if s.get("seeded") or not M.trained():
        say("Training the credit, churn and anomaly models...")
        M.train_all()
    from app import bank_data as BK
    try:                                                  # the bank's own data as datasets for analysis and dashboards
        done = BK.sync(max_age_hours=24 if (light or config.azure()) else 0)   # a restart reuses fresh profiles
        log.info("bank datasets ready: %s", ", ".join(f"{k} ({n:,} rows)" if n is not None else f"{k} (current)" for k, n in done))
    except Exception:
        log.exception("bank datasets could not be prepared. Uploads still work")
    if not light:
        # runs left mid-flight by a restart resume from their last saved node. Serverless: the next status check does it
        for r in db.query("SELECT run_id FROM runs WHERE status='running'"):
            A.start(r["run_id"])
        for name in ("credit_pd", "churn", "anomaly"):   # load models now, not on the first click
            M.load(name)
        db.start_heartbeat()
        start_maintenance_timer()
    boot.done = True
    log.info("ready in %.1fs%s", time.time() - t0, " (serverless)" if light else "")


_boot_lock = __import__("threading").Lock()


@app.on_event("startup")
def _warm_start():
    """Azure App Service runs the app through uvicorn. Prepare it in the background as soon as the server starts,
    so the first visitor does not wait for it. Serverless hosts prepare on the first request instead."""
    if config.serverless() or getattr(boot, "done", False) or not config.configured():
        return
    import threading
    threading.Thread(target=ensure_booted, daemon=True, name="warm-start").start()


def ensure_booted():
    """Serverless: there is no start-up step, so the first request of each instance prepares it."""
    if getattr(boot, "done", False):
        return
    with _boot_lock:
        if not getattr(boot, "done", False):
            boot()


# ── errors and request log ──────────────────────────────────────

@app.middleware("http")
async def _timing(request: Request, call_next):
    t0 = time.time()
    try:
        if request.url.path.startswith("/api/") and not getattr(boot, "done", False):
            if not config.configured():
                return JSONResponse({"detail": "This deployment has no DATABASE_URL. Set it in the project settings."},
                                    status_code=503)
            import anyio
            await anyio.to_thread.run_sync(ensure_booted)
        response = await call_next(request)
    except Exception:
        ref = logs.reference()
        log.exception("%s %s failed, reference %s", request.method, request.url.path, ref)
        return JSONResponse({"detail": f"Something went wrong (reference {ref}). An administrator can find the "
                             "details under Diagnostic log.", "reference": ref}, status_code=500)
    ms = (time.time() - t0) * 1000
    if request.url.path.startswith("/api/"):
        level = 30 if ms > 3000 or response.status_code >= 500 else 20
        log.log(level, "%s %s %s %.0fms%s", request.method, request.url.path, response.status_code, ms,
                " SLOW" if ms > 3000 else "")
    return response


@app.exception_handler(PermissionError)
async def _perm(request, e):
    log.info("refused %s %s: %s", request.method, request.url.path, e)
    return JSONResponse({"detail": str(e)}, status_code=403)


@app.exception_handler(ValueError)
async def _val(request, e):
    log.info("not done %s %s: %s", request.method, request.url.path, e)
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(Exception)
async def _any(request, e):
    ref = logs.reference()
    log.exception("%s %s failed, reference %s", request.method, request.url.path, ref)
    return JSONResponse({"detail": f"Something went wrong (reference {ref}). An administrator can find the details "
                         "under Diagnostic log.", "reference": ref}, status_code=500)


# ── sign in ─────────────────────────────────────────────────────

def current_user(request: Request) -> dict:
    u = auth.user_for_token(request.cookies.get(COOKIE))
    if not u:
        raise HTTPException(401, "Sign in to continue.")
    return u


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Only an administrator can do this.")
    return user


def writable(user: dict = Depends(current_user)) -> dict:
    """Public showcase: settings, policies and staff accounts are read-only, so one visitor cannot change them for all."""
    if config.showcase():
        raise HTTPException(403, "This is read-only in the public demo. Run Analytixa on your own computer to change it.")
    return user


def _set_cookie(response, token):
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=config.hosted(),
                        max_age=auth.SESSION_HOURS * 3600)


@app.post("/api/login")
async def login(req: Request, response: Response):
    b = await req.json()
    token, user = auth.login(b.get("email", ""), b.get("password", ""))
    _set_cookie(response, token)
    return user


@app.get("/api/public")
def public_info():
    """What the sign-in page needs before anyone signs in."""
    return {"showcase": config.showcase(), "demo": auth.demo_accounts() if config.showcase() else []}


@app.post("/api/demo-login")
async def demo_login(req: Request, response: Response):
    if not config.showcase():
        raise HTTPException(404, "Not found.")
    b = await req.json()
    token, user = auth.demo_login(str(b.get("role", "")))
    _set_cookie(response, token)
    return user


@app.get("/api/cron/daily")
def cron_daily(request: Request):
    """Vercel Cron, once a day. Keeps the free database awake, refreshes the bank datasets and, in the public
    demo, removes uploads and open decisions older than a day. Vercel sends CRON_SECRET as a bearer token."""
    secret = config.get("CRON_SECRET", "")
    if not secret or request.headers.get("authorization", "") != f"Bearer {secret}":
        raise HTTPException(401, "Not allowed.")
    return daily_maintenance("system:cron")


def daily_maintenance(actor="system:maintenance") -> dict:
    """Refresh the bank datasets and, in the public demo, remove uploads and open decisions older than a day."""
    from app import bank_data as BK
    out = {"bank_datasets": [k for k, _ in BK.sync()]}
    if config.showcase():
        old = db.query("SELECT dataset_id, owner_email FROM datasets WHERE status='ready' AND owner_email<>'bank' "
                       "AND uploaded_at::timestamptz < now() - interval '1 day'")
        for d in old:
            who = db.one("SELECT email, role FROM app_users WHERE lower(email)=?", (d["owner_email"],))
            if who:
                DS.delete(d["dataset_id"], who)
        stale = db.query("SELECT DISTINCT run_id FROM checkpoints WHERE status='open' AND created_at < now() - interval '1 day'")
        for r in stale:
            db.execute("UPDATE checkpoints SET status='superseded' WHERE run_id=? AND status='open'", (r["run_id"],))
            db.execute("UPDATE runs SET status='cancelled' WHERE run_id=?", (r["run_id"],))
        out.update(uploads_removed=len(old), decisions_withdrawn=len(stale))
    db.ledger("daily_maintenance", actor=actor, payload=out)
    return out


def maintenance_due(hours=24) -> bool:
    """True when no daily maintenance is in the ledger for the last `hours` hours."""
    r = db.one("SELECT ts FROM ledger WHERE event='daily_maintenance' ORDER BY seq DESC LIMIT 1")
    if not r or not r["ts"]:
        return True
    import datetime as _dt
    last = _dt.datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
    if last.tzinfo is None:
        last = last.replace(tzinfo=_dt.timezone.utc)
    return _dt.datetime.now(_dt.timezone.utc) - last > _dt.timedelta(hours=hours)


def start_maintenance_timer(check_every=3600):
    """A long-running server (a laptop or Azure App Service with Always On) runs the daily maintenance itself.
    It checks once an hour and runs when the last run is a day old, so a restart never runs it twice."""
    import threading

    def loop():
        while True:
            try:
                if maintenance_due():
                    daily_maintenance()
            except Exception:
                log.exception("daily maintenance failed")
            time.sleep(check_every)
    threading.Thread(target=loop, daemon=True, name="daily-maintenance").start()


@app.post("/api/logout")
def logout(request: Request, response: Response):
    auth.logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return user


# ── pages ───────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta(user: dict = Depends(current_user)):
    co = db.one("SELECT name, short_name, country, currency, data_as_of FROM company")
    return {"company": co, "user": user, "roles": [{"id": r, "title": t, "description": d} for r, t, d in auth.ROLES],
            "use_cases": A.USE_CASES, "version": app.version,
            "uploads": {"allowed": user["role"] in P.body("uploads").get("roles", []) and P.body("uploads").get("enabled", True),
                        "max_file_mb": DS.max_file_mb(), "max_rows": P.body("uploads")["max_rows"]},
            "showcase": config.showcase(), "serverless": config.serverless(),
            "llm": llm.public(),
            "policy_versions": P.versions(), "models": M.registry(), "tools": T.catalogue(),
            "scopes": {k: sorted(v) for k, v in A.SCOPES.items()}, "agents": A.AGENT_LABELS,
            "plans": {k: [{"id": n, "title": t, "agent": a} for n, t, a, _ in v] for k, v in A.PLANS.items()},
            "database": {"schema": db.schema(), "host": _db_host()},
            "counts": {t: db.one(f"SELECT COUNT(*) n FROM {t}")["n"] for t in
                       ["customers", "transactions", "loans", "loan_applications", "atms"]}}


def _db_host():
    url = config.get("DATABASE_URL", "")
    return url.split("@")[-1].split("/")[0].split("?")[0] if "@" in url else "local"


@app.get("/api/home")
def home(user: dict = Depends(current_user)):
    role = user["role"]
    waiting = db.query("SELECT c.cp_id, c.run_id, c.kind, c.title, c.created_at, r.use_case, r.prompt, (r.state::json->'params'->>'dataset_name') AS dataset_name FROM checkpoints c "
                       "JOIN runs r ON r.run_id=c.run_id WHERE c.status='open' AND c.assigned_role=? "
                       "AND (c.assigned_user IS NULL OR lower(c.assigned_user)=lower(?)) "
                       "ORDER BY c.created_at", (role, user["email"]))
    mine = db.query("SELECT run_id, created_at, use_case, prompt, status, (state::json->'params'->>'dataset_name') AS dataset_name "
                    "FROM runs WHERE created_by=? "
                    "ORDER BY created_at DESC LIMIT 6", (user["email"],))
    out = {"waiting": waiting, "recent": mine}
    if role in ("loan_officer", "credit_manager"):
        out["pending_applications"] = db.one("SELECT COUNT(*) n FROM loan_applications WHERE status='pending'")["n"]
    return out


@app.post("/api/settings")
async def settings(req: Request, user: dict = Depends(admin_user)):
    writable()
    b = await req.json()
    llm.configure(b.get("api_key"), b.get("model"), b.get("enabled"), b.get("fallbacks"),
                  b.get("anthropic_key"), b.get("anthropic_model"))
    pub = llm.public()
    db.ledger("llm_settings_changed", actor=auth.actor(user),
              payload={k: pub[k] for k in ("model", "enabled", "fallbacks", "openrouter_key_set", "anthropic_key_set",
                                           "anthropic_model", "chain")})
    return pub


@app.get("/api/applications")
def applications(status: str = "pending", user: dict = Depends(current_user)):
    return db.query("SELECT a.app_id, a.customer_id, c.name, b.name AS branch, a.amount, a.tenor_months, a.purpose, "
                    "a.product, a.status, a.applied_on FROM loan_applications a "
                    "JOIN customers c ON c.customer_id=a.customer_id JOIN branches b ON b.branch_id=c.branch_id "
                    "WHERE (?='all' OR a.status=?) ORDER BY a.applied_on DESC, a.app_id", (status, status))


# ── runs ────────────────────────────────────────────────────────

@app.post("/api/runs")
async def new_run(req: Request, user: dict = Depends(current_user)):
    b = await req.json()
    if not b.get("prompt", "").strip():
        raise HTTPException(400, "Type a question or task first.")
    uc = b.get("use_case") or None
    if uc and uc not in A.PLANS:
        raise HTTPException(400, "Unknown task type.")
    params = b.get("params") or {}
    if uc in (None, "bi"):              # a full analysis or dashboard typed on Ask runs on the matching bank dataset
        from app import bank_data as BK
        did = BK.pick(b["prompt"], user["role"])
        if did:
            uc, params = "upload", {"dataset_id": did}
        elif BK.wants_dataset(b["prompt"]):
            raise HTTPException(400, BK.unmatched_message(user["role"]))
    if uc == "upload":
        ds = DS.get(params.get("dataset_id", ""), user, preview=False)     # only the uploader may ask
        params = {"dataset_id": ds["dataset_id"], "dataset_name": ds["name"], "question": b["prompt"].strip()}
    rid = A.create_run(b["prompt"].strip(), user["role"], "haara", uc, params, who=user)
    A.start(rid)
    return {"run_id": rid}


@app.get("/api/runs")
def runs(limit: int = 50, mine: int = 0, user: dict = Depends(current_user)):
    sql = ("SELECT r.run_id, r.created_at, r.use_case, r.prompt, r.role, r.status, r.created_by, u.full_name, "
           "(r.state::json->'params'->>'dataset_name') AS dataset_name "
           "FROM runs r LEFT JOIN app_users u ON lower(u.email)=r.created_by WHERE r.mode='haara' ")
    if mine:
        return db.query(sql + "AND r.created_by=? ORDER BY r.created_at DESC LIMIT ?", (user["email"], limit))
    return db.query(sql + "ORDER BY r.created_at DESC LIMIT ?", (limit,))


def _plan_approver(run_id: str, st: dict):
    """A full analysis is confirmed when its plan is approved before it runs, so nobody is asked twice."""
    if (st.get("result") or {}).get("type") != "report":
        return None
    row = db.one("SELECT decided_by FROM checkpoints WHERE run_id=? AND kind='plan' AND decision='approve' "
                 "ORDER BY decided_at DESC LIMIT 1", (run_id,))
    return f"{row['decided_by']}, who approved the plan before it ran" if row and row.get("decided_by") else None


@app.get("/api/runs/{run_id}")
def run(run_id: str, user: dict = Depends(current_user)):
    if config.serverless():          # no background threads: each status check moves a running task forward
        A.advance(run_id, 4.0)
    r = db.one("SELECT r.*, u.full_name FROM runs r LEFT JOIN app_users u ON lower(u.email)=r.created_by "
               "WHERE run_id=?", (run_id,))
    if not r:
        raise HTTPException(404, "This task was not found.")
    st = json.loads(r["state"])
    cp = db.one("SELECT * FROM checkpoints WHERE run_id=? AND status='open'", (run_id,))
    if cp:
        cp["evidence"] = json.loads(cp["evidence"])
        cp["mine"] = cp["assigned_role"] == user["role"] and \
            (not cp.get("assigned_user") or cp["assigned_user"].lower() == user["email"].lower())
        cp["personal"] = bool(cp.get("assigned_user"))
    return {"run_id": run_id, "use_case": r["use_case"], "prompt": r["prompt"], "role": r["role"], "mode": r["mode"],
            "status": r["status"], "created_at": r["created_at"], "created_by": r["full_name"] or r["created_by"],
            "plan": json.loads(r["plan"]),
            "result": st.get("result"), "params": st.get("params"), "policy_versions": st.get("policy_versions"),
            "models": st.get("models"), "error": st.get("error"), "blocked_reason": st.get("blocked_reason"),
            "confidence": st.get("confidence"), "checkpoint": cp, "asked_by_me": (r["created_by"] or "").lower() == user["email"].lower(),
            "interpretation_confirmed": st.get("interpretation_confirmed") or _plan_approver(run_id, st),
            "history": db.query("SELECT cp_id, kind, title, assigned_role, status, decision, decided_by, comment, created_at, "
                                "decided_at FROM checkpoints WHERE run_id=? ORDER BY created_at", (run_id,)),
            "dialogue": db.query("SELECT ts, speaker, text FROM dialogue WHERE run_id=? ORDER BY id", (run_id,)),
            "calls": db.query("SELECT seq, ts, actor, tool_id, action_class, level, node_id, event FROM ledger "
                              "WHERE run_id=? AND event IN ('tool_call','tool_blocked') ORDER BY seq", (run_id,))}


@app.post("/api/runs/{run_id}/pause")
def pause(run_id: str, user: dict = Depends(current_user)):
    A.pause(run_id, user["role"], who=user)
    return {"ok": True}


@app.post("/api/runs/{run_id}/steer")
async def steer(run_id: str, req: Request, user: dict = Depends(current_user)):
    b = await req.json()
    A.steer(run_id, b["node_id"], b.get("params") or {}, user["role"], who=user)
    return {"ok": True}


@app.post("/api/runs/{run_id}/correct")
async def correct(run_id: str, req: Request, user: dict = Depends(current_user)):
    b = await req.json()
    if not (b.get("text", "").strip() or b.get("intent")):
        raise HTTPException(400, "Say what the agent got wrong, or change the fields.")
    return A.correct_interpretation(run_id, b.get("text", "").strip(), user["role"], b.get("intent"), who=user)


@app.post("/api/runs/{run_id}/confirm")
async def confirm(run_id: str, req: Request, user: dict = Depends(current_user)):
    b = await req.json() if (await req.body()) else {}
    return A.confirm_interpretation(run_id, user["role"], who=user, plan_version=b.get("plan_version"),
                                    plan_hash=b.get("plan_hash"))


@app.get("/api/semantic")
def semantic(user: dict = Depends(current_user)):
    sl = P.body("semantic_layer")
    ent = P.body("entitlements").get(user["role"], {"metrics": []})
    metrics = [{"id": k, "label": m["label"], "fact": m["fact"]} for k, m in sl["metrics"].items()
               if ent["metrics"] == "*" or k in ent["metrics"]]
    return {"metrics": metrics, "dimensions": {d: sorted(v) for d, v in sl["dimensions"].items()},
            "values": {"city": T.CITY_VALUES, "region": T.REGION_VALUES, "segment": ["Retail", "SME"]},
            "periods": ["last month", "last quarter", "q2", "august", "september", "last 30 days", "last 90 days"]}


@app.post("/api/runs/{run_id}/challenge")
async def challenge(run_id: str, req: Request, user: dict = Depends(current_user)):
    b = await req.json()
    if not b.get("text", "").strip():
        raise HTTPException(400, "Type what you want the agent to reconsider.")
    return A.challenge(run_id, b["text"].strip(), user["role"], who=user)


# ── your data (uploads) ─────────────────────────────────────────

@app.post("/api/datasets")
async def upload_dataset(req: Request, user: dict = Depends(current_user)):
    data = await req.body()
    name = req.headers.get("x-file-name", "upload.csv")
    from urllib.parse import unquote
    return DS.create(unquote(name), data, user, req.query_params.get("name"))


@app.get("/api/datasets")
def list_datasets(user: dict = Depends(current_user)):
    from app import bank_data as BK
    return DS.mine(user) + BK.listing(user)


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str, user: dict = Depends(current_user)):
    out = DS.get(dataset_id, user)
    out["questions"] = db.query("SELECT run_id, created_at, prompt, status FROM runs WHERE use_case='upload' AND created_by=? "
                                "AND state::json->'params'->>'dataset_id'=? ORDER BY created_at DESC LIMIT 12",
                                (user["email"], dataset_id))
    return out


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user: dict = Depends(current_user)):
    return DS.delete(dataset_id, user)


# ── oversight ───────────────────────────────────────────────────

@app.get("/api/inbox")
def inbox(user: dict = Depends(current_user)):
    mine = db.query("SELECT c.cp_id, c.run_id, c.kind, c.level, c.title, c.created_at, r.use_case, r.prompt, (r.state::json->'params'->>'dataset_name') AS dataset_name FROM checkpoints c "
                    "JOIN runs r ON r.run_id=c.run_id WHERE c.status='open' AND c.assigned_role=? AND r.mode='haara' "
                    "AND (c.assigned_user IS NULL OR lower(c.assigned_user)=lower(?)) "
                    "ORDER BY c.created_at", (user["role"], user["email"]))
    done = db.query("SELECT c.cp_id, c.run_id, c.kind, c.title, c.decision, c.decided_at, c.decided_by, r.use_case "
                    "FROM checkpoints c JOIN runs r ON r.run_id=c.run_id WHERE c.status='closed' AND c.assigned_role=? "
                    "AND r.mode='haara' ORDER BY c.decided_at DESC LIMIT 20", (user["role"],))
    return {"mine": mine, "done": done}


@app.post("/api/checkpoints/{cp_id}/decide")
async def decide(cp_id: str, req: Request, user: dict = Depends(current_user)):
    b = await req.json()
    if b.get("decision") not in ("approve", "reject"):
        raise HTTPException(400, "Choose approve or reject.")
    A.decide(cp_id, b["decision"], user["role"], b.get("comment", ""), b.get("edits") or {}, who=user)
    return {"ok": True}


@app.get("/api/diagnostics")
def diagnostics(lines: int = 400, user: dict = Depends(admin_user)):
    """The latest log lines and the system facts needed to investigate a problem. Secrets are removed."""
    pub = llm.public()
    t0 = time.time()
    db.one("SELECT 1 AS ok")
    head = [f"Analytixa {app.version}, Python {platform.python_version()}, {platform.system()} {platform.release()}",
            f"Database host {_db_host()}, schema {db.schema()}, round trip now {1000 * (time.time() - t0):.0f} ms",
            f"Language model: {'on' if pub['available'] else 'off'}, order {', '.join(pub['chain']) or 'none'}",
            f"Last model answer: {pub['last'].get('model') or 'none'}, model calls {pub['stats']}",
            f"Generated {db.now()} for {user['email']}", "-" * 60]
    return {"text": logs.redact("\n".join(head)) + "\n" + logs.tail(max(50, min(lines, 2000)))}


@app.get("/api/llm")
def llm_status(user: dict = Depends(admin_user)):
    return llm.public()


@app.get("/api/calibration")
def calibration(user: dict = Depends(current_user)):
    return {"suggestions": C.calibration_suggestions()}


# ── policies ────────────────────────────────────────────────────

@app.get("/api/policies")
def policies(user: dict = Depends(current_user)):
    out = {}
    for k in P.DEFAULTS:
        b, v = P.get(k)
        out[k] = {"version": v, "body": b}
    return out


@app.get("/api/policies/{key}/history")
def policy_history(key: str, user: dict = Depends(current_user)):
    return P.history(key)


@app.put("/api/policies/{key}")
async def put_policy(key: str, req: Request, user: dict = Depends(current_user)):
    writable()
    b = await req.json()
    return P.update(key, b["body"], user["role"], b.get("note", ""), who=auth.actor(user))


@app.post("/api/policies/{key}/reset")
def reset_policy(key: str, user: dict = Depends(current_user)):
    writable()
    return P.reset(key, user["role"], who=auth.actor(user))


# ── staff accounts ──────────────────────────────────────────────

@app.get("/api/users")
def users(user: dict = Depends(admin_user)):
    return auth.list_users()


@app.post("/api/users")
async def add_user(req: Request, user: dict = Depends(admin_user)):
    writable()
    b = await req.json()
    return auth.create_user(b.get("full_name", ""), b.get("email", ""), b.get("role", ""), b.get("password", ""),
                            by=auth.actor(user))


@app.patch("/api/users/{user_id}")
async def edit_user(user_id: int, req: Request, user: dict = Depends(admin_user)):
    writable()
    b = await req.json()
    if user_id == user["user_id"] and (b.get("active") is False or b.get("role") not in (None, "admin")):
        raise ValueError("You cannot remove your own administrator access.")
    return auth.update_user(user_id, auth.actor(user), role=b.get("role"), active=b.get("active"),
                            password=b.get("password") or None, full_name=b.get("full_name"))


# ── provenance ──────────────────────────────────────────────────

@app.get("/api/ledger")
def ledger(run_id: str = None, limit: int = 100, offset: int = 0, user: dict = Depends(current_user)):
    if run_id:
        rows = db.query("SELECT * FROM ledger WHERE run_id=? ORDER BY seq DESC LIMIT ? OFFSET ?", (run_id, limit, offset))
    else:
        rows = db.query("SELECT * FROM ledger ORDER BY seq DESC LIMIT ? OFFSET ?", (limit, offset))
    return {"rows": rows, "total": db.one("SELECT COUNT(*) n FROM ledger")["n"]}


@app.get("/api/ledger/verify")
def verify(user: dict = Depends(current_user)):
    return db.verify_ledger()


@app.get("/api/external")
def external(user: dict = Depends(current_user)):
    return {t: db.query(f"SELECT * FROM {t} ORDER BY ts DESC LIMIT 50") for t in
            ["ext_loan_decisions", "ext_cases", "ext_holds", "ext_str_filings", "ext_campaigns", "ext_cash_orders"]}


@app.get("/api/evaluation")
def evaluation(user: dict = Depends(current_user)):
    p = ROOT / "evaluation" / "results.json"
    return json.loads(p.read_text()) if p.exists() else {"available": False}


if __name__ == "__main__":
    if not config.configured():
        print("No database is set up yet. Run:  python setup.py", flush=True)
        sys.exit(1)
    port = int(os.environ.get("PORT", 4000))
    try:
        boot(verbose=True)
    except Exception as e:  # show a plain message instead of a stack trace
        print(f"Could not start: {e}", flush=True)
        print("Check the connection string in config.env, or run python setup.py again.", flush=True)
        sys.exit(1)
    if not db.one("SELECT 1 FROM app_users WHERE role='admin' AND active"):
        print("No administrator account exists. Run:  python setup.py", flush=True)
        sys.exit(1)
    print(f"Ready. Open http://127.0.0.1:{port} in your browser. Keep this window open. Press Ctrl+C to stop.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
