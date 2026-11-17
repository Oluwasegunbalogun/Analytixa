"""L1 Data and provenance.

Holds the bank database (PostgreSQL, for example a Supabase project) and the
append-only provenance ledger. Every ledger entry carries the hash of the
previous entry, so any edit or deletion of a past entry breaks the chain and is
detected by verify_ledger().

Each installation lives in its own Postgres schema (HAARA_SCHEMA). The
evaluation builds a second schema so it never touches live data.
"""
import functools
import logging
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
import numpy as np
from psycopg.types.bool import BoolDumper
from psycopg.types.numeric import FloatLoader, IntDumper
from psycopg.types.string import TextLoader

from . import config

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
_lock = threading.RLock()
_log = logging.getLogger("analytixa.db")
_conn = None
_last_used = 0.0
IDLE_RECONNECT = 60   # seconds; poolers drop idle sessions, so reconnect after a quiet spell


def schema() -> str:
    s = config.get("HAARA_SCHEMA", "bank")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", s):
        raise ValueError("HAARA_SCHEMA must be lower case letters, digits and underscores.")
    return s


def database_url() -> str:
    url = config.get("DATABASE_URL")
    if not url:
        raise RuntimeError("No database is configured. Run 'python setup.py' first.")
    if ("supabase" in url or "database.azure.com" in url) and "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _NpIntDumper(IntDumper):
    def dump(self, obj):
        return super().dump(int(obj))


class _NpBoolDumper(BoolDumper):
    def dump(self, obj):
        return super().dump(bool(obj))


class DatabaseUnreachable(psycopg.OperationalError):
    """Every connection attempt failed. Not retried again by _run."""


def _connect():
    """Open the connection, retrying with longer waits. A pooler or home network can be slow for a moment."""
    t0 = time.time()
    last = None
    for attempt, wait in enumerate((15, 30, 45), start=1):
        t1 = time.time()
        try:
            c = psycopg.connect(database_url(), autocommit=True, prepare_threshold=None,
                                row_factory=dict_row, connect_timeout=wait,
                                keepalives=1, keepalives_idle=20, keepalives_interval=10, keepalives_count=3)
            break
        except psycopg.OperationalError as e:
            last = e
            _log.warning("database connection attempt %s of 3 failed after %.1fs (limit %ss): %s", attempt,
                         time.time() - t1, wait, str(e).splitlines()[0][:160])
            if attempt < 3:
                time.sleep(2 * attempt)
    else:
        host = database_url().split("@")[-1].split("/")[0]
        raise DatabaseUnreachable(
            f"Could not reach the database at {host} after 3 tries over {time.time() - t0:.0f} seconds. "
            "Check your internet connection, open your Supabase project to confirm it is not paused, then try again. "
            f"Last error: {str(last).splitlines()[0][:160]}")
    # Return money as float, dates as ISO text and jsonb as text, so the
    # analytics code works with plain Python values.
    c.adapters.register_loader("numeric", FloatLoader)
    for t in (np.int64, np.int32, np.int16):
        c.adapters.register_dumper(t, _NpIntDumper)
    c.adapters.register_dumper(np.bool_, _NpBoolDumper)
    for t in ("date", "timestamp", "timestamptz", "jsonb", "json"):
        c.adapters.register_loader(t, TextLoader)
    c.execute(f"CREATE SCHEMA IF NOT EXISTS {schema()}")
    c.execute(f"SET search_path TO {schema()}, public")
    c.execute("SET TIME ZONE 'UTC'")
    _log.info("database connected in %.1fs (schema %s)", time.time() - t0, schema())
    return c


def conn():
    global _conn, _last_used
    if _conn is not None and not _conn.closed and time.time() - _last_used > IDLE_RECONNECT:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None
        _log.info("database connection idle for over %ss, reconnecting", IDLE_RECONNECT)
    if _conn is None or _conn.closed:
        _conn = _connect()
    _last_used = time.time()
    return _conn


def reset_connection():
    global _conn
    with _lock:
        if _conn is not None and not _conn.closed:
            _conn.close()
        _conn = None


@functools.lru_cache(maxsize=2048)
def _pg(sql: str) -> str:
    """Turn '?' placeholders into psycopg '%s', leaving quoted text alone."""
    out, quote = [], False
    for ch in sql:
        if ch == "'":
            quote = not quote
            out.append(ch)
        elif ch == "?" and not quote:
            out.append("%s")
        elif ch == "%":
            out.append("%%")
        else:
            out.append(ch)
    return "".join(out)


def _run(fn):
    with _lock:
        try:
            return fn(conn())
        except DatabaseUnreachable:
            raise
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            _log.warning("database connection dropped (%s), retrying once", str(e).splitlines()[0][:160])
            reset_connection()      # dropped by the pooler or the network: retry once
            return fn(conn())


def start_heartbeat(every: int = 30):
    """Keep the database session warm with a tiny query when the app is idle.
    Without it, the first click after a quiet minute waits for a fresh connection."""
    def beat():
        while True:
            time.sleep(every)
            if time.time() - _last_used < every - 2:
                continue
            try:
                t0 = time.time()
                _run(lambda c: c.execute("SELECT 1").fetchall())
                if time.time() - t0 > 3:
                    _log.warning("database slow: keep-alive query took %.1fs", time.time() - t0)
            except Exception as e:
                _log.warning("database keep-alive failed: %s", str(e).splitlines()[0][:160])
    threading.Thread(target=beat, daemon=True, name="db-heartbeat").start()


def query(sql: str, params=()) -> list:
    p = tuple(params) if params else None
    return _run(lambda c: c.execute(_pg(sql) if p else sql, p).fetchall())


def one(sql: str, params=()):
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=()) -> int:
    p = tuple(params) if params else None
    return _run(lambda c: c.execute(_pg(sql) if p else sql, p).rowcount)


def executemany(sql: str, rows) -> None:
    rows = list(rows)

    def go(c):
        with c.transaction():
            c.cursor().executemany(_pg(sql), rows)
    _run(go)


def atomic(fn):
    """Run fn inside one database transaction (all of its statements commit together)."""
    def go(c):
        with c.transaction():
            return fn()
    return _run(go)


def script(sql: str) -> None:
    _run(lambda c: c.execute(sql))


def copy_rows(table: str, columns: list, rows) -> None:
    """Bulk load with COPY. Much faster than INSERT over a network link."""
    def go(c):
        with c.transaction(), c.cursor() as cur:
            with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as cp:
                for r in rows:
                    cp.write_row(r)
    _run(go)


def table_exists(name: str) -> bool:
    return bool(one("SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
                    (schema(), name)))


def drop_schema() -> None:
    script(f"DROP SCHEMA IF EXISTS {schema()} CASCADE")
    reset_connection()


CORE = """
CREATE TABLE IF NOT EXISTS ledger (
  seq            bigserial PRIMARY KEY,
  ts             text NOT NULL,          -- kept as text so the hash is byte-stable
  run_id         text,
  node_id        text,
  actor          text NOT NULL,          -- agent name, 'human:<role>:<user>', or 'system'
  event          text NOT NULL,
  tool_id        text,
  action_class   text,
  level          text,
  policy_version integer,
  model_version  text,
  payload        text,
  prev_hash      char(64) NOT NULL,
  hash           char(64) NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_ledger_run ON ledger(run_id);

CREATE TABLE IF NOT EXISTS policies (
  key        text NOT NULL,
  version    integer NOT NULL,
  body       json NOT NULL,          -- json keeps the key order an admin sees
  created_at timestamptz NOT NULL,
  created_by text NOT NULL,
  note       text,
  PRIMARY KEY (key, version)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id     text PRIMARY KEY,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  use_case   text NOT NULL,
  prompt     text NOT NULL,
  role       text NOT NULL,
  mode       text NOT NULL CHECK (mode IN ('haara','autonomous','uniform','guarded')),
  status     text NOT NULL,
  plan       text,
  state      text,
  result     text,
  created_by text
);
CREATE INDEX IF NOT EXISTS ix_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
  cp_id         text PRIMARY KEY,
  run_id        text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  node_id       text,
  kind          text NOT NULL,
  level         text,
  assigned_role text NOT NULL,
  assigned_user text,                  -- set when the decision belongs to the person who asked
  title         text,
  evidence      text,
  status        text NOT NULL CHECK (status IN ('open','closed','superseded')),
  created_at    timestamptz NOT NULL,
  decided_at    timestamptz,
  decided_by    text,
  decision      text CHECK (decision IN ('approve','reject')),
  comment       text,
  edits         text
);
CREATE INDEX IF NOT EXISTS ix_cp_open ON checkpoints(status, assigned_role);
CREATE INDEX IF NOT EXISTS ix_cp_run ON checkpoints(run_id);

CREATE TABLE IF NOT EXISTS notifications (
  id     bigserial PRIMARY KEY,
  ts     timestamptz NOT NULL,
  run_id text REFERENCES runs(run_id) ON DELETE CASCADE,
  role   text NOT NULL,
  title  text,
  body   text,
  seen   boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS oversight_memory (
  id       bigserial PRIMARY KEY,
  ts       timestamptz NOT NULL,
  run_id   text REFERENCES runs(run_id) ON DELETE CASCADE,
  use_case text, kind text, decision text,
  changed  integer, role text, comment text
);

CREATE TABLE IF NOT EXISTS dialogue (
  id      bigserial PRIMARY KEY,
  ts      timestamptz NOT NULL,
  run_id  text REFERENCES runs(run_id) ON DELETE CASCADE,
  speaker text, text text, data text
);
"""


MIGRATIONS = """
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS assigned_user text;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS lease_until timestamptz;
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_mode_check;
ALTER TABLE runs ADD CONSTRAINT runs_mode_check CHECK (mode IN ('haara','autonomous','uniform','guarded'));
-- one row per external effect: a second attempt with the same key returns the first result
CREATE TABLE IF NOT EXISTS action_effects (
  effect_key text PRIMARY KEY,
  tool_id    text NOT NULL,
  run_id     text,
  ts         timestamptz NOT NULL DEFAULT now(),
  result     text
);
-- copies of the ledger head kept outside the chain (see anchor_ledger)
CREATE TABLE IF NOT EXISTS ledger_anchors (
  id      bigserial PRIMARY KEY,
  ts      timestamptz NOT NULL DEFAULT now(),
  seq     bigint NOT NULL,
  head    char(64) NOT NULL,
  entries bigint NOT NULL
);
"""


def init_core() -> None:
    script(CORE)
    script(MIGRATIONS)


# ── provenance ledger ─────────────────────────────────────────────

GENESIS = "0" * 64


def _digest(fields: dict) -> str:
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def ledger(event: str, actor: str = "system", run_id=None, node_id=None, tool_id=None,
           action_class=None, level=None, policy_version=None, model_version=None,
           payload=None) -> dict:
    """Append one entry to the hash-chained provenance ledger."""
    entry = {
        "ts": now(), "run_id": run_id, "node_id": node_id, "actor": actor, "event": event,
        "tool_id": tool_id, "action_class": action_class, "level": level,
        "policy_version": policy_version, "model_version": model_version,
        "payload": json.dumps(payload, default=str) if payload is not None else None,
    }

    def go(c):
        with c.transaction():
            # one writer at a time across every process that shares this schema
            c.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema() || '.ledger'))")
            last = c.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
            e = dict(entry, prev_hash=last["hash"] if last else GENESIS)
            e["hash"] = _digest(e)
            c.execute("INSERT INTO ledger (ts,run_id,node_id,actor,event,tool_id,action_class,level,"
                      "policy_version,model_version,payload,prev_hash,hash) VALUES "
                      "(%(ts)s,%(run_id)s,%(node_id)s,%(actor)s,%(event)s,%(tool_id)s,%(action_class)s,"
                      "%(level)s,%(policy_version)s,%(model_version)s,%(payload)s,%(prev_hash)s,%(hash)s)", e)
            return e
    return _run(go)


def check_chain(rows, anchor=None) -> dict:
    """Recompute the hash chain over rows ordered by seq.
    Without an anchor this detects edits, insertions, deletions and reordering inside the
    chain. It cannot detect removal of the newest entries or a full rewrite by someone who
    recomputes every hash. An anchor (a head hash kept elsewhere) closes those gaps."""
    prev = GENESIS
    by_seq = {}
    for r in rows:
        fields = {k: r[k] for k in ("ts", "run_id", "node_id", "actor", "event", "tool_id",
                                    "action_class", "level", "policy_version", "model_version",
                                    "payload", "prev_hash")}
        if r["prev_hash"] != prev or _digest(fields) != r["hash"]:
            return {"ok": False, "entries": len(rows), "broken_at": r["seq"], "reason": "chain"}
        prev = r["hash"]
        by_seq[r["seq"]] = r["hash"]
    if anchor:
        if by_seq.get(anchor["seq"]) != anchor["head"]:
            return {"ok": False, "entries": len(rows), "broken_at": anchor["seq"], "reason": "anchor"}
    return {"ok": True, "entries": len(rows), "head": prev}


def anchor_ledger() -> dict:
    """Record the current head. Keep a copy outside the database (for example in a signed
    email or an external store) so a rewrite of the whole table is detectable."""
    r = one("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1")
    n = one("SELECT COUNT(*) n FROM ledger")["n"]
    if not r:
        return {}
    execute("INSERT INTO ledger_anchors (seq, head, entries) VALUES (?,?,?)", (r["seq"], r["hash"], n))
    return {"seq": r["seq"], "head": r["hash"], "entries": n}


def verify_ledger(anchor=None) -> dict:
    """Recompute the hash chain. Any edited, inserted or deleted entry breaks it."""
    return check_chain(query("SELECT * FROM ledger ORDER BY seq"), anchor)
