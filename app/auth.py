"""Staff accounts, roles and sign-in sessions (part of L8 Human interaction).

Passwords are stored as salted PBKDF2-SHA256 hashes. Session tokens are random
and only their SHA-256 hash is stored, so a copy of the database cannot be used
to sign in.
"""
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from . import l1_data as db

ROLES = [
    ("analyst", "Analyst", "Asks questions and releases reports"),
    ("loan_officer", "Loan officer", "Decides loans within delegated authority"),
    ("credit_manager", "Credit manager", "Approves credit exceptions and large loans"),
    ("compliance_officer", "Compliance officer", "Reviews AML alerts, holds and suspicious transaction reports"),
    ("retail_manager", "Retail manager", "Approves retention strategies and customer outreach"),
    ("branch_ops", "Branch operations", "Approves ATM cash orders outside the envelope"),
    ("admin", "Administrator", "Manages staff accounts, policies and settings"),
]
ROLE_IDS = [r[0] for r in ROLES]
ROLE_TITLES = {r[0]: r[1] for r in ROLES}
SESSION_HOURS = 12
ITERATIONS = 200_000

DDL = """
CREATE TABLE IF NOT EXISTS roles (
  role        text PRIMARY KEY,
  title       text NOT NULL,
  description text NOT NULL,
  sort_order  smallint NOT NULL
);
CREATE TABLE IF NOT EXISTS app_users (
  user_id       serial PRIMARY KEY,
  full_name     text NOT NULL,
  email         text NOT NULL,
  role          text NOT NULL REFERENCES roles(role),
  password_hash text NOT NULL,
  active        boolean NOT NULL DEFAULT true,
  created_at    timestamptz NOT NULL DEFAULT now(),
  last_login    timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email ON app_users (lower(email));
CREATE TABLE IF NOT EXISTS sessions (
  token_hash char(64) PRIMARY KEY,
  user_id    integer NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL
);
"""


def init() -> None:
    db.script(DDL)
    for i, (r, t, d) in enumerate(ROLES):
        db.execute("INSERT INTO roles VALUES (?,?,?,?) ON CONFLICT (role) DO UPDATE SET title=EXCLUDED.title, "
                   "description=EXCLUDED.description, sort_order=EXCLUDED.sort_order", (r, t, d, i))


def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${h}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split("$")
    except ValueError:
        return False
    test = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), int(it)).hex()
    return hmac.compare_digest(test, h)


def validate(full_name: str, email: str, role: str, password: str = None) -> None:
    if not full_name or len(full_name.strip()) < 2:
        raise ValueError("Enter the person's full name.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""):
        raise ValueError("Enter a valid email address.")
    if role not in ROLE_IDS:
        raise ValueError(f"Role must be one of: {', '.join(ROLE_IDS)}.")
    if password is not None and len(password) < 8:
        raise ValueError("Passwords need at least 8 characters.")


def _public(u):
    if not u:
        return None
    return {"user_id": u["user_id"], "full_name": u["full_name"], "email": u["email"], "role": u["role"],
            "role_title": ROLE_TITLES.get(u["role"], u["role"]), "active": u["active"],
            "created_at": u.get("created_at"), "last_login": u.get("last_login")}


def create_user(full_name, email, role, password, by="setup") -> dict:
    validate(full_name, email, role, password)
    if db.one("SELECT 1 FROM app_users WHERE lower(email)=lower(?)", (email,)):
        raise ValueError(f"A user with {email} already exists.")
    db.execute("INSERT INTO app_users (full_name, email, role, password_hash) VALUES (?,?,?,?)",
               (full_name.strip(), email.strip().lower(), role, hash_password(password)))
    u = db.one("SELECT * FROM app_users WHERE lower(email)=lower(?)", (email,))
    db.ledger("user_created", actor=by, payload={"user_id": u["user_id"], "email": u["email"], "role": role})
    return _public(u)


def update_user(user_id: int, by: str, role=None, active=None, password=None, full_name=None) -> dict:
    u = db.one("SELECT * FROM app_users WHERE user_id=?", (user_id,))
    if not u:
        raise ValueError("User not found.")
    changes = {}
    if full_name is not None and full_name.strip() != u["full_name"]:
        validate(full_name, u["email"], u["role"])
        changes["full_name"] = full_name.strip()
    if role is not None and role != u["role"]:
        validate(u["full_name"], u["email"], role)
        changes["role"] = role
    if active is not None and bool(active) != u["active"]:
        changes["active"] = bool(active)
    if password:
        validate(u["full_name"], u["email"], u["role"], password)
        changes["password_hash"] = hash_password(password)
    if not changes:
        return _public(u)
    if u["role"] == "admin" and (changes.get("role", "admin") != "admin" or changes.get("active") is False):
        others = db.one("SELECT COUNT(*) n FROM app_users WHERE role='admin' AND active AND user_id<>?", (user_id,))["n"]
        if not others:
            raise ValueError("Keep at least one active administrator.")
    sets = ", ".join(f"{k}=?" for k in changes)
    db.execute(f"UPDATE app_users SET {sets} WHERE user_id=?", (*changes.values(), user_id))
    if "active" in changes and not changes["active"] or "password_hash" in changes or "role" in changes:
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    db.ledger("user_updated", actor=by, payload={"user_id": user_id,
                                                 "changed": [k if k != "password_hash" else "password" for k in changes],
                                                 "role": changes.get("role"), "active": changes.get("active")})
    return _public(db.one("SELECT * FROM app_users WHERE user_id=?", (user_id,)))


def list_users() -> list:
    return [_public(u) for u in db.query("SELECT * FROM app_users ORDER BY active DESC, role, full_name")]


def login(email: str, password: str) -> tuple:
    u = db.one("SELECT * FROM app_users WHERE lower(email)=lower(?)", ((email or "").strip(),))
    if not u or not u["active"] or not check_password(password or "", u["password_hash"]):
        db.ledger("sign_in_failed", actor="system", payload={"email": (email or "").strip().lower()})
        raise PermissionError("The email or password is not correct.")
    return _session(u)


def _session(u, how="password") -> tuple:
    token = secrets.token_urlsafe(32)
    exp = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?,?,?)",
               (hashlib.sha256(token.encode()).hexdigest(), u["user_id"], exp.isoformat()))
    db.execute("UPDATE app_users SET last_login=now() WHERE user_id=?", (u["user_id"],))
    db.execute("DELETE FROM sessions WHERE expires_at < now()")
    db.ledger("signed_in", actor=actor(_public(u)), payload={"user_id": u["user_id"], "how": how})
    return token, _public(u)


# Only synthetic accounts on a .test address back the demo buttons, so a real person's account (such as the
# administrator who ran setup.py with their own email) is never offered to the public. A role with no .test
# account gets no button.
DEMO_ONLY = "lower(email) LIKE '%%.test'"


def demo_accounts() -> list:
    """Public showcase: one account per role, used by the 'Try as' buttons. Passwords are never needed or shown."""
    out = []
    for r, title, desc in ROLES:
        u = db.one(f"SELECT user_id, full_name FROM app_users WHERE role=? AND active AND {DEMO_ONLY} ORDER BY user_id LIMIT 1", (r,))
        if u:
            out.append({"role": r, "title": title, "description": desc, "name": u["full_name"]})
    return out


def demo_login(role: str) -> tuple:
    u = db.one(f"SELECT * FROM app_users WHERE role=? AND active AND {DEMO_ONLY} ORDER BY user_id LIMIT 1", (role,))
    if not u:
        raise PermissionError("There is no demo account for that role.")
    return _session(u, how="showcase")


def user_for_token(token: str):
    if not token:
        return None
    u = db.one("SELECT u.* FROM sessions s JOIN app_users u ON u.user_id=s.user_id "
               "WHERE s.token_hash=? AND s.expires_at > now() AND u.active",
               (hashlib.sha256(token.encode()).hexdigest(),))
    return _public(u)


def logout(token: str) -> None:
    if token:
        db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))


def actor(u: dict) -> str:
    """Ledger actor string: role plus the named person."""
    return f"human:{u['role']}:{u['email']}"


def label(u: dict) -> str:
    return f"{u['full_name']} ({ROLE_TITLES.get(u['role'], u['role'])})"
