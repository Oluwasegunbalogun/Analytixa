"""Diagnostic log: what the server did, how long requests took, and every error with a reference.

The log is written to logs/analytixa.log (rotated at 1 MB, five files kept) and to the console.
Secrets are removed before anything is written: database passwords, API keys, session tokens and passwords.
An administrator can copy the latest lines from the menu under their name, Diagnostic log.
"""
import logging
import logging.handlers
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
if os.environ.get("VERCEL") or os.environ.get("HAARA_SERVERLESS") == "1":
    LOG_DIR = Path("/tmp/analytixa-logs")          # the deployed code folder is read-only; /tmp lasts per instance
elif os.environ.get("WEBSITE_SITE_NAME"):
    LOG_DIR = Path("/home/LogFiles/analytixa")      # Azure App Service: /home survives restarts and redeployments
LOG_FILE = LOG_DIR / "analytixa.log"

_SECRETS = [
    (re.compile(r"(postgres(?:ql)?://[^:/\s]+:)([^@\s]+)(@)"), r"\1***\3"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9]{8,}"), "sk-or-v1-***"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-***"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "sk-***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1***"),
    (re.compile(r"(?i)(\"?(?:password|pw|passwd|api_key|anthropic_key|token|secret|x-api-key)\"?\s*[:=]\s*\"?)([^\"\s,}&]+)"), r"\1***"),
    (re.compile(r"(?i)(haara_session=)[^;\s]+"), r"\1***"),
]


def redact(text: str) -> str:
    s = str(text)
    for pat, rep in _SECRETS:
        s = pat.sub(rep, s)
    return s


class _Redacting(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


log = logging.getLogger("analytixa")


def setup(console_level=logging.WARNING):
    if getattr(setup, "done", False):
        return log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = _Redacting("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(_Redacting("%(levelname)s: %(message)s"))
    ch.setLevel(logging.INFO if os.environ.get("VERCEL") else console_level)   # Vercel keeps what goes to the console
    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False
    setup.done = True
    return log


def reference() -> str:
    return "E-" + uuid.uuid4().hex[:6].upper()


def tail(lines: int = 400) -> str:
    """The latest lines of the log, already free of secrets."""
    files = sorted(LOG_DIR.glob("analytixa.log*"), key=lambda p: p.stat().st_mtime) if LOG_DIR.exists() else []
    out = []
    for f in files:
        try:
            out += f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
    return redact("\n".join(out[-lines:]))
