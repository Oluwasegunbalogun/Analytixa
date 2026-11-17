"""Installation settings.

setup.py writes config.env in the project folder. Values already set in the
environment win over the file, so the evaluation can point at its own schema.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.env"


def read_file(path: Path = CONFIG_FILE) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def write_file(values: dict, path: Path = CONFIG_FILE) -> None:
    lines = ["# Analytixa HAARA settings. Written by setup.py.",
             "# This file holds your database password. Do not share it or commit it."]
    lines += [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


for _k, _v in read_file().items():
    os.environ.setdefault(_k, _v)


def get(key: str, default=None):
    return os.environ.get(key, default)


def configured() -> bool:
    return bool(get("DATABASE_URL"))


def serverless() -> bool:
    """True on Vercel (or with HAARA_SERVERLESS=1). No background threads, no local files, lazy start-up."""
    return bool(os.environ.get("VERCEL")) or os.environ.get("HAARA_SERVERLESS", "") == "1"


def azure() -> bool:
    """True on Azure App Service, which sets WEBSITE_SITE_NAME for every web app."""
    return bool(os.environ.get("WEBSITE_SITE_NAME"))


def hosted() -> bool:
    """Served over HTTPS by a hosting platform, so session cookies are marked secure."""
    return serverless() or azure() or os.environ.get("HAARA_SECURE_COOKIE", "") == "1"


def showcase() -> bool:
    """True for a public demonstration: one-click sign-in, read-only settings, daily clean-up of uploads."""
    return os.environ.get("HAARA_SHOWCASE", "") == "1"
