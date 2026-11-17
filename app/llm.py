"""LLM access for L3 (foundation models) through OpenRouter, with Anthropic as a second provider.

Models are tried in order: the chosen OpenRouter model, then the fallback chain, then Anthropic when a key is set.
A model is skipped when it errors, times out, returns nothing or, for JSON calls, returns output that is not JSON.
Every switch is written to the ledger, and the model that answered is kept in LAST.

LLM output is never trusted directly. Callers validate it and fall back to deterministic logic when every model
fails, so every use case runs without an API key.
"""
import json
import os
import re
import threading
import time

import httpx

from . import config  # noqa: F401  loads config.env into the environment, so a key saved there is used

DEFAULT_MODEL = os.environ.get("HAARA_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
DEFAULT_FALLBACKS = ["poolside/laguna-s-2.1:free", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                     "openrouter/free", "openai/gpt-4o-mini", "deepseek/deepseek-chat"]
DEFAULT_ANTHROPIC_MODEL = os.environ.get("HAARA_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
SETTINGS = {"api_key": os.environ.get("OPENROUTER_API_KEY", ""), "model": DEFAULT_MODEL, "enabled": True,
            "fallbacks": list(DEFAULT_FALLBACKS), "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic_model": DEFAULT_ANTHROPIC_MODEL, "timeout": 25}
STATS = {"calls": 0, "ok": 0, "fallback": 0, "switches": 0}
LAST = {}
_local = threading.local()


def configure(api_key: str = None, model: str = None, enabled: bool = None, fallbacks=None,
              anthropic_key: str = None, anthropic_model: str = None):
    if api_key is not None:
        SETTINGS["api_key"] = api_key.strip()
    if model:
        SETTINGS["model"] = model.strip()
    if enabled is not None:
        SETTINGS["enabled"] = bool(enabled)
    if fallbacks is not None:
        items = fallbacks if isinstance(fallbacks, list) else re.split(r"[,\n]+", str(fallbacks))
        SETTINGS["fallbacks"] = [m.strip() for m in items if m and m.strip()][:10]
    if anthropic_key is not None:
        SETTINGS["anthropic_key"] = anthropic_key.strip()
    if anthropic_model:
        SETTINGS["anthropic_model"] = anthropic_model.strip()


def available() -> bool:
    return bool(SETTINGS["enabled"] and (SETTINGS["api_key"] or SETTINGS["anthropic_key"]))


def chain() -> list:
    """The order models are tried in, as (provider, model)."""
    out = []
    if SETTINGS["api_key"]:
        for m in [SETTINGS["model"]] + SETTINGS["fallbacks"]:
            if ("openrouter", m) not in out:
                out.append(("openrouter", m))
    if SETTINGS["anthropic_key"]:
        out.append(("anthropic", SETTINGS["anthropic_model"]))
    return out


def public() -> dict:
    """Settings safe to show in the interface. Keys are never returned."""
    return {"available": available(), "model": SETTINGS["model"], "enabled": SETTINGS["enabled"],
            "fallbacks": SETTINGS["fallbacks"], "openrouter_key_set": bool(SETTINGS["api_key"]),
            "anthropic_key_set": bool(SETTINGS["anthropic_key"]), "anthropic_model": SETTINGS["anthropic_model"],
            "chain": [f"{p}:{m}" for p, m in chain()], "last": dict(LAST), "stats": dict(STATS)}


def _openrouter(model, system, user, max_tokens):
    r = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=SETTINGS["timeout"],
                   headers={"Authorization": f"Bearer {SETTINGS['api_key']}",
                            "HTTP-Referer": "https://analytixa.local", "X-Title": "Analytixa HAARA"},
                   json={"model": model, "max_tokens": max_tokens, "temperature": 0.1,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    if r.status_code != 200:
        msg = ""
        try:
            msg = r.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {r.status_code} {msg}".strip()[:160])
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def _anthropic(model, system, user, max_tokens):
    r = httpx.post("https://api.anthropic.com/v1/messages", timeout=SETTINGS["timeout"],
                   headers={"x-api-key": SETTINGS["anthropic_key"], "anthropic-version": "2023-06-01",
                            "content-type": "application/json"},
                   json={"model": model, "max_tokens": max_tokens, "temperature": 0.1, "system": system,
                         "messages": [{"role": "user", "content": user}]})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}"[:160])
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


def _record_switch(frm, to, reason):
    STATS["switches"] += 1
    import logging
    logging.getLogger("analytixa.llm").warning("model %s did not answer (%s), trying %s", frm, reason[:120], to)
    try:
        from . import l1_data as db
        db.ledger("model_fallback", actor="L3.llm", payload={"from": frm, "to": to, "reason": reason[:200]})
    except Exception:
        pass


def _call(system, user, max_tokens, accept=None):
    """Try each model in the chain until one gives an acceptable answer."""
    if not available():
        return None
    STATS["calls"] += 1
    tried = []
    models = chain()
    for i, (prov, model) in enumerate(models):
        t0 = time.time()
        try:
            text = (_openrouter if prov == "openrouter" else _anthropic)(model, system, user, max_tokens)
            if not text:
                raise RuntimeError("empty response")
            value = accept(text) if accept else text
            if value is None:
                raise RuntimeError("output could not be read")
            STATS["ok"] += 1
            LAST.clear()
            LAST.update({"provider": prov, "model": model, "fallbacks_used": len(tried), "tried": tried,
                         "seconds": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _local.last = dict(LAST)
            return value
        except Exception as e:
            reason = str(e) or type(e).__name__
            tried.append({"model": f"{prov}:{model}", "reason": reason[:120]})
            if i + 1 < len(models):
                _record_switch(f"{prov}:{model}", f"{models[i + 1][0]}:{models[i + 1][1]}", reason)
    STATS["fallback"] += 1
    LAST.clear()
    LAST.update({"provider": None, "model": None, "fallbacks_used": len(tried), "tried": tried,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _local.last = dict(LAST)
    return None


def last_call() -> dict:
    """The model that answered the most recent call on this thread."""
    return dict(getattr(_local, "last", {}) or {})


def chat(system: str, user: str, max_tokens: int = 700, timeout: float = None) -> str | None:
    return _call(system, user, max_tokens)


def _parse_json(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def chat_json(system: str, user: str, max_tokens: int = 500) -> dict | None:
    return _call(system + "\nReply with one JSON object only. No prose, no code fences.", user, max_tokens, _parse_json)
