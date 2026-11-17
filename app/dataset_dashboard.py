"""Interactive dashboard for an uploaded dataset.

"Create a dashboard" becomes a dashboard plan: the measures to show as totals, the ratios the person defined, the date
and its grain, the columns to filter and break down by, and the identifier to rank. The server aggregates the rows into
a small cube (period x filter columns, with the sums and row counts). The page filters and recomputes from that cube,
so every number on screen is a sum of stored rows, and ratios are always ratios of sums. Totals are checked against SQL.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from . import dataset_report as RP
from . import datasets as DS
from . import l5_tools as T

WORDS = r"\bdashboards?\b|\bscorecard\b|\bkpi (?:page|board|view)\b"
GRAINS = {"day": r"\b(daily|by day|per day|each day)\b", "week": r"\b(weekly|by week|per week|each week)\b",
          "month": r"\b(monthly|by month|per month|each month)\b", "quarter": r"\b(quarterly|by quarter|per quarter)\b"}
MAX_DIMS, MAX_DIM_VALUES, TOP_ITEMS, MAX_CUBE = 4, 40, 300, 60000


def is_dashboard(question: str) -> bool:
    return bool(re.search(WORDS, (question or "").lower()))


# ═════════════════════════ the plan ═════════════════════════

def _mentioned(q, cols):
    seen = []
    for _, _, name in DS._mentions(" " + q + " ", cols):
        if name not in seen:
            seen.append(name)
    return seen


def build_plan(question: str, cols: list, values: dict, distinct: dict) -> dict:
    q = question.lower()
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    by = {c["name"]: c for c in usable}
    named = _mentioned(q, usable)
    meas_all = [c["name"] for c in usable if c["role"] == "measure"]
    meas = [n for n in named if n in meas_all] or meas_all[:6]
    groupable = [c["name"] for c in usable if c["role"] in ("category", "identifier")]
    small = [n for n in groupable if 1 < distinct.get(n, 0) <= MAX_DIM_VALUES]
    dims = [n for n in named if n in small] or [n for n in small if by[n]["role"] == "category"]   # named ones only, when named
    big = [n for n in groupable if distinct.get(n, 0) > MAX_DIM_VALUES]
    item = next((n for n in named if n in big), None) or next((n for n in big if by[n]["role"] == "identifier"), None) \
        or (big[0] if big else None)
    dates = [c["name"] for c in usable if c["role"] == "date"]
    date = next((n for n in named if n in dates), None) or (dates[0] if dates else None)
    grain = next((g for g, p in GRAINS.items() if re.search(p, q)), "auto")
    ratios = RP._ratios(q, [by[m] for m in meas_all])
    for r in ratios:                               # a ratio needs both of its measures on the page
        for side in ("num", "den"):
            if r[side] not in meas:
                meas.append(r[side])
    plan = {"type": "dashboard", "measures": meas, "ratios": ratios, "date": date, "grain": grain if date else None,
            "dims": dims[:MAX_DIMS], "item": item, "thresholds": {"min_denominator": 1000, "top_n": 10},
            "filters": {}, "source": "rules", "notes": []}
    plan["unavailable"] = [w for w in ("conversion", "revenue", "roas", "profit", "margin")
                           if re.search(rf"\b{w}", q) and not any(w in c["name"] for c in cols)]
    if len(dims) > MAX_DIMS:
        plan["notes"].append(f"{len(dims) - MAX_DIMS} more columns could be filters. Name them to use them instead.")
    return plan


def validate(plan: dict, cols: list) -> dict:
    by = {c["name"]: c for c in cols}
    p = dict(plan)
    p["measures"] = [m for m in dict.fromkeys(p.get("measures") or []) if by.get(m, {}).get("role") == "measure"][:8]
    p["ratios"] = [r for r in p.get("ratios") or [] if r["num"] in p["measures"] and r["den"] in p["measures"]]
    p["dims"] = [d for d in dict.fromkeys(p.get("dims") or []) if by.get(d, {}).get("role") in ("category", "identifier")][:MAX_DIMS]
    if p.get("item") and by.get(p["item"], {}).get("role") not in ("category", "identifier"):
        p["item"] = None
    if p.get("date") and by.get(p["date"], {}).get("role") != "date":
        p["date"] = None
    p["grain"] = p.get("grain") if p.get("date") and p.get("grain") in ("auto", "day", "week", "month", "quarter") else (
        "auto" if p.get("date") else None)
    if not p["measures"] and not p["dims"] and not p.get("date"):
        raise ValueError("This dataset has no number, group or date columns to build a dashboard from.")
    th = p.get("thresholds") or {}
    p["thresholds"] = {"min_denominator": max(0, float(th.get("min_denominator", 1000))),
                       "top_n": max(3, min(int(th.get("top_n", 10)), 50))}
    p["version"] = int(p.get("version") or 1)
    p["filters"] = {}
    p.setdefault("unavailable", [])
    p.setdefault("notes", [])
    return p


def revise(prev: dict, text: str, cols: list, distinct: dict) -> dict:
    """A correction in words: 'add State', 'remove Gender', 'only Amount and Outstanding', 'show it weekly'."""
    q = " " + text.lower() + " "
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    by = {c["name"]: c for c in usable}
    p = json.loads(json.dumps(prev))
    removed = set()
    for m in re.finditer(r"\b(?:remove|drop|without|exclude|leave out|take out|hide|no)\s+(?:the\s+)?([^.,;]+)", q):
        removed.update(_mentioned(m.group(1), usable))
    named = [n for n in _mentioned(q, usable) if n not in removed]
    only = bool(re.search(r"\bonly\b|\bjust\b|\binstead\b", q))
    meas = [n for n in named if by[n]["role"] == "measure"]
    dims = [n for n in named if by[n]["role"] in ("category", "identifier") and 1 < distinct.get(n, 0) <= MAX_DIM_VALUES]
    big = [n for n in named if by[n]["role"] in ("category", "identifier") and distinct.get(n, 0) > MAX_DIM_VALUES]
    if meas:
        p["measures"] = meas if only else p["measures"] + [m for m in meas if m not in p["measures"]]
    if dims:
        p["dims"] = dims if only and not meas else p["dims"] + [d for d in dims if d not in p["dims"]]
    if big:
        p["item"] = big[0]
    p["measures"] = [m for m in p["measures"] if m not in removed]
    p["dims"] = [d for d in p["dims"] if d not in removed]
    if p.get("item") in removed:
        p["item"] = None
    g = next((g for g, pat in GRAINS.items() if re.search(pat, q)), None)
    if g and p.get("date"):
        p["grain"] = g
    new = RP._ratios(q, [by[m] for m in by if by[m]["role"] == "measure"])
    if new:
        names = {r["name"] for r in new}
        p["ratios"] = [r for r in p["ratios"] if r["name"] not in names] + new
        for r in new:
            for side in ("num", "den"):
                if r[side] not in p["measures"]:
                    p["measures"].append(r[side])
    p["ratios"] = [r for r in p["ratios"] if r["name"].lower() not in {x.lower() for x in re.findall(r"\b(?:remove|drop|without|hide)\s+(\w+)", q)}]
    p["version"] = int(prev.get("version", 1)) + 1
    p["source"] = "corrected_text"
    return p


def _label(by, x):
    return by.get(x, x)


def describe(plan: dict, cols: list) -> str:
    by = {c["name"]: c["label"] for c in cols}
    parts = [f"Dashboard of {', '.join(_label(by, m) for m in plan['measures']) or 'row counts'}"]
    if plan["ratios"]:
        parts.append("with " + ", ".join(r["name"] for r in plan["ratios"]))
    out = " ".join(parts)
    if plan.get("date"):
        out += f", over time by {_label(by, plan['date'])} ({'grain chosen from the date range' if plan['grain'] == 'auto' else plan['grain']})"
    if plan["dims"]:
        out += ", filtered and broken down by " + ", ".join(_label(by, d) for d in plan["dims"])
    if plan.get("item"):
        out += f", top {plan['thresholds']['top_n']} {_label(by, plan['item'])}"
    return out + ". All rows."


def plan_details(plan: dict, cols: list) -> list:
    by = {c["name"]: c["label"] for c in cols}
    rows = [["Totals", ", ".join(_label(by, m) for m in plan["measures"]) + (", and the number of rows" if plan["measures"] else "Number of rows")]]
    for r in plan["ratios"]:
        rows.append([r["label"], f"{r['formula']}, recomputed for every filter. Blank when the denominator is 0. "
                                 f"{'Lower' if r['better'] == 'lower' else 'Higher'} is better."])
    if plan.get("date"):
        rows.append(["Over time", f"{_label(by, plan['date'])}, by {plan['grain'] if plan['grain'] != 'auto' else 'day, week or month, from the date range'}"])
    if plan["dims"]:
        rows.append(["Filters and breakdowns", ", ".join(_label(by, d) for d in plan["dims"])])
    if plan.get("item"):
        rows.append(["Ranking", f"Top {plan['thresholds']['top_n']} {_label(by, plan['item'])}"])
    rows.append(["Rows used", "Every row. Filters on the page narrow the view, not the data"])
    if plan.get("unavailable"):
        rows.append(["Not in the data", ", ".join(plan["unavailable"]) + ". Not shown and not estimated"])
    return rows


def plan_hash(plan: dict) -> str:
    core = {k: v for k, v in plan.items() if k not in ("version", "notes", "source")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ═════════════════════════ computing the dashboard ═════════════════════════

def _grain(span_days, grain):
    if grain and grain != "auto":
        return grain
    return "day" if span_days <= 62 else "week" if span_days <= 210 else "month"


def _period(s, grain):
    if grain == "day":
        out = s.dt.strftime("%Y-%m-%d")
    elif grain == "week":
        out = (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    elif grain == "month":
        out = s.dt.strftime("%Y-%m")
    else:
        out = s.dt.year.astype("Int64").astype(str) + "-Q" + s.dt.quarter.astype("Int64").astype(str)
    return out.where(s.notna(), "No date")


def _num(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    v = float(v)
    return int(v) if v.is_integer() and abs(v) < 2 ** 53 else round(v, 6)


def _cube(df, keys, cols):
    g = df.groupby(keys, dropna=False, sort=False)
    a = g[cols].sum(min_count=1) if cols else pd.DataFrame(index=g.size().index)
    a["__rows"] = g.size()
    return a.reset_index()


def compute(df: pd.DataFrame, plan: dict, cols: list, profile: dict) -> dict:
    by = {c["name"]: c["label"] for c in cols}
    M, R, dims, D, item = plan["measures"], plan["ratios"], list(plan["dims"]), plan.get("date"), plan.get("item")
    n = len(df)
    for m in M:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    sums = list(M)
    for r in R:                                   # ratio parts only where both values exist, as in the report
        both = df[r["num"]].notna() & df[r["den"]].notna()
        df[f"__n_{r['name']}"] = df[r["num"]].where(both)
        df[f"__d_{r['name']}"] = df[r["den"]].where(both)
        sums += [f"__n_{r['name']}", f"__d_{r['name']}"]
    for d in dims + ([item] if item else []):
        df[d] = df[d].astype("string").fillna("(blank)")
    grain, periods, span = None, [], None
    if D:
        dt = pd.to_datetime(df[D], errors="coerce")
        lo, hi = dt.min(), dt.max()
        span = [str(lo.date()) if pd.notna(lo) else None, str(hi.date()) if pd.notna(hi) else None]
        grain = _grain((hi - lo).days if pd.notna(lo) else 0, plan.get("grain"))
        df["__p"] = _period(dt, grain)
        periods = sorted(p for p in df["__p"].unique() if p != "No date") + (["No date"] if (df["__p"] == "No date").any() else [])
    keys = (["__p"] if D else []) + dims
    while True:                                   # keep the cube small enough for the page
        cube = _cube(df, keys, sums) if keys else _cube(df.assign(__all=1), ["__all"], sums)
        if len(cube) <= MAX_CUBE or not dims:
            break
        dims.pop()
        keys = (["__p"] if D else []) + dims
    lead = M[0] if M else "__rows"
    values = {}
    for d in dims:
        order = df.groupby(d, dropna=False)[lead].sum() if M else df.groupby(d, dropna=False).size()
        values[d] = [str(v) for v in order.sort_values(ascending=False).index]
    idx = {d: {v: i for i, v in enumerate(values[d])} for d in dims}
    pidx = {p: i for i, p in enumerate(periods)}
    rows = []
    for rec in cube.itertuples(index=False):
        r = dict(zip(cube.columns, rec))
        key = ([pidx[r["__p"]]] if D else []) + [idx[d][str(r[d])] for d in dims]
        rows.append(key + [_num(r[c]) for c in sums] + [int(r["__rows"])])
    items = None
    if item:
        tot = (df.groupby(item)[lead].sum() if M else df.groupby(item).size()).sort_values(ascending=False)
        keep = list(tot.index[:TOP_ITEMS])
        sub = df[df[item].isin(keep)]
        ik = [item] + (["__p"] if D else []) + dims
        icube = _cube(sub, ik, sums)
        while len(icube) > MAX_CUBE and len(keep) > 20:
            keep = keep[: len(keep) // 2]
            sub = df[df[item].isin(keep)]
            icube = _cube(sub, ik, sums)
        kidx = {v: i for i, v in enumerate(keep)}
        irows = []
        for rec in icube.itertuples(index=False):
            r = dict(zip(icube.columns, rec))
            irows.append([kidx[r[item]]] + ([pidx[r["__p"]]] if D else []) + [idx[d][str(r[d])] for d in dims]
                         + [_num(r[c]) for c in sums] + [int(r["__rows"])])
        items = {"name": item, "label": by[item], "values": [str(v) for v in keep], "rows": irows,
                 "total_distinct": int(df[item].nunique()), "covered": len(keep)}
    totals = {m: _num(df[m].sum(min_count=1)) for m in M}
    check = {m: abs(sum((r[len(keys) + i] or 0) for r in rows) - (totals[m] or 0)) <= max(0.01, abs(totals[m] or 0) * 1e-9)
             for i, m in enumerate(M)}
    issues = [{"column": i.get("column"), "issue": i.get("issue"), "severity": i.get("severity")}
              for i in (profile.get("issues") or []) if i.get("severity") in ("warning", "error")][:8]
    return {"rows": n, "totals": totals, "cube_matches_totals": check,
            "measures": [{"name": m, "label": by[m]} for m in M],
            "ratios": [{k: r[k] for k in ("name", "label", "num", "den", "mult", "better", "formula")} for r in R],
            "date": {"name": D, "label": by[D], "grain": grain, "periods": periods, "range": span} if D else None,
            "dims": [{"name": d, "label": by[d], "values": values[d]} for d in dims],
            "cube": {"columns": (["period"] if D else []) + dims + sums + ["rows"], "rows": rows},
            "items": items, "thresholds": plan["thresholds"], "issues": issues,
            "dropped_dims": [d for d in plan["dims"] if d not in dims]}


@T.tool("C14", "Dataset dashboard", "L5.4", "compute", policy="uploads")
def dataset_dashboard(ctx, frame, plan, columns, profile):
    """Aggregates the extracted rows into the cube behind the dashboard. Every tile is a sum of stored rows."""
    return compute(frame, plan, columns, profile)
