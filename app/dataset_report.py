"""Full analysis of an uploaded dataset: a multi-section report built only from the person's data.

A long request ("analyse the dataset: overall performance, efficiency, trends, data quality ...") does not fit one
query. It becomes a report plan instead: the measures, the ratios the person defined, the levels to compare, the time
grains and the sections. The person approves the plan before anything runs. Every figure is computed from the data,
every formula is shown, totals are checked against SQL, and anything the data cannot support is listed as a limitation.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from . import datasets as DS
from . import l1_data as db
from . import l5_tools as T

SECTION_CUES = {
    "overall": r"overall|total|summary|headline",
    "efficiency": r"efficien|ratio|rate|cost per|per click|per thousand|\bctr\b|\bcpc\b|\bcpm\b|divided by",
    "groups": r"compare|comparison|across|each|per |by |differences between|level performance|rank",
    "trends": r"trend|over time|by day|by week|by month|daily|weekly|monthly|period|time",
    "relationships": r"relationship|correlat|associated|association|scatter",
    "quality": r"data quality|missing|duplicate|anomal|invalid|outlier|negative",
    "concentration": r"concentrat|distribut|share of|top accounts|top ads|depend",
    "recommendations": r"recommend|action|next step",
}
REPORT_WORDS = r"comprehensive|full analysis|complete analysis|in-depth|analysis report|analy[sz]e (?:the|this|my) (?:uploaded |whole |entire )?(?:\w+ )?(?:data|dataset|file)"
COST = r"spend|cost|fee|expense|budget"          # a ratio of these is a cost: lower is better unless the person says otherwise


def is_report(question: str) -> bool:
    q = question.lower()
    sections = sum(bool(re.search(p, q)) for p in SECTION_CUES.values())
    return bool(re.search(REPORT_WORDS, q)) and sections >= 3 or sections >= 5 or (len(q) > 700 and sections >= 4)


# ═════════════════════════ the report plan ═════════════════════════

def _ratios(q, meas):
    """Ratios the person defined, such as 'CTR = total clicks divided by total impressions, multiplied by 100'."""
    out = []
    pat = re.compile(r"(?:^|[\n\-*•]|\.\s)\s*([a-z][a-z /\-]{1,60}?)\s*(?:\(([a-z]{2,6})\))?\s*=\s*(?:the\s+)?(?:total\s+|sum of\s+)?"
                     r"([a-z_ ]{2,40}?)\s+divided by\s+(?:the\s+)?(?:total\s+|sum of\s+)?([a-z_ ]{2,40}?)"
                     r"(?:\s*,?\s*(?:multiplied|times)\s+by\s+([\d,]+))?\s*(?=[.;\n]|$)")
    for m in pat.finditer(q):
        num, den = DS._best(" " + m.group(3) + " ", meas), DS._best(" " + m.group(4) + " ", meas)
        if not (num and den) or num["name"] == den["name"]:
            continue
        word = m.group(1).strip()
        short = re.fullmatch(r"[a-z]{2,5}", word)              # 'CTR = ...' is an abbreviation, kept in capitals
        name = (m.group(2) or "").upper() or (word.upper() if short else re.sub(r"\s+", " ", word)[:30].capitalize())
        label = (word.upper() if short else word.capitalize()) + (f" ({m.group(2).upper()})" if m.group(2) else "")
        mult = float((m.group(5) or "1").replace(",", ""))
        tail = q[m.end():m.end() + 80].split("\n")[0]
        said = re.search(r"\b(lower|higher) (?:is|are) better\b", tail)
        cost = said.group(1) == "lower" if said else bool(re.search(COST, num["name"]))
        out.append({"name": name, "label": label, "num": num["name"], "den": den["name"], "mult": mult,
                    "better": "lower" if cost else "higher", "kind": "cost" if cost else "rate",
                    "formula": f"SUM({num['name']}) / SUM({den['name']})" + (f" x {mult:g}" if mult != 1 else "")})
    seen, uniq = set(), []
    for r in out:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)
    return uniq


def _per(q, meas, ids):
    """'spend per ad', 'spend per account': a measure's total over the number of distinct values of an identifier."""
    out = []
    for m in re.finditer(r"\b([a-z_ ]{2,30}?)\s+per\s+([a-z_ ]{2,30}?)(?=[,.;\n]|\s+and\b|$)", q):
        x, y = DS._best(" " + m.group(1) + " ", meas), DS._best(" " + m.group(2) + " ", ids)
        if x and y and (x["name"], y["name"]) not in [(p["measure"], p["per"]) for p in out]:
            out.append({"measure": x["name"], "per": y["name"], "label": f"{x['label']} per {y['label']}",
                        "formula": f"SUM({x['name']}) / COUNT(DISTINCT {y['name']})"})
    return out


def build_plan(question: str, cols: list, values: dict, distinct: dict) -> dict:
    q = question.lower()
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    meas = [c for c in usable if c["role"] == "measure"]
    seen = []
    for _, _, name in DS._mentions(" " + q + " ", usable):     # in the order the person named them
        if name not in seen:
            seen.append(name)
    named_meas = [next(c for c in meas if c["name"] == n) for n in seen if any(c["name"] == n for c in meas)]
    if named_meas:                        # only the measures the person named; summing every number column misleads
        meas = named_meas
    ids = [c for c in usable if c["role"] in ("identifier", "category")]
    dates = [c for c in usable if c["role"] == "date"]
    named = [c for c in ids if DS._find(" " + q + " ", c)] or ids
    groups = [c["name"] for c in named if distinct.get(c["name"], 0) <= 50]
    items = [c["name"] for c in named if 50 < distinct.get(c["name"], 0)]
    tq = " ".join(l for l in q.split("\n") if re.search(SECTION_CUES["trends"].replace("|time", "|over time"), l) and "per day" not in l) or q
    grains = [g for g in ("day", "week", "month") if re.search(r"\b(day|daily)\b" if g == "day" else rf"\b{g}", tq)] or ["month"]
    sections = ["overall"] + [k for k, p in SECTION_CUES.items() if k != "overall" and re.search(p, q)]
    if "groups" in sections:
        sections.remove("groups")
    order = ["overall", "efficiency"] + [f"group:{g}" for g in groups] + [f"item:{i}" for i in items] + \
        ["trends", "relationships", "quality", "concentration", "recommendations"]
    have = set(sections) | {f"group:{g}" for g in groups} | {f"item:{i}" for i in items}
    plan = {"type": "report", "measures": [c["name"] for c in meas], "ratios": _ratios(q, meas), "per": _per(q, meas, ids),
            "groups": groups, "items": items, "date": dates[0]["name"] if dates else None, "grains": grains if dates else [],
            "sections": [s for s in order if s in have and (s != "trends" or dates)] + ["summary"],
            "filters": {}, "thresholds": {"min_denominator": 1000, "top_n": 10}, "source": "rules", "notes": []}
    unknown = [w for w in ("conversion", "revenue", "roas", "return on ad spend", "acquisition cost", "profit", "sales")
               if re.search(rf"\b{w}", q) and not any(w.split()[0] in c["name"] for c in cols)]
    plan["unavailable"] = unknown
    if not plan["ratios"] and "efficiency" in plan["sections"]:
        plan["notes"].append("No ratio formulas were given, so efficiency ratios are not computed. Define them as "
                             "'NAME = total X divided by total Y'.")
    return plan


def validate(plan: dict, cols: list) -> dict:
    by = {c["name"]: c for c in cols}
    p = dict(plan)
    for k in ("measures",):
        p[k] = [m for m in p.get(k) or [] if by.get(m, {}).get("role") == "measure"]
    p["ratios"] = [r for r in p.get("ratios") or [] if r["num"] in p["measures"] and r["den"] in p["measures"]]
    p["per"] = [r for r in p.get("per") or [] if r["measure"] in p["measures"] and by.get(r["per"], {}).get("role") in ("identifier", "category")]
    p["groups"] = [g for g in p.get("groups") or [] if by.get(g, {}).get("role") in ("identifier", "category")]
    p["items"] = [g for g in p.get("items") or [] if by.get(g, {}).get("role") in ("identifier", "category")]
    if p.get("date") and by.get(p["date"], {}).get("role") != "date":
        p["date"] = None
    if not p["measures"]:
        raise ValueError("This dataset has no number columns to analyse.")
    th = p.get("thresholds") or {}
    p["thresholds"] = {"min_denominator": max(0, float(th.get("min_denominator", 1000))), "top_n": max(3, min(int(th.get("top_n", 10)), 50))}
    p["version"] = int(p.get("version") or 1)
    p["filters"] = {}
    return p


def describe(plan: dict, cols: list) -> str:
    by = {c["name"]: c["label"] for c in cols}
    names = lambda xs: ", ".join(by.get(x, x) for x in xs)
    titles = {"overall": "overall totals", "efficiency": "efficiency ratios", "trends": "trends", "relationships":
              "relationships", "quality": "data quality", "concentration": "concentration", "recommendations":
              "recommendations", "summary": "summary"}
    secs = [titles.get(s) or (f"comparison by {by.get(s.split(':')[1])}" if s.startswith("group:") else
                              f"ranking by {by.get(s.split(':')[1])}") for s in plan["sections"]]
    out = f"Full analysis of {names(plan['measures'])}"
    if plan["ratios"]:
        out += ", with " + ", ".join(r["name"] for r in plan["ratios"])
    out += ", in " + str(len(secs)) + " sections: " + ", ".join(secs) + ". All rows, no filters."
    return out


def plan_details(plan: dict, cols: list) -> list:
    """The plan as rows for the person to check before it runs."""
    by = {c["name"]: c["label"] for c in cols}
    rows = [["Measures", ", ".join(by.get(m, m) for m in plan["measures"]) + ": totals (SUM), averages per row and per day (AVG)"]]
    for r in plan["ratios"]:
        rows.append([r["label"], f"{r['formula']}. Undefined (not zero) when the denominator is 0 or missing. "
                                 f"{'Lower' if r['better'] == 'lower' else 'Higher'} is better."])
    for r in plan["per"]:
        rows.append([r["label"], r["formula"]])
    if plan.get("date"):
        rows.append(["Daily averages", f"Total divided by the number of distinct {by.get(plan['date'])} values"])
        rows.append(["Time grains", ", ".join(plan["grains"]) + f" of {by.get(plan['date'])}"])
    if plan["groups"]:
        rows.append(["Compared by", ", ".join(by.get(g, g) for g in plan["groups"])])
    if plan["items"]:
        rows.append(["Ranked by", ", ".join(by.get(g, g) for g in plan["items"]) +
                     f". Ratios ranked only where the denominator is at least {plan['thresholds']['min_denominator']:,.0f}"])
    rows.append(["Filters", "None. Every row is used"])
    names = {"overall": "Overall performance", "efficiency": "Efficiency", "trends": "Trends over time",
             "relationships": "Spend and engagement" if _is_ads(plan) else "Relationships", "quality": "Data quality", "concentration": "Concentration",
             "recommendations": "Recommendations", "summary": "Executive summary"}
    rows.append(["Sections", ", ".join(names.get(x) or f"Performance by {by.get(x.split(':')[1], x)}" for x in plan["sections"])])
    if plan.get("unavailable"):
        rows.append(["Not in the data", ", ".join(plan["unavailable"]) + ". These are reported as unavailable, not estimated"])
    return rows


def plan_hash(plan: dict) -> str:
    import hashlib
    core = {k: v for k, v in plan.items() if k not in ("version", "notes", "source")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ═════════════════════════ computing the report ═════════════════════════

def _f(v, d=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), d)


def _n(v):
    """A number for a sentence: whole numbers with separators, small values with four significant figures."""
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}" if abs(v) >= 10 else f"{v:.4g}"


def _ratio(num_sum, den_sum, mult):
    return None if not den_sum or pd.isna(den_sum) else num_sum / den_sum * mult


def _agg(df, keys, plan):
    """Totals, ratios and activity per group, with ratios undefined where the denominator is 0."""
    g = df.groupby(keys, dropna=False)
    out = g[plan["measures"]].sum(min_count=1)
    for r in plan["ratios"]:
        both = df[df[r["num"]].notna() & df[r["den"]].notna()]
        s = both.groupby(keys, dropna=False)[[r["num"], r["den"]]].sum()
        out[r["name"]] = (s[r["num"]] / s[r["den"]].where(s[r["den"]] > 0) * r["mult"]).reindex(out.index)
    if plan.get("date"):
        out["active_days"] = g[plan["date"]].nunique()
    out["rows"] = g.size()
    return out


def _table(frame, labels, n=None, digits=2):
    f = frame.reset_index()
    f = f.head(n) if n else f
    cols = [str(c) for c in f.columns]
    rows = [[(None if pd.isna(v) else (round(float(v), digits) if isinstance(v, (float, np.floating)) else
             (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else str(v)))) for v in rec]
            for rec in f.itertuples(index=False)]
    return {"columns": [labels.get(c, c) for c in cols], "rows": rows}


def _is_ads(plan):
    """Advertising data (clicks and impressions) gets advertising wording. Any other data gets neutral wording."""
    return any(re.search(r"click|impression", r["num"] + r["den"]) for r in plan["ratios"])


def _levels(plan, by):
    names = [by.get(x, x) for x in plan["items"] + plan["groups"]]
    return " or ".join(names) if names else "groups"


def compute(df: pd.DataFrame, plan: dict, cols: list, profile: dict) -> dict:
    by = {c["name"]: c["label"] for c in cols}
    ads = _is_ads(plan)
    L = dict(by)
    for r in plan["ratios"]:
        L[r["name"]] = r["name"]
    L.update({"active_days": "Active days", "rows": "Rows"})
    M, R = plan["measures"], plan["ratios"]
    D = plan.get("date")
    th = plan["thresholds"]
    if D:
        df[D] = pd.to_datetime(df[D], errors="coerce")
    sections, findings, recs = [], {}, []
    n = len(df)
    eng = next((r for r in R if r["better"] == "higher"), None)
    cost = next((r for r in R if r["better"] == "lower"), None)
    if not ads:                 # the paired spend/engagement rules below are for advertising data
        eng = cost = None
    lead = R[0] if R else None  # the first ratio the person defined
    totals = {m: _f(df[m].sum(min_count=1)) for m in M}
    days = int(df[D].nunique()) if D else None

    # 1 overall
    if "overall" in plan["sections"]:
        rows = []
        for m in M:
            rows.append([f"Total {by[m]}", _f(df[m].sum(min_count=1)), f"SUM({m})"])
            if days:
                rows.append([f"Average {by[m]} per day", _f(df[m].sum() / days), f"SUM({m}) / {days} distinct days"])
            rows.append([f"Average {by[m]} per row", _f(df[m].mean()), f"AVG({m}), {int(df[m].notna().sum()):,} rows with a value"])
        for c in plan["groups"] + plan["items"]:
            rows.append([f"Distinct {by[c]}", int(df[c].nunique()), f"COUNT(DISTINCT {c})"])
        if D:
            rows.append([f"Distinct {by[D]}", days, f"COUNT(DISTINCT {D}), {df[D].min().date()} to {df[D].max().date()}"])
        rows.append(["Total rows", n, "COUNT(*)"])
        for p in plan["per"]:
            k = df[p["per"]].nunique()
            rows.append([p["label"], _f(df[p["measure"]].sum() / k if k else None), p["formula"]])
        sections.append({"id": "overall", "title": "Overall performance",
                         "text": ["Sums, averages and distinct counts are labelled separately. Averages per day divide the total "
                                  "by the number of distinct dates, not by the number of rows."],
                         "tables": [{"columns": ["Measure", "Value", "Formula"], "rows": rows}]})

    # 2 efficiency
    if "efficiency" in plan["sections"] and R:
        rows, text = [], []
        for r in R:
            both = df[df[r["num"]].notna() & df[r["den"]].notna()]
            v = _ratio(both[r["num"]].sum(), both[r["den"]].sum(), r["mult"])
            excl = n - len(both)
            rows.append([r["label"], _f(v, 4), f"{r['formula']} over {len(both):,} rows" + (f", {excl:,} rows without both values left out" if excl else "")])
            findings[f"overall_{r['name']}"] = v
            if D:
                daily = both.groupby(D)[[r["num"], r["den"]]].sum()
                dr = daily[r["num"]] / daily[r["den"]].where(daily[r["den"]] > 0) * r["mult"]
                undefined = int(dr.isna().sum())
                rows.append([f"Average daily {r['name']}", _f(dr.mean(), 4),
                             f"Mean of {int(dr.notna().sum())} daily values" + (f", {undefined} days undefined (denominator 0)" if undefined else "")])
                findings[f"daily_{r['name']}"] = dr.mean()
                if r["better"] == "higher" and v is not None and not pd.isna(dr.mean()):
                    text.append(f"The overall {r['name']} ({v:,.4g}) weights every day by its {by[r['den']]}. The average of daily "
                                f"{r['name']} ({dr.mean():,.4g}) gives every day equal weight, so days with few {by[r['den']]} "
                                f"count as much as busy days. The two differ when small days have unusually high or low rates.")
        for p in plan["per"]:
            k = df[p["per"]].nunique()
            rows.append([p["label"], _f(df[p["measure"]].sum() / k if k else None), p["formula"]])
        sections.append({"id": "efficiency", "title": "Efficiency", "text": text + [
            "A ratio is computed from totals (ratio of sums), not as an average of row ratios. Where the denominator is 0, "
            "the ratio is undefined and shown as blank, never as 0."],
            "tables": [{"columns": ["Metric", "Value", "How it was calculated"], "rows": rows}]})

    # 3 groups (few values, such as accounts)
    for gname in plan["groups"]:
        if f"group:{gname}" not in plan["sections"]:
            continue
        a = _agg(df, gname, plan)
        for other in plan["items"]:
            a[f"distinct_{other}"] = df.groupby(gname)[other].nunique()
            L[f"distinct_{other}"] = f"Distinct {by[other]}"
        a = a.sort_values(M[0], ascending=False)
        text = []
        hi, lo = a[M[0]].idxmax(), a[M[0]].idxmin()
        text.append(f"Highest total {by[M[0]]}: {hi} ({a.loc[hi, M[0]]:,.2f}). Lowest: {lo} ({a.loc[lo, M[0]]:,.2f}).")
        for r in R:
            s = a[r["name"]].dropna()
            if len(s) > 1:
                best = s.idxmin() if r["better"] == "lower" else s.idxmax()
                worst = s.idxmax() if r["better"] == "lower" else s.idxmin()
                text.append(f"Best {r['name']}: {best} ({_n(s[best])}). Weakest: {worst} ({_n(s[worst])}).")
        if R:
            rank = pd.concat([a[r["name"]].rank(ascending=r["better"] == "lower") for r in R], axis=1).mean(axis=1)
            a["efficiency_rank"] = rank.rank(method="min")
            L["efficiency_rank"] = "Efficiency rank (1 = best)"
            text.append("The efficiency rank averages the rank on each ratio. Without conversion or revenue data, a high "
                        f"{eng['name'] if eng else 'rate'} shows engagement, not business success." if ads else
                        "The rank averages the rank on each ratio in the plan. It does not measure outcomes the data does not hold.")
        findings[f"group_{gname}"] = a
        sections.append({"id": f"group:{gname}", "title": f"Performance by {by[gname]}", "text": text,
                         "tables": [_table(a, L)]})

    # 4 items (many values, such as ads)
    for iname in plan["items"]:
        if f"item:{iname}" not in plan["sections"]:
            continue
        a = _agg(df, iname, plan)
        text, tables = [], []
        top = a.sort_values(M[0], ascending=False)
        tables.append({"title": f"Top {th['top_n']} {by[iname]} by total {by[M[0]]}", **_table(top, L, th["top_n"])})
        if eng and cost:
            el = a[a[eng["den"]] >= th["min_denominator"]].copy()
            text.append(f"{len(el):,} of {len(a):,} {by[iname]} values have at least {th['min_denominator']:,.0f} {by[eng['den']]}, "
                        "the minimum for ranking ratios. Smaller ones are left out of the ratio groups below, because a few "
                        f"{by[eng['den']]} make {eng['name']} unstable.")
            if len(el) >= 8:
                q = lambda col, p: el[col].quantile(p)
                groups = [
                    ("High spend, weak engagement", (el[M[0]] >= q(M[0], .75)) & (el[eng["name"]] <= q(eng["name"], .25)),
                     f"total {by[M[0]]} in the top quarter and {eng['name']} in the bottom quarter"),
                    ("High engagement for the spend", (el[eng["name"]] >= q(eng["name"], .75)) & (el[cost["name"]] <= q(cost["name"], .25)),
                     f"{eng['name']} in the top quarter and {cost['name']} in the bottom quarter"),
                    (f"Many {by[eng['den']]}, few {by[eng['num']]}", (el[eng["den"]] >= q(eng["den"], .75)) & (el[eng["name"]] <= q(eng["name"], .10)),
                     f"{by[eng['den']]} in the top quarter and {eng['name']} in the bottom tenth"),
                ]
                for title, mask, rule in groups:
                    sel = el[mask].sort_values(M[0], ascending=False)
                    findings[f"{iname}:{title}"] = sel
                    share = sel[M[0]].sum() / a[M[0]].sum() if a[M[0]].sum() else 0
                    text.append(f"{title}: {len(sel):,} {by[iname]} values ({share:.1%} of total {by[M[0]]}). Rule: {rule}.")
                    tables.append({"title": f"{title} (top {th['top_n']} by {by[M[0]]})", **_table(sel, L, th["top_n"])})
        if not ads and lead:
            floor = min(th["min_denominator"], float(a[lead["den"]].median() or 0))   # adapts to small counts
            el = a[a[lead["den"]] >= floor].copy()
            text.append(f"{len(el):,} of {len(a):,} {by[iname]} values have at least {floor:,.0f} {by[lead['den']]}, "
                        f"the minimum for ranking {lead['name']}. Smaller ones are left out of the groups below.")
            if len(el) >= 8:
                q = lambda col, p: el[col].quantile(p)
                low = lead["better"] == "lower"
                v, med = el[lead["name"]], el[lead["name"]].median()
                # strictly worse (or better) than the median too, so a ratio that is mostly 0 does not put every row in a group
                weak = ((v >= q(lead["name"], .75)) & (v > med)) if low else ((v <= q(lead["name"], .25)) & (v < med))
                strong = ((v <= q(lead["name"], .25)) & (v < med)) if low else ((v >= q(lead["name"], .75)) & (v > med))
                big = el[M[0]] >= q(M[0], .75)
                worst_q = "top" if low else "bottom"
                best_q = "bottom" if low else "top"
                groups = [(f"High {by[M[0]]}, weak {lead['name']}", big & weak,
                           f"total {by[M[0]]} in the top quarter and {lead['name']} in the {worst_q} quarter and worse than the median ({'lower' if low else 'higher'} is better)"),
                          (f"High {by[M[0]]}, strong {lead['name']}", big & strong,
                           f"total {by[M[0]]} in the top quarter and {lead['name']} in the {best_q} quarter and better than the median")]
                for title, mask, rule in groups:
                    sel = el[mask].sort_values(M[0], ascending=False)
                    findings[f"{iname}:{title.split(',')[1].strip().split()[0]}"] = sel
                    share = sel[M[0]].sum() / a[M[0]].sum() if a[M[0]].sum() else 0
                    text.append(f"{title}: {len(sel):,} {by[iname]} values ({share:.1%} of total {by[M[0]]}). Rule: {rule}.")
                    tables.append({"title": f"{title} (top {th['top_n']} by {by[M[0]]})", **_table(sel, L, th["top_n"])})
        findings[f"item_{iname}"] = a
        sections.append({"id": f"item:{iname}", "title": f"Performance by {by[iname]}", "text": text, "tables": tables})

    # 5 trends
    if D and "trends" in plan["sections"]:
        text, tables = [], []
        freq = {"day": "D", "week": "W-MON", "month": "MS"}
        for gr in plan["grains"]:
            t = df.set_index(D).groupby(pd.Grouper(freq=freq[gr], label="left", closed="left"))
            a = t[M].sum(min_count=1)
            for r in R:
                s = df[df[r["num"]].notna() & df[r["den"]].notna()].set_index(D).groupby(pd.Grouper(freq=freq[gr], label="left", closed="left"))[[r["num"], r["den"]]].sum()
                a[r["name"]] = s[r["num"]] / s[r["den"]].where(s[r["den"]] > 0) * r["mult"]
            for c in plan["items"] + plan["groups"]:
                a[f"active_{c}"] = t[c].nunique()
                L[f"active_{c}"] = f"Active {by[c]}"
            if plan["items"]:
                a[f"{M[0]}_per_active"] = a[M[0]] / a[f"active_{plan['items'][0]}"].where(a[f"active_{plan['items'][0]}"] > 0)
                L[f"{M[0]}_per_active"] = f"{by[M[0]]} per active {by[plan['items'][0]]}"
            a.index = a.index.strftime("%Y-%m-%d")
            a.index.name = gr
            findings[f"trend_{gr}"] = a
            for col in [M[0]] + ([eng["num"]] if eng else []) + ([eng["name"]] if eng else []):
                s = a[col].dropna()
                if len(s) > 1:
                    text.append(f"By {gr}, {L.get(col, col)} was highest on {s.idxmax()} ({_n(s.max())}) and lowest on {s.idxmin()} ({_n(s.min())}).")
            if gr == "day":
                s = a[M[0]].fillna(0)
                mad = (s - s.median()).abs().median()
                if mad > 0:
                    z = 0.6745 * (s - s.median()) / mad
                    spikes = z[z > 3.5].sort_values(ascending=False)
                    findings["spikes"] = spikes
                    text.append(f"{len(spikes)} days had unusually high {by[M[0]]} (robust z-score above 3.5 against the daily median of "
                                f"{s.median():,.2f})" + (f", the largest on {spikes.index[0]} ({s[spikes.index[0]]:,.2f})." if len(spikes) else "."))
                tables.append({"title": "Daily figures (first and last 7 days)", **_table(pd.concat([a.head(7), a.tail(7)]), L)})
            else:
                tables.append({"title": f"By {gr}", **_table(a, L, 60)})
            if gr == "week" and len(a) > 2:
                ch = a[M[0]].pct_change()
                jumps = ch[ch.abs() > 0.5].dropna()
                text.append(f"{len(jumps)} weeks changed total {by[M[0]]} by more than 50% from the week before.")
                if eng:
                    e = a[eng["name"]]
                    half = len(e) // 2
                    if half >= 2 and e.iloc[:half].notna().any() and e.iloc[half:].notna().any():
                        first, second = e.iloc[:half].mean(), e.iloc[half:].mean()
                        findings["eng_trend"] = (first, second)
                        text.append(f"Average weekly {eng['name']} moved from {first:,.4g} in the first half of the period to {second:,.4g} in the second.")
            if gr == ("week" if "week" in plan["grains"] else "month") and lead and not ads and len(a) > 3:
                e = a[lead["name"]]
                half = len(e) // 2
                if half >= 2 and e.iloc[:half].notna().any() and e.iloc[half:].notna().any():
                    first, second = e.iloc[:half].mean(), e.iloc[half:].mean()
                    findings["lead_trend"] = (first, second)
                    worse = (second > first) == (lead["better"] == "lower")
                    text.append(f"Average {gr}ly {lead['name']} moved from {first:,.4g} in the first half of the period to "
                                f"{second:,.4g} in the second, {'a worsening' if worse else 'an improvement'} "
                                f"({lead['better']} is better).")
            if plan["items"] and gr in ("week", "month"):
                c = a[[M[0], f"active_{plan['items'][0]}"]].dropna()
                if len(c) > 3 and c.nunique().min() > 1:          # a constant column has no correlation
                    rho = stats.spearmanr(c[M[0]], c[f"active_{plan['items'][0]}"]).statistic
                    text.append(f"By {gr}, total {by[M[0]]} and the number of active {by[plan['items'][0]]} move together "
                                f"(Spearman {rho:.2f}). Changes in volume partly reflect how many {by[plan['items'][0]]} were running; "
                                f"the column {L[f'{M[0]}_per_active']} separates that from {by[M[0]]} per {by[plan['items'][0]]}.")
        sections.append({"id": "trends", "title": "Trends over time", "text": text, "tables": tables})

    # 6 relationships
    if "relationships" in plan["sections"] and len(M) >= 2:
        text, rows = [], []
        for lvl in plan["items"] + plan["groups"]:
            a = findings.get(f"item_{lvl}") if f"item_{lvl}" in findings else findings.get(f"group_{lvl}")
            if a is None:
                a = _agg(df, lvl, plan)
            if len(a) < 10:
                text.append(f"At {by[lvl]} level there are only {len(a)} values, too few for a reliable correlation. Compare them in the table above instead.")
                continue
            pairs = [(M[0], m) for m in M[1:]] + [(M[0], r["name"]) for r in R]
            for x, y in pairs:
                sub = a[[x, y]].dropna()
                if y in [r["name"] for r in R]:
                    rr = next(r for r in R if r["name"] == y)
                    sub = sub[a.loc[sub.index, rr["den"]] >= th["min_denominator"]]
                if len(sub) < 10 or sub.nunique().min() < 2:
                    continue
                sp = stats.spearmanr(sub[x], sub[y])
                pe = stats.pearsonr(np.log1p(sub[x].clip(lower=0)), np.log1p(sub[y].clip(lower=0)))
                rows.append([by[lvl], L.get(x, x), L.get(y, y), len(sub), _f(sp.statistic, 3), _f(pe.statistic, 3),
                             "below 0.001" if sp.pvalue < 0.001 else _f(sp.pvalue, 3)])
            skew = a[M[0]].skew()
            text.append(f"At {by[lvl]} level, {by[M[0]]} is skewed (skewness {skew:.1f}), so the table uses Spearman rank correlation "
                        "and a Pearson correlation on log values, which are less driven by a few very large values.")
        text.append("Correlation shows that two measures move together across ads or accounts. It does not show that spending "
                    "more causes more clicks: larger budgets often go to ads that were already performing, and the data has no "
                    "conversions to judge value." if ads else
                    f"Correlation shows that two measures move together across {_levels(plan, by)} values. It does not show "
                    "that one causes the other. A third factor, such as size, location or timing, can drive both.")
        sections.append({"id": "relationships", "title": "Spend and engagement" if ads else "Relationships between measures", "text": text,
                         "tables": [{"columns": ["Level", "X", "Y", "Values", "Spearman", "Pearson (log)", "p-value (Spearman)"], "rows": rows}] if rows else []})

    # 7 data quality
    if "quality" in plan["sections"]:
        rows = []
        add = lambda issue, count, severity, kind, note="": rows.append([issue, int(count), severity, kind, note])
        for c in cols:
            if c["role"] == "empty":
                continue
            miss = int(df[c["name"]].isna().sum()) if c["name"] in df else 0
            add(f"Missing {c['label']}", miss, "high" if miss > 0.1 * n else "medium" if miss else "none",
                "Confirmed gap", f"{miss / n:.1%} of rows" if miss else "")
        add("Exact duplicate rows", df.duplicated().sum(), "medium" if df.duplicated().sum() else "none", "Confirmed error")
        key = [k for k in plan["items"] + plan["groups"] + ([D] if D else []) if k in df]
        if key:
            kd = int(df.duplicated(key).sum())
            add(f"Repeated {' / '.join(by.get(k, k) for k in key)} combinations", kd, "medium" if kd else "none",
                "Possible duplicate", "Same keys on more than one row")
        for m in M:
            neg = int((df[m] < 0).sum())
            add(f"Negative {by[m]}", neg, "high" if neg else "none", "Confirmed error")
            zero = int((df[m] == 0).sum())
            add(f"{by[m]} equal to 0", zero, "low" if zero else "none", "Unusual but possibly valid", f"{zero / n:.1%} of rows" if zero else "")
        for r in R:
            if r["mult"] == 100:
                over = int((df[r["num"]] > df[r["den"]]).sum())
                add(f"{by[r['num']]} greater than {by[r['den']]}", over, "high" if over else "none", "Confirmed error",
                    f"makes {r['name']} exceed 100%")
            undefined = int(((df[r["den"]] == 0) & df[r["num"]].notna()).sum())
            add(f"{r['name']} undefined ({by[r['den']]} = 0)", undefined, "low" if undefined else "none", "Expected",
                "left out of the ratio, not counted as 0")
        if D:
            bad = next((c.get("unreadable", 0) for c in cols if c["name"] == D), 0)
            add(f"Unreadable {by[D]}", bad, "high" if bad else "none", "Confirmed error")
            fut = int((df[D] > pd.Timestamp.today()).sum())
            add(f"{by[D]} in the future", fut, "medium" if fut else "none", "Possible error")
            full = pd.date_range(df[D].min(), df[D].max(), freq="D")
            gaps = len(full.difference(pd.DatetimeIndex(df[D].dropna().unique())))
            add("Days with no activity at all", gaps, "low" if gaps else "none", "Unusual but possibly valid",
                f"between {df[D].min().date()} and {df[D].max().date()}")
        for it in plan["items"]:
            for gname in plan["groups"]:
                multi = int((df.groupby(it)[gname].nunique() > 1).sum())
                add(f"{by[it]} linked to more than one {by[gname]}", multi, "medium" if multi else "none",
                    "Possible error if each belongs to one", "")
        for m in M:
            v = df[m].dropna()
            q1, q3 = v.quantile(.25), v.quantile(.75)
            out = int(((v > q3 + 3 * (q3 - q1))).sum()) if q3 > q1 else 0
            add(f"Very high {by[m]} (above Q3 + 3 x IQR)", out, "low" if out else "none", "Unusual but possibly valid",
                f"threshold {q3 + 3 * (q3 - q1):,.2f}" if q3 > q1 else "")
        for it in plan["items"]:
            a = findings.get(f"item_{it}")
            for r in R:
                if a is None:
                    break
                s = a.loc[a[r["den"]] >= th["min_denominator"], r["name"]].dropna()
                if len(s) > 10:
                    q1, q3 = s.quantile(.25), s.quantile(.75)
                    hi = int((s > q3 + 3 * (q3 - q1)).sum()) if q3 > q1 else 0
                    add(f"{by[it]} with very high {r['name']}", hi, "low" if hi else "none", "Unusual but possibly valid", "")
        findings["quality"] = rows
        sections.append({"id": "quality", "title": "Data quality and anomalies", "text": [
            "Nothing was deleted, changed or deduplicated. Confirmed errors break a rule the data must follow. Unusual "
            "observations may be valid and need checking with the data owner."],
            "tables": [{"columns": ["Check", "Rows affected", "Severity", "Type", "Note"], "rows": rows}]})

    # 8 concentration
    if "concentration" in plan["sections"]:
        rows, text = [], []
        conc = [m for m in M if re.search(T.__dict__.get("ADDITIVE", r"spend|cost|amount|clicks|impressions"), m) or True][:3]
        for lvl in plan["items"] + plan["groups"]:
            for m in conc:
                s = df.groupby(lvl)[m].sum().sort_values(ascending=False)
                tot = s.sum()
                if not tot:
                    continue
                share = s / tot
                k10 = max(1, int(np.ceil(len(s) * 0.1)))
                n80 = int((share.cumsum() < 0.8).sum() + 1)
                rows.append([by[lvl], by[m], len(s), f"{share.iloc[0]:.1%}", f"{share.head(min(3, len(s))).sum():.1%}",
                             f"{share.head(k10).sum():.1%}", n80, _f((share ** 2).sum() * 10000, 0)])
                findings[f"conc_{lvl}_{m}"] = (share.iloc[0], share.head(k10).sum(), n80, len(s))
        text.append("HHI is the Herfindahl index of shares (10,000 means one value holds everything). A share held by few "
                    "accounts or ads is an operational risk: a pause, a policy rejection or a tracking fault on one of them "
                    "moves the total. Before moving budget, check conversions, audience overlap and seasonality." if ads else
                    "HHI is the Herfindahl index of shares (10,000 means one value holds everything). A share held by few "
                    f"{_levels(plan, by)} values is a concentration risk: a problem with one of them moves the total.")
        sections.append({"id": "concentration", "title": "Concentration", "text": text, "tables": [{"columns": [
            "Level", "Measure", "Values", "Top 1 share", "Top 3 share", "Top 10% share", "Values making 80%", "HHI"], "rows": rows}]})

    # 9 recommendations
    if "recommendations" in plan["sections"]:
        recs = _recommendations(df, plan, by, findings, totals)
        un = plan.get("unavailable") or []
        missing = (", ".join(un[:-1]) + " and " + un[-1]) if len(un) > 1 else "".join(un)
        sections.append({"id": "recommendations", "title": "Recommendations", "text": [
            "Each recommendation rests only on the figures above. Conversions, revenue, return on spend and acquisition cost "
            "are not in the data, so none is estimated." if ads else "Each recommendation rests only on the figures above."
            + (f" {missing[0].upper() + missing[1:]} {'is' if len(un) == 1 else 'are'} not in the data, so none is estimated." if missing else "")],
            "recommendations": recs})

    # 10 summary
    summary = _summary(df, plan, by, findings, totals, recs)
    lim = ["Human review: every figure comes from the dataset rows only. The analysis cannot see campaign objectives, audiences or creative content."
           if ads else "Human review: every figure comes from the dataset rows only. The analysis cannot see context outside the data, such as targets, policies or market conditions."]
    if plan.get("unavailable"):
        lim.append("Requested but not in the data, so not reported: " + ", ".join(plan["unavailable"]) + ".")
    lim.append("Ratios with missing numerator or denominator values leave those rows out. The efficiency table states how many.")
    sections.append({"id": "summary", "title": "Executive summary", "text": summary, "limitations": lim})
    return {"sections": sections, "totals": totals, "rows": n}


def _generic_recommendations(df, plan, by, f, totals):
    """Recommendations for any dataset. They name the evidence and never assume what the business is."""
    M, R = plan["measures"], plan["ratios"]
    lead = R[0] if R else None
    out = []
    more = lambda idx: ", ".join(str(x) for x in idx[:5]) + (" and more" if len(idx) > 5 else "")
    for it in plan["items"]:
        weak = f.get(f"{it}:weak")
        if lead is not None and weak is not None and len(weak):
            share = weak[M[0]].sum() / (totals[M[0]] or 1)
            out.append({"title": f"Review the {len(weak)} {by[it]} values with high {by[M[0]]} and weak {lead['name']}",
                        "evidence": f"They hold {share:.1%} of total {by[M[0]]}, and their {lead['name']} is in the weakest quarter.",
                        "affected": more(weak.index),
                        "action": f"Check each one against the {by[it]} values with strong {lead['name']}. Confirm whether the gap is real or a data issue.",
                        "impact": f"{'Lower' if lead['better'] == 'lower' else 'Higher'} {lead['name']} if the cause is fixed.",
                        "confidence": "Medium. A ratio shows where to look, not why.",
                        "data_needed": "The history and context of these records."})
    spread = []                     # only the grouping where the ratio differs most, so the list does not repeat itself
    for g in plan["groups"]:
        a = f.get(f"group_{g}")
        if lead is not None and a is not None and a[lead["name"]].notna().sum() > 1:
            v = a[lead["name"]].dropna()
            spread.append(((v.max() - v.min()) / (abs(v).mean() or 1), g))
    for _, g in sorted(spread, reverse=True)[:1]:
        a = f.get(f"group_{g}")
        if True:
            s_ = a[lead["name"]].dropna()
            worst = s_.idxmax() if lead["better"] == "lower" else s_.idxmin()
            best = s_.idxmin() if lead["better"] == "lower" else s_.idxmax()
            out.append({"title": f"Investigate why {lead['name']} differs between {by[g]} values",
                        "evidence": f"{worst} has {lead['name']} {s_[worst]:,.4g} against {s_[best]:,.4g} for {best} ({lead['better']} is better).",
                        "affected": f"{by[g]} {worst}",
                        "action": f"Compare {worst} with {best}: mix, size and timing of the records behind the ratio.",
                        "impact": f"{lead['name']} for {worst} closer to {best}.",
                        "confidence": "Medium to low. Some differences are intended.",
                        "data_needed": f"Targets or policies for each {by[g]}."})
    if f.get("lead_trend") and lead is not None:
        a, b = f["lead_trend"]
        if (b > a) == (lead["better"] == "lower"):
            out.append({"title": f"Find the cause of the change in {lead['name']} over time",
                        "evidence": f"Average {lead['name']} moved from {a:,.4g} to {b:,.4g} between the two halves of the period.",
                        "affected": "The second half of the period",
                        "action": "Check which groups drive the change, using the trend and group tables.",
                        "impact": f"{lead['name']} back toward {a:,.4g}.",
                        "confidence": "Medium. The change is measured. Its cause is not.",
                        "data_needed": "Events in the period, such as policy or product changes."})
    q = f.get("quality") or []
    worst = sorted([r for r in q if r[2] in ("high", "medium") and r[1] > 0], key=lambda r: (r[2] != "high", -r[1]))
    if worst:
        r = worst[0]
        out.append({"title": f"Fix the data issue: {r[0].lower()}",
                    "evidence": f"{r[1]:,} rows affected" + (f" ({r[4]})." if r[4] else "."),
                    "affected": "Every figure that uses this field",
                    "action": "Ask the data owner what causes it and fix it at the source. Nothing was changed here.",
                    "impact": "More reliable figures.", "confidence": "High that the issue exists. Its cause is unknown.",
                    "data_needed": "The source system's rules for this field."})
    for lvl in plan["items"] + plan["groups"]:
        c = f.get(f"conc_{lvl}_{M[0]}")
        if c and (c[0] > 0.4 or (lvl in plan["items"] and c[1] > 0.5)):
            out.append({"title": f"Watch the concentration of {by[M[0]]} in a few {by[lvl]} values",
                        "evidence": f"The largest {by[lvl]} holds {c[0]:.0%} of total {by[M[0]]}. {c[2]} of {c[3]} values make 80%.",
                        "affected": f"The largest {by[lvl]} values",
                        "action": "Set limits or closer monitoring for the largest values.",
                        "impact": "Lower concentration risk.", "confidence": "Medium.",
                        "data_needed": "Agreed limits for each value."})
            break
    return out[:5]


def _recommendations(df, plan, by, f, totals):
    if not _is_ads(plan):
        return _generic_recommendations(df, plan, by, f, totals)
    M, R = plan["measures"], plan["ratios"]
    eng = next((r for r in R if r["better"] == "higher"), None)
    cost = next((r for r in R if r["better"] == "lower"), None)
    out = []
    for it in plan["items"]:
        weak = f.get(f"{it}:High spend, weak engagement")
        if weak is not None and len(weak) and eng:
            share = weak[M[0]].sum() / (totals[M[0]] or 1)
            out.append({"title": f"Review the {len(weak)} {by[it]} values with high spend and weak {eng['name']}",
                        "evidence": f"They take {share:.1%} of total {by[M[0]]} with {eng['name']} in the bottom quarter of eligible {by[it]}.",
                        "affected": ", ".join(str(x) for x in weak.index[:5]) + (" and more" if len(weak) > 5 else ""),
                        "action": "Test new creative or targeting, or cap their budgets, and compare against the strong group.",
                        "impact": f"Lower {by[M[0]]}, higher {eng['name']}, lower {cost['name'] if cost else 'cost per result'}.",
                        "confidence": "Medium. Engagement is not value: these may still convert.",
                        "data_needed": "Conversions or revenue per ad, and the campaign objective."})
        strong = f.get(f"{it}:High engagement for the spend")
        if strong is not None and len(strong) and eng:
            out.append({"title": f"Test more budget on the {len(strong)} efficient {by[it]} values",
                        "evidence": f"{eng['name']} in the top quarter and {cost['name'] if cost else 'cost'} in the bottom quarter.",
                        "affected": ", ".join(str(x) for x in strong.index[:5]) + (" and more" if len(strong) > 5 else ""),
                        "action": "Raise budgets in small steps and watch whether efficiency holds at higher spend.",
                        "impact": f"More {by[eng['num']]} at similar or lower {cost['name'] if cost else 'cost'}.",
                        "confidence": "Medium. Efficiency often falls as budgets grow.",
                        "data_needed": "Conversions, and spend limits or audience size for these ads."})
        few = next((v for k, v in f.items() if k.startswith(f"{it}:Many")), None)
        if few is not None and len(few) and eng:
            out.append({"title": f"Refresh the {len(few)} {by[it]} values with many {by[eng['den']]} but few {by[eng['num']]}",
                        "evidence": f"{by[eng['den']]} in the top quarter, {eng['name']} in the bottom tenth of eligible {by[it]}.",
                        "affected": ", ".join(str(x) for x in few.index[:5]) + (" and more" if len(few) > 5 else ""),
                        "action": "Check placement, creative fatigue and audience fit.",
                        "impact": f"Higher {eng['name']}; {by[M[0]]} unchanged or lower.",
                        "confidence": "Medium.", "data_needed": "Placement and frequency data."})
    for gname in plan["groups"]:
        a = f.get(f"group_{gname}")
        if a is not None and cost and a[cost["name"]].notna().sum() > 1:
            s = a[cost["name"]].dropna()
            out.append({"title": f"Investigate why {cost['name']} differs between {by[gname]} values",
                        "evidence": f"{s.idxmax()} pays {s.max():,.4g} against {s.min():,.4g} for {s.idxmin()}.",
                        "affected": f"{by[gname]} {s.idxmax()}",
                        "action": "Compare bidding, audiences and placements of the costliest account with the cheapest.",
                        "impact": f"Lower {cost['name']} for {s.idxmax()}.",
                        "confidence": "Medium to low. Accounts may target different markets on purpose.",
                        "data_needed": "Campaign objectives and markets per account."})
    q = f.get("quality") or []
    worst = sorted([r for r in q if r[2] == "high" and r[1] > 0], key=lambda r: -r[1])
    if worst:
        r = worst[0]
        out.append({"title": f"Fix the data gap: {r[0].lower()}",
                    "evidence": f"{r[1]:,} rows affected ({r[4] or 'see data quality'}).",
                    "affected": "All reports that use this field",
                    "action": "Ask the data owner whether blanks mean 0 or not recorded, and fix the export.",
                    "impact": "More reliable ratios. No direct effect on spend.",
                    "confidence": "High that the gap exists. Its cause is unknown.",
                    "data_needed": "The tracking or export specification."})
    for lvl in plan["items"]:
        c = f.get(f"conc_{lvl}_{plan['measures'][0]}")
        if c and c[1] > 0.5:
            out.append({"title": f"Reduce dependence on the top {by[lvl]} values",
                        "evidence": f"The top 10% of {by[lvl]} hold {c[1]:.0%} of total {by[plan['measures'][0]]}; {c[2]} of {c[3]} make 80%.",
                        "affected": f"The top 10% of {by[lvl]}",
                        "action": "Keep a tested pipeline of new ads so a pause on one large ad does not move the total.",
                        "impact": "Spread of spend; lower risk, not lower cost.",
                        "confidence": "Medium.", "data_needed": "Planned budgets and ad lifecycles."})
    return out[:5]


def _summary(df, plan, by, f, totals, recs):
    M, R = plan["measures"], plan["ratios"]
    out = [f"{len(df):,} rows analysed with no filters. Totals: " + ", ".join(f"{by[m]} {totals[m]:,.2f}" for m in M if totals[m] is not None) + "."]
    for r in R:
        v = f.get(f"overall_{r['name']}")
        if v is not None:
            out.append(f"Overall {r['name']} {v:,.4g}" + (f", average daily {r['name']} {f.get('daily_' + r['name']):,.4g}." if f.get("daily_" + r["name"]) is not None else "."))
    for g in plan["groups"]:
        a = f.get(f"group_{g}")
        if a is not None and "efficiency_rank" in a:
            word = ("Most efficient", "Least efficient") if _is_ads(plan) else ("Best ranked", "Lowest ranked")
            rk = a["efficiency_rank"]
            if rk.min() == rk.max():
                out.append(f"{by[g]} values rank level on the combined ratios. See the {by[g]} table for each ratio.")
            else:
                best, worst = rk[rk == rk.min()].index, rk[rk == rk.max()].index
                out.append(f"{word[0]} {by[g]}: {', '.join(map(str, best))}. {word[1]}: {', '.join(map(str, worst))}.")
    if f.get("eng_trend"):
        a, b = f["eng_trend"]
        out.append(f"Engagement trend: {'down' if b < a else 'up'} from {a:,.4g} to {b:,.4g} between the two halves of the period.")
    if f.get("lead_trend") and R:
        a, b = f["lead_trend"]
        out.append(f"{R[0]['name']} trend: {'down' if b < a else 'up'} from {a:,.4g} to {b:,.4g} between the two halves of the period "
                   f"({R[0]['better']} is better).")
    q = [r for r in (f.get("quality") or []) if r[2] in ("high", "medium") and r[1] > 0]
    if q:
        out.append("Material data issues: " + "; ".join(f"{r[0]} ({r[1]:,} rows)" for r in q[:4]) + ".")
    if recs:
        out.append("Next steps: " + "; ".join(r["title"] for r in recs[:3]) + ".")
    return out


# ═════════════════════════ tools (through the L7 guard) ═════════════════════════

@T.tool("D19", "Uploaded dataset extract", "L5.1", "read", policy="uploads",
        precheck=lambda ctx, dataset_id, **_: DS._query_precheck(ctx, dataset_id, {}))
def dataset_extract(ctx, dataset_id):
    """Every row of a person's uploaded dataset, for the full analysis. Personal columns stay as tokens."""
    r = DS._row(dataset_id)
    cols = ", ".join(f'"{c["name"]}"' for c in r["columns"])
    return pd.DataFrame(db.query(f"SELECT {cols} FROM {r['table_name']} ORDER BY {DS.order_by(r)}"))


@T.tool("C13", "Dataset analysis report", "L5.4", "compute", policy="uploads")
def dataset_report(ctx, frame, plan, columns, profile):
    """Computes the approved report plan from the extracted rows. Every figure comes with its formula."""
    return compute(frame, plan, columns, profile)


def distinct_counts(dataset_id):
    r = DS._row(dataset_id)
    return {c["name"]: db.one(f'SELECT COUNT(DISTINCT "{c["name"]}") n FROM {r["table_name"]}')["n"]
            for c in r["columns"] if c["role"] in ("identifier", "category", "date")}
