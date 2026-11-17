"""Uploaded datasets: a person's own spreadsheet, governed like the bank's data.

L1 stores each upload in its own table, with a profile and a fingerprint of the file.
Personal columns (names, phone numbers, emails, BVN, account numbers) are replaced by
tokens before anything is written, so raw personal data never reaches the database.
L5 tools interpret a question against the dataset's columns and run a governed query.
Agents reach the data only through these tools, so the L7 guard checks every call:
only the person who uploaded a dataset can analyse it.
"""
import hashlib
import io
import json
import re
import uuid
from datetime import date

import numpy as np
import pandas as pd

from . import l1_data as db
from . import l2_policy as P
from . import l5_tools as T
from . import llm

TABLE = """
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY, name TEXT NOT NULL, file_name TEXT, owner_email TEXT NOT NULL,
  owner_role TEXT, uploaded_at TEXT NOT NULL, row_count INTEGER, columns JSON, profile JSON,
  table_name TEXT NOT NULL, sha256 TEXT, status TEXT NOT NULL DEFAULT 'ready');
CREATE INDEX IF NOT EXISTS ix_datasets_owner ON datasets (owner_email, status);
"""

ADDITIVE = r"amount|balance|value|revenue|sales|income|cost|price|qty|quantity|volume|total|deposit|withdrawal|" \
           r"inflow|outflow|fee|profit|loss|spend|disburs|outstanding|principal|count|units|premium|claim"
PERSONAL_NAMES = r"(^|_)(first|last|middle|full|sur|customer|client|account|holder|applicant|borrower|member|" \
                 r"beneficiary|contact|guarantor)?_?(name|names|surname)$"
PERSONAL_KEYS = r"e_?mail|phone|mobile|msisdn|telephone|tel_no|bvn|\bnin\b|^nin|passport|address|" \
                r"date_of_birth|^dob$|birth_date|account_?(no|num|number)|acct_?(no|num)|nuban|card_?(no|number)|" \
                r"national_id|tax_id|\btin\b|ssn"
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
PHONE = re.compile(r"^(\+?234|0)[789][01]\d{8}$")
MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mn": 1e6, "bn": 1e9, "billion": 1e9, "b": 1e9}
NEG = T.NEG.replace(r"\b(?:not|", r"\b(?:not|leave out|leaving out|left out|leave|but not|minus|drop|remove|") 
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def init():
    db.script(TABLE)


def policy():
    return P.body("uploads")


def max_file_mb():
    """The bank's limit, and at most 4 MB on Vercel, which refuses request bodies over 4.5 MB."""
    from . import config
    limit = float(policy()["max_file_mb"])
    return min(limit, 4.0) if config.serverless() else limit


# ═════════════════════════ reading and storing a file ═════════════════════════

def _read(file_name: str, data: bytes) -> pd.DataFrame:
    ext = (file_name or "").lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xlsm"):
        try:
            return pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=object)
        except ImportError:
            raise ValueError("Reading Excel files needs the openpyxl package. Run: pip install openpyxl")
    if ext == "xls":
        raise ValueError("Save the file as .xlsx or .csv first. The old .xls format is not supported.")
    if ext not in ("csv", "txt", "tsv"):
        raise ValueError("Upload a .csv or .xlsx file.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str, encoding=enc,
                               skipinitialspace=True)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as e:
            raise ValueError(f"The file could not be read as a table: {str(e).splitlines()[0]}")
    raise ValueError("The file's text encoding could not be read. Save it as UTF-8 CSV.")


def _snake(name: str, used: set) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")[:48] or "column"
    if s[0].isdigit():
        s = "c_" + s
    base, i = s, 2
    while s in used or s in ("_row",):
        s = f"{base}_{i}"
        i += 1
    used.add(s)
    return s


def _num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)                 # (1,200) means -1200
    s = s.str.replace(r"[₦$€£%,\s]|NGN|USD", "", regex=True, flags=re.I)
    return pd.to_numeric(s, errors="coerce")


def _date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    vals = series.map(lambda v: v if isinstance(v, (pd.Timestamp, date)) else (str(v).strip() if v is not None else None))
    known = vals.notna().sum() or 1
    iso = pd.to_datetime(vals, errors="coerce", format="ISO8601")
    if iso.notna().sum() / known >= 0.9:
        return iso                                    # 2025-03-04 is always year, month, day
    return pd.to_datetime(vals, errors="coerce", format="mixed", dayfirst=True)   # 04/03/2025 is 4 March in Nigeria


def _token(value, salt: str) -> str:
    return "P-" + hashlib.sha256((salt + str(value).strip().lower()).encode()).hexdigest()[:8]


def _infer(col: str, raw: pd.Series, n: int) -> dict:
    """Decide a column's type (number, date, text) and role (measure, category, identifier, personal)."""
    present = raw[raw.notna() & (raw.astype(str).str.strip() != "")]
    k = len(present)
    info = {"type": "text", "role": "category"}
    lname = col.lower()
    if k == 0:
        return {"type": "text", "role": "empty"}
    sample = present.astype(str).str.strip()
    personal = bool(re.search(PERSONAL_KEYS, lname) or re.search(PERSONAL_NAMES, lname)
                    or sample.map(lambda v: bool(EMAIL.match(v))).mean() > 0.5
                    or sample.str.replace(r"[\s-]", "", regex=True).map(lambda v: bool(PHONE.match(v))).mean() > 0.5)
    if re.search(r"(^|_)(branch|product|bank|company|business|region|state|city|merchant|channel|segment|officer)_?name$", lname):
        personal = False if not re.search(r"officer", lname) else personal
    if personal:
        return {"type": "text", "role": "personal"}
    nums = _num(present)
    if nums.notna().mean() >= 0.95:
        uniq = present.nunique()
        ident = bool(re.search(r"(^|_)(id|no|num|number|code|ref|reference|key)$", lname)) or \
            (uniq == k and k > 20 and (nums.dropna() % 1 == 0).all() and not re.search(ADDITIVE, lname))
        if ident:   # IDs stay text: long numbers such as 120216208152400008 lose digits as floating point
            return {"type": "text", "role": "identifier"}
        return {"type": "number", "role": "measure"}
    if not re.fullmatch(r"[\d.]+", sample.iloc[0]):
        dts = _date(present)
        if dts.notna().mean() >= 0.9:
            return {"type": "date", "role": "date"}
    uniq = present.nunique()
    if uniq > 0.95 * k and k > 50:
        info["role"] = "identifier"
    return info


def create(file_name: str, data: bytes, who: dict, name: str = None) -> dict:
    pol = policy()
    if not pol.get("enabled", True):
        raise PermissionError("Uploading data is turned off in the bank's policy.")
    if who["role"] not in pol.get("roles", []):
        raise PermissionError("Your role is not allowed to upload data under the bank's policy.")
    if len(data) > max_file_mb() * 1024 * 1024:
        raise ValueError(f"The file is larger than {max_file_mb():g} MB, the limit here.")
    if not data:
        raise ValueError("The file is empty.")
    df = _read(file_name, data)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, [not str(c).startswith("Unnamed:") or df[c].notna().any() for c in df.columns]]
    if df.empty:
        raise ValueError("The file has no rows of data.")
    if len(df) > pol["max_rows"]:
        raise ValueError(f"The file has {len(df):,} rows. The bank's limit is {pol['max_rows']:,}.")
    if df.shape[1] > pol["max_columns"]:
        raise ValueError(f"The file has {df.shape[1]} columns. The bank's limit is {pol['max_columns']}.")
    did = "DS-" + uuid.uuid4().hex[:8]
    table = "uds_" + did[3:].lower()
    used, cols, out = set(), [], {}
    salt = did
    n = len(df)
    for c in df.columns:
        label = str(c).strip() or "Column"
        name_ = _snake(label, used)
        raw = df[c].where(df[c].notna(), None)
        inf = _infer(name_, raw, n)
        meta = {"name": name_, "label": label, **inf}
        if inf["type"] == "number":
            vals = _num(raw)
            meta["unreadable"] = int((raw.notna() & (raw.astype(str).str.strip() != "") & vals.isna()).sum())
            out[name_] = vals
        elif inf["type"] == "date":
            vals = _date(raw)
            meta["unreadable"] = int((raw.notna() & vals.isna()).sum())
            out[name_] = vals.dt.date
        elif inf["role"] == "personal" and pol.get("mask_personal_columns", True):
            out[name_] = raw.map(lambda v: _token(v, salt) if v is not None and str(v).strip() else None)
            meta["masked"] = True
        else:
            out[name_] = raw.map(lambda v: str(v).strip() if v is not None and str(v).strip() else None)
        cols.append(meta)
    frame = pd.DataFrame(out)
    ddl = ", ".join(f'"{m["name"]}" ' + {"number": "DOUBLE PRECISION", "date": "DATE"}.get(m["type"], "TEXT") for m in cols)
    db.script(f'CREATE TABLE {table} (_row INTEGER, {ddl})')
    names = ["_row"] + [f'"{m["name"]}"' for m in cols]

    def rows():
        for i, rec in enumerate(frame.itertuples(index=False), start=1):
            yield [i] + [None if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT) else
                         (float(v) if isinstance(v, (np.floating, np.integer)) else v) for v in rec]
    db.copy_rows(table, names, rows())
    profile = _profile(frame, cols)
    sha = hashlib.sha256(data).hexdigest()
    db.execute("INSERT INTO datasets (dataset_id, name, file_name, owner_email, owner_role, uploaded_at, row_count, columns, "
               "profile, table_name, sha256, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (did, (name or "").strip() or re.sub(r"\.[a-z]+$", "", file_name or "Upload", flags=re.I), file_name,
                who["email"].lower(), who["role"], db.now(), n, json.dumps(cols), json.dumps(profile, default=str), table, sha, "ready"))
    db.ledger("dataset_uploaded", actor=f"human:{who['role']}:{who['email']}", tool_id="D18", action_class="internal_write",
              policy_version=P.versions().get("uploads"),
              payload={"dataset_id": did, "file": file_name, "rows": n, "columns": len(cols), "sha256": sha,
                       "masked_columns": [m["label"] for m in cols if m.get("masked")],
                       "quality_score": profile["score"]})
    return get(did, who)


# ═════════════════════════ profile and data quality (L5.1 D18) ═════════════════════════

def _profile(frame: pd.DataFrame, cols: list) -> dict:
    n = len(frame)
    issues, out = [], []
    today = pd.Timestamp.today().date()
    dup = int(frame.duplicated().sum())
    if dup:
        issues.append({"column": "(all)", "issue": f"{dup:,} rows are exact duplicates of another row", "severity": "warning"})
    for m in cols:
        s = frame[m["name"]]
        miss = int(s.isna().sum())
        p = {"name": m["name"], "label": m["label"], "type": m["type"], "role": m["role"], "missing": miss,
             "missing_pct": round(100 * miss / n, 1) if n else 0, "distinct": int(s.nunique(dropna=True))}
        if m["role"] == "empty":
            issues.append({"column": m["label"], "issue": "The column has no values", "severity": "info"})
        if miss and m["role"] != "empty":
            sev = "warning" if miss / n > 0.2 else "info"
            issues.append({"column": m["label"], "issue": f"{miss:,} missing values ({p['missing_pct']}%)", "severity": sev})
        if m.get("unreadable"):
            issues.append({"column": m["label"], "issue": f"{m['unreadable']:,} values could not be read as "
                           f"{'numbers' if m['type'] == 'number' else 'dates'} and are treated as missing", "severity": "warning"})
        if m["type"] == "number":
            v = s.dropna()
            if len(v):
                q1, q3 = v.quantile(.25), v.quantile(.75)
                iqr = q3 - q1
                outl = int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
                p.update(min=float(v.min()), max=float(v.max()), mean=round(float(v.mean()), 2),
                         median=float(v.median()), sum=float(v.sum()), negatives=int((v < 0).sum()), outliers=outl)
                if p["negatives"] and re.search(ADDITIVE, m["name"]) and m["role"] == "measure" and \
                        not re.search(r"profit|loss|margin|change|variance|growth|diff|net|balance|return|gain", m["name"]):
                    issues.append({"column": m["label"], "issue": f"{p['negatives']:,} negative values in a column that "
                                   "looks like an amount", "severity": "warning"})
                if outl:
                    issues.append({"column": m["label"], "issue": f"{outl:,} extreme values far outside the usual range",
                                   "severity": "info"})
        elif m["type"] == "date":
            v = s.dropna()
            if len(v):
                p.update(min=str(min(v)), max=str(max(v)))
                fut = int(sum(1 for d in v if d > today))
                if fut:
                    issues.append({"column": m["label"], "issue": f"{fut:,} dates are in the future", "severity": "warning"})
        else:
            top = s.value_counts().head(6)
            p["top"] = [{"value": str(k), "count": int(c)} for k, c in top.items()] if m["role"] != "personal" else []
            if m["role"] == "personal":
                issues.append({"column": m["label"], "issue": "Personal data. Values were replaced by tokens before storage "
                               "and cannot be shown or used to break results down", "severity": "info"})
        if p["distinct"] == 1 and n > 1:
            issues.append({"column": m["label"], "issue": "Every row has the same value", "severity": "info"})
        out.append(p)
    weight = {"warning": 0.06, "info": 0.01}
    score = max(0.0, round(1 - sum(weight.get(i["severity"], 0) for i in issues), 2))
    return {"rows": n, "columns": out, "issues": issues, "duplicates": dup, "score": score}


# ═════════════════════════ access ═════════════════════════

def _row(dataset_id):
    r = db.one("SELECT * FROM datasets WHERE dataset_id=? AND status='ready'", (dataset_id,))
    if not r:
        raise ValueError("This dataset was not found. It may have been deleted.")
    r["columns"] = json.loads(r["columns"])
    r["profile"] = json.loads(r["profile"])
    return r


def order_by(r):
    """Uploads keep the file's row order. Bank data views are ordered by their first column (the record ID)."""
    return "1" if r["owner_email"] == "bank" else "_row"


def access_problem(dataset_id, email):
    try:
        r = _row(dataset_id)
    except ValueError as e:
        return str(e)
    if r["owner_email"] == "bank":                 # the bank's own data: access by role, set in the uploads policy
        from . import bank_data as BK
        u = db.one("SELECT role FROM app_users WHERE lower(email)=? AND active", ((email or "").lower(),))
        return None if u and BK.allowed(dataset_id, u["role"]) else f"Your role cannot use the bank's {r['name']} data."
    if not email or r["owner_email"] != email.lower():
        return "Only the person who uploaded this dataset can analyse it."
    return None


def get(dataset_id, who, preview=True) -> dict:
    p = access_problem(dataset_id, who["email"])
    if p:
        raise PermissionError(p)
    r = _row(dataset_id)
    out = {k: r[k] for k in ("dataset_id", "name", "file_name", "uploaded_at", "row_count", "columns", "profile", "sha256")}
    if preview:
        cols = ", ".join(f'"{c["name"]}"' for c in r["columns"])
        out["preview"] = db.query(f'SELECT {cols} FROM {r["table_name"]} ORDER BY {order_by(r)} LIMIT 15')
        out["suggestions"] = suggestions(r["columns"])
        out["values"] = category_values(r)
    return out


def mine(who) -> list:
    rows = db.query("SELECT dataset_id, name, file_name, uploaded_at, row_count, profile FROM datasets "
                    "WHERE owner_email=? AND status='ready' ORDER BY uploaded_at DESC", (who["email"].lower(),))
    for r in rows:
        p = json.loads(r.pop("profile"))
        r["columns"] = len(p["columns"])
        r["score"] = p["score"]
        r["warnings"] = sum(1 for i in p["issues"] if i["severity"] == "warning")
    return rows


def delete(dataset_id, who):
    if str(dataset_id).startswith("BANK-"):
        raise PermissionError("The bank's own data cannot be deleted from here.")
    p = access_problem(dataset_id, who["email"])
    if p:
        raise PermissionError(p.replace("analyse", "delete"))
    r = _row(dataset_id)
    db.script(f"DROP TABLE IF EXISTS {r['table_name']}")
    db.execute("UPDATE datasets SET status='deleted' WHERE dataset_id=?", (dataset_id,))
    # questions still waiting on this dataset cannot run any more, so their open decisions are withdrawn
    waiting = db.query("SELECT run_id FROM runs WHERE use_case='upload' AND status NOT IN ('completed','failed','stopped','cancelled') "
                       "AND (state::json->'params'->>'dataset_id')=?", (str(dataset_id),))
    for w in waiting:
        db.execute("UPDATE checkpoints SET status='superseded' WHERE run_id=? AND status='open'", (w["run_id"],))
        db.execute("UPDATE runs SET status='cancelled' WHERE run_id=?", (w["run_id"],))
    db.ledger("dataset_deleted", actor=f"human:{who['role']}:{who['email']}", payload={"dataset_id": dataset_id,
                                                                                         "sha256": r["sha256"]})
    return {"ok": True}


def category_values(r, limit=200) -> dict:
    out = {}
    for c in r["columns"]:
        if c["role"] == "category":
            vals = db.query(f'SELECT "{c["name"]}" v, COUNT(*) n FROM {r["table_name"]} WHERE "{c["name"]}" IS NOT NULL '
                            f'GROUP BY 1 ORDER BY 2 DESC LIMIT {int(limit)}')
            out[c["name"]] = [str(v["v"]) for v in vals]
    return out


def suggestions(cols) -> list:
    meas = sorted([c for c in cols if c["role"] == "measure"], key=lambda c: not re.search(ADDITIVE, c["name"]))
    cats = [c for c in cols if c["role"] == "category"]
    dates = [c for c in cols if c["role"] == "date"]
    out = []
    if meas and cats:
        out.append(f"Total {meas[0]['label']} by {cats[0]['label']}")
    if meas and dates:
        out.append(f"{meas[0]['label']} by month")
    if len(meas) > 1 and cats:
        out.append(f"Average {meas[1]['label']} by {cats[min(1, len(cats) - 1)]['label']}")
    if cats:
        out.append(f"How many rows by {cats[0]['label']}")
    if meas and cats:
        out.append(f"Top 5 {cats[0]['label']} by total {meas[0]['label']}")
    return out[:4] + ["Create a dashboard"]


# ═════════════════════════ the analytical plan ═════════════════════════
#
# A question about a dataset becomes one structured, versioned plan. Every later step reads
# this plan and nothing else: the SQL is compiled from it, checked against it, and the
# person who asked confirms it by version. A correction produces a new version.
#
#   metrics        [{"agg": sum|avg|median|min|max|count|count_distinct, "column": name|None, "alias": text}]
#   dimensions     ["category", "order_date:month", ...]   breakdowns, at most three
#   filters        {column: value | [values] | {"not": value|[values]} | {"gt"|"gte"|"lt"|"lte"|"eq": number}}
#   period         None or {"column", "from", "to", "label"}
#   sort           None or {"by": metric alias, "direction": "asc"|"desc"}
#   limit          None or 1 to 100
#   grain          the columns that identify one output row (the dimensions)
#   output_format  "table" | "chart" | "auto"
#   explain        True when the person asked for an explanation of the figures
#   version        1, 2, 3 ... increases with every change
#   notes          what the interpreter decided and why, shown to the person

AGGS = ("sum", "avg", "median", "min", "max", "count", "count_distinct")
AGG_WORDS = {"sum": "Total", "avg": "Average", "count": "Number of rows", "count_distinct": "Number of distinct",
             "min": "Smallest", "max": "Largest", "median": "Median"}
ALIAS_WORDS = {"sum": "total", "avg": "average", "median": "median", "min": "min", "max": "max",
               "count": "row_count", "count_distinct": "distinct"}
OP_WORDS = {"gt": "above", "gte": "at least", "lt": "below", "lte": "at most", "eq": "exactly"}
SQL_OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
NON_ADDITIVE = r"margin|rate|ratio|percent|pct|share|score|(^|_)age($|_)|avg|average|mean|index|yield|price|tenor|days"
AGG_KEYS = [(r"\b(?:sum|total|totals|sum of)\b", "sum"), (r"\b(?:average|avg|mean)\b", "avg"), (r"\bmedian\b", "median"),
            (r"\b(?:minimum|min|smallest|lowest)\b", "min"), (r"\b(?:maximum|max|largest|biggest|highest)\b", "max")]
FUNC = re.compile(r"\b(sum|avg|average|mean|median|min|max|count)\s*\(\s*(distinct\s+)?([a-z0-9_ %()]+?)\s*\)")
OUTPUT_PHRASES = re.compile(r"\b(?:in|as|into|with|on|using)\s+(?:a|an|the|one)\s+(?:table|chart|graph|plot|list|summary|report|"
                            r"spreadsheet|grid|view|row|column)s?\b|\ba (?:table|chart|graph) (?:with|of|showing)\b|"
                            r"\b(?:the )?(?:results|figures|numbers|output)\b")
REPLACE = re.compile(r"\b(?:discard|ignore|forget|scrap|throw away|drop) (?:the |your |my )?(?:previous |old |earlier |whole |entire |current )*"
                     r"(?:query |analysis )?(?:plan|query|interpretation|analysis|version|request)|\bstart (?:again|over|afresh|from scratch)\b|"
                     r"\bfrom scratch\b|\breplace (?:the |your )?(?:whole |entire |previous )?(?:plan|query|interpretation)\b|"
                     r"\b(?:new|fresh) (?:plan|query)\b|\binstead of (?:the|your) previous\b")
NOT_GROUP = re.compile(r"\b(?:do not|don't|dont|never|not|without|no longer|stop|avoid)\s+(?:group(?:ing|ed)?|break(?:ing)?\s+(?:it\s+|this\s+|them\s+)?down|"
                       r"split(?:ting)?|breakdown|segment(?:ing)?)\s+(?:the results\s+|it\s+|them\s+)?(?:by|on|per)\s+(.+?)(?=[,.;:]|\s+(?:and|but|then)\s|$)|"
                       r"\b(?:remove|drop|exclude|take out)\s+(.+?)\s+from\s+the\s+(?:grouping|breakdown|group by)\b|\bnot by\s+(.+?)(?=[,.;:]|\s+(?:and|but)\s|$)")
BY = re.compile(r"\b(?:group(?:ed)?(?: the results| it| them)? by|by|per|across|for each|for every|each|split by|broken down by|"
                r"one row (?:per|for each|for every|for)|a row (?:per|for each|for every|for))\s+(.+?)"
                r"(?=\s+(?:where|for|in|with|except|excluding|without|not|top|last|during|between|from|since|above|below|over|under|"
                r"more|less|greater|using|and then|then|only|show|return|explain)\b|[,.;:?]|\s*$)")


def _plain(q):
    return " " + re.sub(r"\s+", " ", str(q or "").lower()).strip() + " "


def _phrases(c):
    base = {c["name"].replace("_", " "), c["label"].lower().strip(), c["name"]}
    out = set()
    for b in base:
        b = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_ ]", " ", b)).strip()
        if not b:
            continue
        out.add(b)
        out.add(b + "s" if not b.endswith("s") else b[:-1])
        if re.search(r"(ch|sh|s|x|z)$", b):
            out.add(b + "es")
        if b.endswith("y"):
            out.add(b[:-1] + "ies")
        words = b.split()
        if len(words) == 2 and len(words[0]) <= 3 and len(words[-1]) >= 4 and words[-1] not in ("name", "type", "code", "value", "number", "month", "months", "days",
                                                                    "weeks", "years", "year", "week", "hours", "minutes", "rate", "ratio", "date"):
            last = words[-1]                    # "ad spend" is also "spend", "ad clicks" is also "clicks"
            out.update({last, last[:-1] if last.endswith("s") else last + "s"})
        for suf in (" name", " type", " category", " code", " id"):
            if b.endswith(suf) and len(b) >= len(suf) + 2:
                out.add(b[: -len(suf)])
                out.add(b[: -len(suf)] + "s")
        for pre in ("total ", "avg ", "average ", "loan ", "customer "):
            if b.startswith(pre) and len(b) > len(pre) + 2:
                out.add(b[len(pre):])
    return sorted(out, key=len, reverse=True)


def _spans(q, c):
    """Every place a column is named, as (start, end, phrase length)."""
    out = []
    for ph in _phrases(c):
        for m in re.finditer(rf"(?<![a-z0-9_]){re.escape(ph)}(?![a-z0-9_])", q):
            out.append((m.start(), m.end(), len(ph)))
    return out


def _find(q, c):
    s = _spans(q, c)
    return min(((a, b) for a, b, _ in s), default=None)


def _mentions(q, cols):
    """Column mentions with overlaps resolved: 'profit margin' wins over 'profit' inside it."""
    cand = []
    for c in cols:
        exact = {c["label"].lower().strip(), c["name"], c["name"].replace("_", " ")}
        for a, b, n in _spans(q, c):
            cand.append((a, b, n, c["name"], q[a:b].strip().lower() not in exact))
    cand.sort(key=lambda x: (-x[2], x[4], x[0]))      # longest first; on a tie, the column whose own name it is
    taken, out = [], []
    for a, b, n, name, _ in cand:
        if any(not (b <= x or a >= y) for x, y in taken):
            continue
        taken.append((a, b))
        out.append((a, b, name))
    return sorted(out)


def _best(piece, cols):
    """The one column a phrase names. The longest match wins, then the one ending last ('product category' is category)."""
    best = None
    for c in cols:
        for a, b, n in _spans(piece, c):
            k = (n, b)
            if best is None or k > best[0]:
                best = (k, c)
    return best[1] if best else None


def _amount(num, unit):
    v = float(num.replace(",", ""))
    return v * MULT.get((unit or "").lower(), 1)


def _agg_word(q):
    """The aggregation word that appears first in the text (kept for corrections such as 'use the average')."""
    hits = []
    for pat, agg in [(r"\b(how many|number of|count of|count)\b", "count")] + AGG_KEYS:
        m = re.search(pat, q)
        if m:
            hits.append((m.start(), agg, m))
    if not hits:
        return None, None
    _, agg, m = min(hits, key=lambda h: h[0])
    return agg, m


def _default_agg(col):
    return "avg" if re.search(NON_ADDITIVE, col) else ("sum" if re.search(ADDITIVE, col) else "avg")


def alias(agg, col):
    return f"{ALIAS_WORDS[agg]}_{col}" if col else ALIAS_WORDS[agg]


def _metrics(q, cols, by):
    """Each number column named in the text, with the aggregation word nearest before it in the same clause."""
    meas = [c for c in cols if c["role"] == "measure"]
    found, used = [], []
    for m in FUNC.finditer(q):                                    # SUM(sales), AVG(profit_margin)
        fn, distinct, arg = m.group(1), m.group(2), m.group(3).strip()
        col = _best(" " + arg + " ", [c for c in cols if c["role"] not in ("personal", "empty")])
        agg = {"average": "avg", "mean": "avg"}.get(fn, fn)
        if fn == "count":
            agg = "count_distinct" if distinct or (col and col["role"] != "measure") else "count"
            if arg in ("*", "1"):
                col = None
        if col is None and agg not in ("count",):
            continue
        found.append((m.start(), agg, col["name"] if col else None))
        used.append((m.start(), m.end()))
    aggs = []
    for pat, agg in AGG_KEYS:
        for m in re.finditer(pat, q):
            aggs.append((m.start(), m.end(), agg))
    ment = [x for x in _mentions(q, meas) if not any(a <= x[0] < b for a, b in used)]
    prev_end, prev_agg = -1, None
    for a, b, name in ment:
        clause = max(q.rfind(".", 0, a), q.rfind(";", 0, a), q.rfind(":", 0, a), prev_end)
        near = [g for g in aggs if clause < g[1] <= a and a - g[1] <= 30]
        if near:
            agg = max(near, key=lambda g: g[1])[2]
        elif prev_agg and re.fullmatch(r"[\s,]*(?:and|&|plus|as well as)?[\s,]*", q[prev_end:a]):
            agg = prev_agg                                         # "total sales and profit"
        else:
            agg = _default_agg(name)
        found.append((a, agg, name))
        prev_end, prev_agg = b, agg
    out, seen = [], set()
    for _, agg, col in sorted(found, key=lambda x: x[0]):
        if (agg, col) not in seen:
            seen.add((agg, col))
            out.append({"agg": agg, "column": col})
    return out


def _count_metric(q, cats, dims):
    m = re.search(r"\b(how many|number of|count of|count)\b", q)
    if not m:
        return None
    after = re.sub(r"^\s*(?:distinct|different|unique|separate|of|the)\s+", " ", q[m.end():m.end() + 50])
    after = re.sub(r"^\s*(?:distinct|different|unique|separate|of|the)\s+", " ", after)
    target = next((c for c in cats if _find(" " + after + " ", c) and _find(" " + after + " ", c)[0] <= 2), None)
    dim_cols = [d.split(":")[0] for d in dims]
    if target and target["name"] not in dim_cols and (target["role"] == "identifier" or not dims):
        return {"agg": "count_distinct", "column": target["name"]}
    return {"agg": "count", "column": None}


def interpret(question: str, cols: list, values: dict, max_date=None) -> dict:
    """Rule-based reading of a question against the dataset's own columns. Returns a plan (not yet validated)."""
    q = _plain(question)
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    meas = [c for c in usable if c["role"] == "measure"]
    cats = [c for c in usable if c["role"] in ("category", "identifier")]
    dates = [c for c in usable if c["role"] == "date"]
    plan = {"metrics": [], "dimensions": [], "filters": {}, "period": None, "sort": None, "limit": None,
            "output_format": "auto", "explain": False, "notes": [], "source": "rules"}

    # what the person wants to see
    if re.search(r"\b(?:in|as|into|on) (?:a|an|the|one) (?:table|spreadsheet|grid)\b|\btable (?:with|of|showing)\b|\btabular\b|"
                 r"\bone row (?:per|for)\b|\btabulate\b", q):
        plan["output_format"] = "table"
    elif re.search(r"\b(chart|graph|plot|visuali[sz]e)\b", q):
        plan["output_format"] = "chart"
    plan["explain"] = bool(re.search(r"\b(explain|interpret|insight|commentary|what (?:do |does )?(?:the )?\w+ (?:reveal|mean|tell|show|say)|"
                                     r"what .{0,40}\b(?:reveals?|means?|tells?)\b|why)\b", q))

    # breakdowns the person rules out: "do not group by product_name"
    excluded, masked = [], q
    for m in NOT_GROUP.finditer(q):
        frag = " " + (m.group(1) or m.group(2) or m.group(3) or "") + " "
        for piece in re.split(r"\s*(?:,|\bor\b|\band\b|&|/)\s*", frag):
            c = _best(" " + piece.strip() + " ", cats + dates)
            if c and c["name"] not in excluded:
                excluded.append(c["name"])
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    plan["excluded_dimensions"] = excluded

    # top or bottom N
    tm = re.search(r"\b(top|bottom|best|worst|highest|lowest)\s+(\d+)\s+([a-z_ ]+?)(?=\s+(by|on|for|in|with|where)\b|\s*$)", masked)
    top_dim = None
    if tm:
        plan["limit"] = min(int(tm.group(2)), 100)
        plan["sort"] = {"by": None, "direction": "asc" if tm.group(1) in ("bottom", "worst", "lowest") else "desc"}
        top_dim = _best(" " + tm.group(3) + " ", cats)

    # breakdowns: "for each product category", "by region and product", "one row per category", "monthly"
    dims = []
    for m in BY.finditer(masked):
        for piece in re.split(r"\s*(?:,|\band\b|&|/)\s*", m.group(1)):
            piece = " " + piece.strip() + " "
            if not piece.strip():
                continue
            g = re.search(r"\b(month|week|quarter|year|day)\b", piece)
            dc = _best(piece, dates)
            cc = _best(piece, cats)
            if g and not cc:
                if dates:
                    dims.append(f"{(dc or dates[0])['name']}:{g.group(1)}")
                continue
            c = _best(piece, cats + dates)
            if c:
                dims.append(c["name"] if c["role"] != "date" else f"{c['name']}:month")
    if re.search(r"\b(monthly|month on month|over time|trend)\b", masked) and dates and not any(":" in d for d in dims):
        dims.append(f"{dates[0]['name']}:month")
    if re.search(r"\b(yearly|annual|year on year)\b", masked) and dates and not any(":" in d for d in dims):
        dims.append(f"{dates[0]['name']}:year")
    if top_dim:
        dims.insert(0, top_dim["name"])
    seen = []
    for d in dims:
        if d not in seen and d.split(":")[0] not in excluded:
            seen.append(d)
    plan["dimensions"] = seen[:3]
    plan["dimensions_only"] = bool(re.search(r"\b(?:only|alone|just)\b(?=[^.]{0,3}$|[ ,.;])", q) and seen and
                                   re.search(r"\b(?:by|per)\s+[a-z_ ]+?\s+(?:only|alone)\b|\b(?:only|just)\s+(?:by|per)\b", q))

    # number conditions: "amount above 1 million"
    for c in meas:
        for ph in _phrases(c):
            for m in re.finditer(rf"(?<![a-z0-9_]){re.escape(ph)}(?![a-z0-9_])\s*(?:is\s+|of\s+|was\s+)?"
                                 r"(above|over|more than|greater than|at least|below|under|less than|at most|"
                                 r"not more than|not less than|exactly|equal to)\s*(?:₦|ngn|n|\$)?\s*([\d][\d,]*(?:\.\d+)?)\s*"
                                 r"(k|thousand|m|million|mn|bn|billion|b)?\b", masked):
                op = {"above": "gt", "over": "gt", "more than": "gt", "greater than": "gt", "at least": "gte",
                      "not less than": "gte", "below": "lt", "under": "lt", "less than": "lt", "at most": "lte",
                      "not more than": "lte", "exactly": "eq", "equal to": "eq"}[m.group(1)]
                plan["filters"][c["name"]] = {op: _amount(m.group(2), m.group(3))}
            if c["name"] in plan["filters"]:
                break

    # category values named in the question. Words that describe the output ("in a table") and words that
    # name a column are not values.
    vq = OUTPUT_PHRASES.sub(lambda m: " " * len(m.group(0)), masked)
    colwords = {ph for c in usable for ph in _phrases(c)}
    taken = []
    for c in [c for c in cols if c["role"] == "category"]:
        inc, exc = [], []
        for v in sorted(values.get(c["name"], []), key=len, reverse=True):
            lv = v.lower().strip()
            if len(lv) < 2 or lv in ("yes", "no", "n/a", "na", "none", "other", "all", "total") or lv in colwords:
                continue
            mm = re.search(rf"(?<![a-z0-9]){re.escape(lv)}(?![a-z0-9])", vq)
            if not mm or any(a <= mm.start() < b for a, b in taken):
                continue
            taken.append((mm.start(), mm.end()))
            (exc if re.search(NEG + re.escape(lv) + r"(?![a-z0-9])", vq) else inc).append(v)
        changed = True
        while changed:
            changed = False
            for v in list(inc):
                if any(re.search(rf"{re.escape(e.lower())}\s*(?:,|and|or|nor)\s*(?:the\s+)?{re.escape(v.lower())}(?![a-z0-9])", vq)
                       for e in exc):
                    inc.remove(v)
                    exc.append(v)
                    changed = True
        known = values.get(c["name"], [])
        if inc and not exc and len(known) < 200 and set(inc) >= set(known):
            plan["notes"].append(f"You listed every {c['label']} value, so no {c['label']} filter is applied.")
            inc = []
        f = T.filter_value(inc, exc)
        if f is not None:
            plan["filters"][c["name"]] = f
    # a breakdown on a column the question narrows to one value is a filter, not a breakdown
    plan["dimensions"] = [d for d in plan["dimensions"]
                          if not (d in plan["filters"] and isinstance(plan["filters"][d], str))]

    # periods relative to the latest date in the data
    if dates:
        dc = next((d for d in dates if _find(masked, d)), dates[0])
        end = pd.Timestamp(max_date) if max_date else pd.Timestamp.today()
        per = None
        m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b", masked)
        if m:
            k = int(m.group(1))
            start = end - {"day": pd.DateOffset(days=k), "week": pd.DateOffset(weeks=k), "month": pd.DateOffset(months=k),
                           "year": pd.DateOffset(years=k)}[m.group(2)] + pd.Timedelta(days=1)
            per = (start, end, m.group(0).strip())
        elif re.search(r"\b(last|past|previous) (month|week|year)\b", masked):
            u = re.search(r"\b(last|past|previous) (month|week|year)\b", masked).group(2)
            start = end - {"week": pd.DateOffset(weeks=1), "month": pd.DateOffset(months=1), "year": pd.DateOffset(years=1)}[u] + pd.Timedelta(days=1)
            per = (start, end, f"last {u}")
        else:
            ys = re.findall(r"\b(?:in|during|for|of)\s+(20\d\d|19\d\d)\b", masked)
            mon = re.search(r"\b(?:in|during|for)\s+(" + "|".join(MONTHS) + r")(?:\s+(20\d\d))?\b", masked)
            if mon:
                y = int(mon.group(2) or (ys[0] if ys else end.year))
                mi = MONTHS.index(mon.group(1)) + 1
                if not mon.group(2) and not ys and mi > end.month:
                    y -= 1
                start = pd.Timestamp(y, mi, 1)
                per = (start, start + pd.offsets.MonthEnd(0), f"{mon.group(1).title()} {y}")
            elif ys:
                y = int(ys[0])
                per = (pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), str(y))
            bm = re.search(r"\bbetween\s+(" + "|".join(MONTHS) + r")\s+and\s+(" + "|".join(MONTHS) + r")(?:\s+(20\d\d))?\b", masked)
            if bm:
                y = int(bm.group(3) or end.year)
                a, b = MONTHS.index(bm.group(1)) + 1, MONTHS.index(bm.group(2)) + 1
                s = pd.Timestamp(y, a, 1)
                per = (s, pd.Timestamp(y, b, 1) + pd.offsets.MonthEnd(0), f"{bm.group(1).title()} to {bm.group(2).title()} {y}")
        if per:
            plan["period"] = {"column": dc["name"], "from": str(per[0].date()), "to": str(per[1].date()), "label": per[2]}

    # the metrics
    cond_cols = {k for k, v in plan["filters"].items() if isinstance(v, dict) and set(v) & set(SQL_OPS)}
    mets = [m for m in _metrics(masked, usable, None) if m["column"] not in cond_cols or m["agg"] != _default_agg(m["column"])]
    cnt = _count_metric(masked, cats, plan["dimensions"])
    if cnt and not any(m["agg"] in ("count", "count_distinct") for m in mets):
        pos = re.search(r"\b(how many|number of|count of|count)\b", masked).start()
        first = min([_find(masked, {"name": m["column"], "label": m["column"]})[0] for m in mets if m["column"]
                     and _find(masked, {"name": m["column"], "label": m["column"]})] or [10 ** 6])
        mets = ([cnt] + mets) if pos < first else (mets + [cnt])
    if not mets and re.search(ADDITIVE, masked) and meas and (plan["dimensions"] or re.search(r"\b(trend|how much|what is|show)\b", masked)):
        stems = re.findall(ADDITIVE, masked)
        col = next((c for c in meas if any(st in c["name"] for st in stems)), None) or \
            next((c for c in meas if re.search(ADDITIVE, c["name"])), None)
        if col:
            mets = [{"agg": "sum", "column": col["name"]}]
    if not mets and (plan["dimensions"] or plan["filters"]) and \
            re.search(r"\b(rows|records|entries|transactions|loans|customers|accounts|items|orders)\b", masked):
        mets = [{"agg": "count", "column": None}]
    plan["metrics"] = mets[:6]
    if plan["sort"]:
        plan["sort"]["by"] = alias(mets[0]["agg"], mets[0]["column"]) if mets else None
    if not mets:
        names = ", ".join(c["label"] for c in sorted(meas, key=lambda c: not re.search(ADDITIVE, c["name"]))[:5]) or "a number column"
        plan["clarify"] = (f"Which measure do you need, and how? For example: total, average or count of {names}."
                           if meas else "This dataset has no number columns. Ask for a count, for example: how many rows by "
                           + (cats[0]["label"] if cats else "a column") + ".")
    return plan


def interpret_llm(question: str, cols: list, values: dict) -> dict | None:
    """Language model reading, checked against the columns before use."""
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    desc = [{"column": c["name"], "label": c["label"], "type": c["type"], "role": c["role"],
             "values": values.get(c["name"], [])[:25]} for c in usable]
    sys_ = ("You translate a question about a spreadsheet into a structured analytical plan. Use only the columns given. "
            'Return JSON {"metrics": [{"agg": "sum|avg|count|count_distinct|min|max|median", "column": name or null}], '
            '"dimensions": [column names, or "datecolumn:month|quarter|year"], '
            '"filters": {column: value | [values] | {"not": value or list} | {"gt"|"gte"|"lt"|"lte"|"eq": number}}, '
            '"period": null or {"column": datecolumn, "from": "YYYY-MM-DD", "to": "YYYY-MM-DD", "label": text}, '
            '"sort": null or {"by": "first", "direction": "desc|asc"}, "limit": number or null, '
            '"output_format": "table|chart|auto", "explain": true|false}. '
            "List every metric the person asks for, each with its own aggregation: 'total' and 'sum' mean sum, "
            "'average' and 'mean' mean avg. Add a dimension only when the person asks to break results down by it. "
            "Do not add a filter that lists every value of a column. Read negation carefully: 'not X', 'except X' and "
            "'excluding X' mean {\"not\": \"X\"}. If the question cannot be answered from the columns, return "
            '{"metrics": [], "clarify": "<question to ask the person>"}.')
    out = llm.chat_json(sys_, json.dumps({"question": question, "columns": desc}), max_tokens=600)
    if not out:
        return None
    if out.get("clarify") and not out.get("metrics"):
        return {"metrics": [], "dimensions": [], "filters": {}, "clarify": out["clarify"], "source": "llm"}
    try:
        out = validate({**out, "source": "llm"}, cols)
    except ValueError:
        return None
    used = llm.last_call()
    if used.get("model"):
        out["notes"] = (out.get("notes") or []) + [f"Read by the language model {used['model']}" +
                                                  (f" after {used['fallbacks_used']} unavailable model(s)." if used["fallbacks_used"] else ".")]
    return out


def _normalise(intent: dict) -> dict:
    """Accept the version 4.0 shape ({"measure": ...}) as well as the plan shape."""
    p = dict(intent or {})
    if "metrics" not in p and p.get("measure"):
        p["metrics"] = [p["measure"]]
    if "sort" not in p and p.get("order"):
        p["sort"] = {"by": None, "direction": p["order"]}
    p.pop("measure", None)
    p.pop("order", None)
    return p


def validate(intent: dict, cols: list) -> dict:
    """Every column and operator must exist in the dataset. Personal columns are never used."""
    p = _normalise(intent)
    by = {c["name"]: c for c in cols}
    ok = lambda n, roles: n in by and by[n]["role"] in roles
    mets = []
    for m in p.get("metrics") or []:
        agg, col = (m or {}).get("agg"), (m or {}).get("column")
        if agg not in AGGS:
            raise ValueError("Choose how to measure: total, average, count, distinct count, smallest, largest or median.")
        if agg == "count":
            col = None
        elif agg == "count_distinct":
            if not ok(col, ("category", "identifier", "date", "measure")):
                raise ValueError("Choose a column to count distinct values of.")
        elif not ok(col, ("measure",)):
            raise ValueError(f"Choose a number column to measure{' instead of ' + by[col]['label'] if col in by else ''}.")
        if (agg, col) not in [(x["agg"], x["column"]) for x in mets]:
            mets.append({"agg": agg, "column": col, "alias": alias(agg, col)})
    if not mets:
        raise ValueError("Say at least one measure, for example total sales.")
    if len(mets) > 6:
        raise ValueError("Ask for six measures or fewer at a time.")
    dims = []
    for d in p.get("dimensions") or []:
        name, _, grain = str(d).partition(":")
        if ok(name, ("date",)):
            v = f"{name}:{grain if grain in ('day', 'week', 'month', 'quarter', 'year') else 'month'}"
        elif ok(name, ("category", "identifier")):
            v = name
        else:
            raise ValueError(f"'{d}' is not a column you can break results down by.")
        if v not in dims:
            dims.append(v)
    filters = {}
    for k, v in (p.get("filters") or {}).items():
        if v in (None, "", [], {}):
            continue
        if not ok(k, ("category", "identifier", "measure", "date")):
            raise ValueError(f"'{k}' is not a column you can filter on.")
        if isinstance(v, dict) and set(v) & set(SQL_OPS):
            if by[k]["type"] != "number":
                raise ValueError(f"'{by[k]['label']}' is not a number column.")
            filters[k] = {op: float(x) for op, x in v.items() if op in SQL_OPS}
        elif isinstance(v, dict) and "not" in v:
            vals = v["not"] if isinstance(v["not"], list) else [v["not"]]
            filters[k] = {"not": [str(x) for x in vals] if len(vals) > 1 else str(vals[0])}
        else:
            vals = v if isinstance(v, list) else [v]
            filters[k] = [str(x) for x in vals] if len(vals) > 1 else str(vals[0])
    per = p.get("period")
    if per:
        if not ok(per.get("column"), ("date",)):
            raise ValueError("The period must use a date column.")
        try:
            per = {"column": per["column"], "from": str(pd.Timestamp(per["from"]).date()), "to": str(pd.Timestamp(per["to"]).date()),
                   "label": per.get("label") or f"{per['from']} to {per['to']}"}
        except Exception:
            raise ValueError("The period dates could not be read.")
    lim = p.get("limit")
    lim = max(1, min(int(lim), 100)) if lim else None
    sort = p.get("sort")
    if sort or lim:
        sort = sort or {}
        aliases = [m["alias"] for m in mets]
        sort = {"by": sort.get("by") if sort.get("by") in aliases else aliases[0],
                "direction": sort.get("direction") if sort.get("direction") in ("asc", "desc") else "desc"}
    fmt = p.get("output_format") if p.get("output_format") in ("table", "chart", "auto") else "auto"
    if len(dims) > 3:
        raise ValueError("Break results down by three columns or fewer.")
    return {"metrics": mets, "dimensions": dims, "filters": filters, "period": per, "sort": sort, "limit": lim,
            "grain": [d.split(":")[0] if ":" not in d else d for d in dims], "output_format": fmt,
            "explain": bool(p.get("explain")), "version": int(p.get("version") or 1),
            "notes": [str(n) for n in (p.get("notes") or [])][:5], "source": p.get("source", "rules")}


PLAN_FIELDS = ("metrics", "dimensions", "filters", "period", "sort", "limit", "output_format", "explain")


def plan_hash(plan: dict) -> str:
    """Identifies the analytical content of a plan. Version, notes and source are not part of it."""
    core = {k: plan.get(k) for k in PLAN_FIELDS}
    core["metrics"] = [{"agg": m["agg"], "column": m["column"]} for m in plan.get("metrics") or []]
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _num_text(v):
    v = float(v)
    return f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"


def metric_label(m, by):
    if m["agg"] == "count":
        return "Number of rows"
    lab = by.get(m["column"], m["column"])
    return f"Number of distinct {lab}" if m["agg"] == "count_distinct" else f"{AGG_WORDS[m['agg']]} {lab}"


def describe(intent: dict, cols: list) -> str:
    """The plan in plain words, for the person who asked to check."""
    p = _normalise(intent)
    by = {c["name"]: c["label"] for c in cols}
    labs = [metric_label(m, by) for m in p.get("metrics") or []]
    labs = [labs[0]] + [x[0].lower() + x[1:] for x in labs[1:]] if labs else ["(no measure)"]
    parts = [labs[0] if len(labs) == 1 else ", ".join(labs[:-1]) + " and " + labs[-1]]
    if p.get("dimensions"):
        ds = []
        for d in p["dimensions"]:
            n, _, g = d.partition(":")
            ds.append(f"{g} of {by.get(n, n)}" if g else by.get(n, n))
        parts.append("by " + " and ".join(ds))
    for k, v in (p.get("filters") or {}).items():
        lab = by.get(k, k)
        if isinstance(v, dict) and "not" in v:
            vals = v["not"] if isinstance(v["not"], list) else [v["not"]]
            parts.append(f"{lab} is not {vals[0]}" if len(vals) == 1 else
                         f"{lab} is neither {vals[0]} nor {vals[1]}" if len(vals) == 2 else
                         f"{lab} is none of {', '.join(vals[:-1])} and {vals[-1]}")
        elif isinstance(v, dict):
            parts.append(" and ".join(f"{lab} {OP_WORDS[o]} {_num_text(x)}" for o, x in v.items()))
        else:
            vals = v if isinstance(v, list) else [v]
            parts.append(f"{lab} is {' or '.join(vals)}")
    if p.get("limit"):
        sort = p.get("sort") or {}
        parts.append(f"{'bottom' if sort.get('direction') == 'asc' else 'top'} {p['limit']}")
    per = p.get("period")
    if per:
        lab = per["label"]
        parts.append(f"{by.get(per['column'], per['column'])} in {'the ' if lab.startswith('last') else ''}{lab}")
    else:
        parts.append("all rows")
    return ", ".join(parts) + "."


def constraints(text: str, cols: list) -> dict:
    """Explicit instructions in a correction that any plan built from it must respect."""
    q = _plain(text)
    usable = [c for c in cols if c["role"] not in ("personal", "empty")]
    p = interpret(text, cols, {}, None)
    named = {(m["agg"], m["column"]) for m in _metrics(q, usable, None)
             if FUNC.search(q) or re.search(r"\b(sum|total|average|avg|mean|median|min|max)\b", q)}
    return {"metrics": sorted(named, key=str), "excluded_dimensions": p.get("excluded_dimensions", []),
            "only_dimensions": p["dimensions"] if p.get("dimensions_only") else None,
            "no_filters": bool(re.search(r"\b(no filters?|without (?:any )?filters?|all rows|do not filter|don't filter)\b", q))}


def conflicts(plan: dict, cons: dict, cols: list) -> list:
    """Where a plan contradicts the person's explicit instructions. Each item is a sentence for the person."""
    if not cons:
        return []
    by = {c["name"]: c["label"] for c in cols}
    out = []
    have = {(m["agg"], m["column"]) for m in plan.get("metrics") or []}
    for agg, col in cons.get("metrics") or []:
        if (agg, col) not in have:
            out.append(f"You asked for {metric_label({'agg': agg, 'column': col}, by).lower()}, but the plan does not include it.")
    dims = [d.split(":")[0] for d in plan.get("dimensions") or []]
    for d in cons.get("excluded_dimensions") or []:
        if d in dims:
            out.append(f"You said not to group by {by.get(d, d)}, but the plan groups by it.")
    only = cons.get("only_dimensions")
    if only is not None and sorted(only) != sorted(plan.get("dimensions") or []):
        out.append("You asked to group by " + " and ".join(by.get(d.split(':')[0], d) for d in only) + " only, but the plan groups differently.")
    if cons.get("no_filters") and plan.get("filters"):
        out.append("You asked for no filters, but the plan filters the rows.")
    return out


# ═════════════════════════ governed query (L5.1 D17) ═════════════════════════

def build_sql(r: dict, intent: dict):
    """Compile a validated plan into SQL. Identifiers come only from the plan; values are parameters."""
    cols = r["columns"]
    p = validate(intent, cols)
    by = {c["name"]: c for c in cols}
    q = lambda n: f'"{n}"'
    exprs = []
    for m in p["metrics"]:
        c = m["column"]
        e = {"sum": f"SUM({q(c)})", "avg": f"AVG({q(c)})", "min": f"MIN({q(c)})", "max": f"MAX({q(c)})",
             "median": f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {q(c)})", "count": "COUNT(*)",
             "count_distinct": f"COUNT(DISTINCT {q(c)})"}[m["agg"]]
        exprs.append(f'{e} AS "{m["alias"]}"')
    sel, groups, labels = [], [], []
    for d in p["dimensions"]:
        n, _, g = d.partition(":")
        lab = (by[n]["label"] + (f" ({g})" if g else ""))[:60].replace('"', "")
        e = {"day": f"to_char({q(n)}, 'YYYY-MM-DD')", "week": f"to_char(date_trunc('week', {q(n)}), 'YYYY-MM-DD')",
             "month": f"to_char({q(n)}, 'YYYY-MM')", "quarter": f"to_char({q(n)}, 'YYYY-\"Q\"Q')",
             "year": f"to_char({q(n)}, 'YYYY')"}[g] if g else f"{q(n)}::text"
        sel.append(f'{e} AS "{lab}"')
        groups.append(e)
        labels.append(lab)
    where, params = [], []
    for k, v in p["filters"].items():
        if isinstance(v, dict) and "not" in v:
            vals = v["not"] if isinstance(v["not"], list) else [v["not"]]
            where.append(f"({q(k)}::text NOT IN ({', '.join('?' for _ in vals)}) OR {q(k)} IS NULL)")
            params += vals
        elif isinstance(v, dict):
            for op, x in v.items():
                where.append(f"{q(k)} {SQL_OPS[op]} ?")
                params.append(x)
        else:
            vals = v if isinstance(v, list) else [v]
            where.append(f"{q(k)}::text IN ({', '.join('?' for _ in vals)})")
            params += vals
    if p["period"]:
        where.append(f"{q(p['period']['column'])} BETWEEN ? AND ?")
        params += [p["period"]["from"], p["period"]["to"]]
    body = f" FROM {r['table_name']}" + (" WHERE " + " AND ".join(where) if where else "")
    sql = f"SELECT {', '.join(sel + exprs)}{body}"
    if groups:
        sql += " GROUP BY " + ", ".join(groups)
        if p["limit"]:
            sql += f' ORDER BY "{p["sort"]["by"]}" {"ASC" if p["sort"]["direction"] == "asc" else "DESC"} NULLS LAST LIMIT {p["limit"]}'
        elif any(":" in d for d in p["dimensions"]):
            sql += " ORDER BY " + ", ".join(str(n + 1) for n in range(len(groups)))
        else:
            sql += f' ORDER BY "{p["metrics"][0]["alias"]}" DESC NULLS LAST LIMIT 500'
    grain_sql = (f"SELECT COUNT(*) AS n FROM (SELECT {', '.join(groups)}{body} GROUP BY {', '.join(groups)}) g"
                 if groups else None)
    return sql, params, p, labels, grain_sql


def check_sql(sql: str, plan: dict, table: str) -> list:
    """Independent check that compiled SQL matches the plan: table, aggregations, grouping, filters and grain.
    Returns a list of problems. An empty list means the SQL does exactly what the plan says."""
    problems = []
    s = sql
    if not re.search(rf"\bFROM {re.escape(table)}\b", s):
        problems.append("the query does not read the dataset's own table")
    head = s.split(" FROM ", 1)[0]
    found = re.findall(r'(SUM|AVG|MIN|MAX|COUNT|percentile_cont)\((DISTINCT )?(?:0\.5\) WITHIN GROUP \(ORDER BY )?("?[a-z0-9_*]+"?)\)+ AS "([a-z0-9_]+)"', head)
    want = {(m["agg"], m["column"], m["alias"]) for m in plan["metrics"]}
    fmap = {"SUM": "sum", "AVG": "avg", "MIN": "min", "MAX": "max", "percentile_cont": "median"}
    got = set()
    for fn, dist, col, al in found:
        col = col.strip('"')
        agg = ("count_distinct" if dist else "count") if fn == "COUNT" else fmap[fn]
        got.add((agg, None if col == "*" else col, al))
    if got != want:
        problems.append(f"aggregations {sorted(map(str, got))} do not match the plan {sorted(map(str, want))}")
    gb = s.split(" GROUP BY ", 1)[1].split(" ORDER BY ")[0] if " GROUP BY " in s else ""
    gcols = re.findall(r'"([a-z0-9_]+)"', gb)
    if sorted(gcols) != sorted(d.split(":")[0] for d in plan["dimensions"]):
        problems.append(f"grouping by {gcols or 'nothing'} does not match the plan {plan['dimensions'] or 'nothing'}")
    where = s.split(" WHERE ", 1)[1].split(" GROUP BY ")[0] if " WHERE " in s else ""
    fcols = sorted(set(re.findall(r'"([a-z0-9_]+)"', where)) - ({plan["period"]["column"]} if plan.get("period") else set()))
    if fcols != sorted(plan["filters"]):
        problems.append(f"filters on {fcols or 'nothing'} do not match the plan {sorted(plan['filters']) or 'nothing'}")
    if bool(plan.get("period")) != (plan.get("period") is not None and "BETWEEN" in where):
        problems.append("the period in the query does not match the plan")
    if plan.get("limit") and f"LIMIT {plan['limit']}" not in s:
        problems.append("the row limit does not match the plan")
    return problems


def _query_precheck(ctx, dataset_id, intent, **_):
    p = access_problem(dataset_id, ctx.created_by)
    if p:
        return p
    pol = policy()
    from . import bank_data as BK
    if BK.is_bank(dataset_id):
        if not BK.allowed(dataset_id, ctx.role):
            return f"Role {ctx.role} may not use this bank data under the bank's policy."
    elif not pol.get("enabled", True) or ctx.role not in pol.get("roles", []):
        return f"Role {ctx.role} may not analyse uploaded data under the bank's policy."
    cols = {c["name"]: c for c in _row(dataset_id)["columns"]}
    it = _normalise(intent)
    used = [m.get("column") for m in it.get("metrics") or []] + [str(d).split(":")[0] for d in it.get("dimensions") or []] \
        + list((it.get("filters") or {}).keys())
    if any(u and cols.get(u, {}).get("role") == "personal" for u in used):
        return "Personal columns are masked and cannot be used in a query."
    return None


@T.tool("D17", "Uploaded dataset query", "L5.1", "read", policy="uploads", precheck=_query_precheck)
def dataset_query(ctx, dataset_id, intent):
    """Governed query over a person's uploaded dataset. SQL is compiled from the validated plan and checked against it."""
    r = _row(dataset_id)
    sql, params, p, labels, grain_sql = build_sql(r, intent)
    problems = check_sql(sql, p, r["table_name"])
    if problems:
        raise ValueError("The query did not match the plan, so it was not run: " + "; ".join(problems) + ".")
    rows = db.query(sql, params)
    for row in rows:
        for m in p["metrics"]:
            if row.get(m["alias"]) is not None:
                row[m["alias"]] = round(float(row[m["alias"]]), 4)
    groups = db.one(grain_sql, params)["n"] if grain_sql else 1
    by = {c["name"]: c["label"] for c in r["columns"]}
    return {"rows": rows, "sql": sql, "params": params, "plan": p, "plan_hash": plan_hash(p), "plan_version": p["version"],
            "dims": labels, "metrics": [{"alias": m["alias"], "label": metric_label(m, by), "agg": m["agg"]} for m in p["metrics"]],
            "label": describe({**p, "dimensions": [], "filters": {}, "period": None, "limit": None}, r["columns"]).replace(", all rows.", ""),
            "expected_groups": groups, "sql_check": "The query matches the plan: aggregations, grouping, filters, period and limit.",
            "rows_in_dataset": r["row_count"], "dataset": r["name"]}


@T.tool("D18", "Uploaded dataset profile", "L5.1", "read", policy="uploads",
        precheck=lambda ctx, dataset_id, **_: access_problem(dataset_id, ctx.created_by))
def dataset_profile(ctx, dataset_id):
    """Column types, data quality findings and the latest date of a person's uploaded dataset."""
    r = _row(dataset_id)
    dates = [c for c in r["profile"]["columns"] if c["type"] == "date" and c.get("max")]
    return {"columns": r["columns"], "issues": r["profile"]["issues"], "score": r["profile"]["score"],
            "rows": r["row_count"], "values": category_values(r), "max_date": max((c["max"] for c in dates), default=None),
            "name": r["name"]}


@T.tool("C12", "Dataset question interpreter", "L5.4", "compute", policy="uploads")
def dataset_intent(ctx, question, columns, values, max_date=None):
    """Rule-based reading of a question against an uploaded dataset's columns."""
    return interpret(question, columns, values, max_date)


@T.tool("M18", "LLM dataset interpreter", "L5.2", "compute", model="llm")
def dataset_intent_llm(ctx, question, columns, values):
    """Language model reading of a question against an uploaded dataset, validated before use."""
    return interpret_llm(question, columns, values)
