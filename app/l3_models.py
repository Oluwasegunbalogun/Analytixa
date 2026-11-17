"""L3 Models: credit PD, churn, transaction anomaly and ATM cash demand.

Models are trained once on the synthetic data and stored in the database model
registry with a version, validation metrics and the fitted artefact, so the
whole installation lives in Postgres. Agents reach them only through L5.2
model services.
"""
import io
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import features as F
from . import l1_data as db

_cache = {}
DDL = """
CREATE TABLE IF NOT EXISTS model_registry (
  name       text PRIMARY KEY,
  version    text NOT NULL,
  model_type text NOT NULL,
  metrics    text NOT NULL,
  artefact   bytea NOT NULL,
  trained_at timestamptz NOT NULL DEFAULT now()
);
"""


def _save(name, obj):
    db.script(DDL)
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    metrics = {k: v for k, v in obj["metrics"].items() if k != "holdout_ids"}
    db.execute("INSERT INTO model_registry (name, version, model_type, metrics, artefact) VALUES (?,?,?,?,?) "
               "ON CONFLICT (name) DO UPDATE SET version=EXCLUDED.version, model_type=EXCLUDED.model_type, "
               "metrics=EXCLUDED.metrics, artefact=EXCLUDED.artefact, trained_at=now()",
               (name, obj["version"], obj["type"], json.dumps(metrics), buf.getvalue()))
    _cache[name] = obj


def trained() -> bool:
    return db.table_exists("model_registry") and \
        db.one("SELECT COUNT(*) n FROM model_registry")["n"] >= 3


def load(name):
    if name not in _cache:
        r = db.one("SELECT artefact FROM model_registry WHERE name=?", (name,))
        if not r:
            raise RuntimeError(f"Model {name} is not trained. Restart the server to train it.")
        _cache[name] = joblib.load(io.BytesIO(r["artefact"]))
    return _cache[name]


def clear_cache():
    _cache.clear()


def train_credit():
    h = pd.DataFrame(db.query("SELECT * FROM loan_history ORDER BY history_id"))
    X = F.credit_frame(h)
    y = h.defaulted
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=1, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
    auc = roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])
    _save("credit_pd", {"version": "credit_pd-v1", "scaler": sc, "model": m, "features": F.CREDIT_FEATURES,
                        "metrics": {"auc_holdout": round(float(auc), 3), "train_rows": len(Xtr),
                                    "default_rate": round(float(y.mean()), 3)},
                        "type": "logistic regression"})
    return auc


def train_churn():
    fr = F.churn_frame()
    truth = pd.DataFrame(db.query("SELECT * FROM _truth_churn")).set_index("customer_id")
    y = truth.churned_next_90d.reindex(fr.index)
    X = fr[F.CHURN_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=2, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    auc = roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])
    _save("churn", {"version": "churn-v1", "scaler": sc, "model": m, "features": F.CHURN_FEATURES,
                    "metrics": {"auc_holdout": round(float(auc), 3), "train_rows": len(Xtr),
                                "holdout_ids": list(Xte.index), "base_rate": round(float(y.mean()), 3)},
                    "type": "logistic regression (class-balanced)"})
    return auc


TXN_FEATURES = ["amount_ratio", "night", "is_pos", "is_cash", "is_transfer_in", "new_counterparty"]


def txn_frame(tx: pd.DataFrame) -> pd.DataFrame:
    med = tx.groupby("customer_id").amount.transform("median").clip(lower=100)
    hour = pd.to_datetime(tx.ts).dt.hour
    cp_count = tx.groupby(["customer_id", "counterparty"]).txn_id.transform("count")
    return pd.DataFrame({
        "amount_ratio": np.log1p(tx.amount / med),
        "night": hour.between(0, 4).astype(int),
        "is_pos": (tx.channel == "POS").astype(int),
        "is_cash": (tx.channel == "branch_cash").astype(int),
        "is_transfer_in": ((tx.channel == "transfer") & (tx.direction == "credit")).astype(int),
        "new_counterparty": (cp_count <= 1).astype(int),
    })


def train_anomaly():
    tx = pd.DataFrame(db.query("SELECT DISTINCT txn_id, customer_id, ts, amount, direction, channel, counterparty "
                               "FROM transactions WHERE ts < '2026-09-20' ORDER BY txn_id"))
    X = txn_frame(tx)
    m = IsolationForest(n_estimators=150, contamination=.01, random_state=5).fit(X)
    s = -m.score_samples(X)
    _save("anomaly", {"version": "anomaly-v1", "model": m, "features": TXN_FEATURES,
                      "score_min": float(np.percentile(s, 1)), "score_max": float(np.percentile(s, 99.9)),
                      "metrics": {"train_rows": len(X)}, "type": "isolation forest"})


def train_all() -> dict:
    out = {"credit_auc": train_credit(), "churn_auc": train_churn()}
    train_anomaly()
    for name in ["credit_pd", "churn", "anomaly"]:
        m = load(name)
        db.ledger("model_registered", payload={"model": m["version"], "type": m["type"],
                                               "metrics": {k: v for k, v in m["metrics"].items() if k != "holdout_ids"}},
                  model_version=m["version"])
    return out


def registry() -> list:
    """Model names, versions and validation metrics. Reads the registry table only, never the trained artefacts."""
    rows = {r["name"]: r for r in db.query("SELECT name, version, model_type, metrics FROM model_registry")}
    out = []
    for name in ["credit_pd", "churn", "anomaly"]:
        r = rows.get(name)
        if not r:
            continue
        metrics = json.loads(r["metrics"]) if isinstance(r["metrics"], str) else (r["metrics"] or {})
        out.append({"name": name, "version": r["version"], "type": r["model_type"],
                    "metrics": {k: v for k, v in metrics.items() if k != "holdout_ids"}})
    out.append({"name": "cash_demand", "version": "cash_demand-v1", "type": "seasonal day-of-week model with payday effect",
                "metrics": {"history_days": 180}})
    return out


# ── scoring helpers used by L5.2 services ─────────────────────────

def credit_score(frame: pd.DataFrame):
    m = load("credit_pd")
    z = m["scaler"].transform(frame[m["features"]])
    return m["model"].predict_proba(z)[:, 1], z, m


def churn_score(frame: pd.DataFrame):
    m = load("churn")
    z = m["scaler"].transform(frame[m["features"]])
    return m["model"].predict_proba(z)[:, 1], z, m


def anomaly_score(tx: pd.DataFrame):
    m = load("anomaly")
    s = -m["model"].score_samples(txn_frame(tx)[m["features"]])
    return np.clip((s - m["score_min"]) / (m["score_max"] - m["score_min"]), 0, 1), m


def cash_forecast(hist: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Day-of-week mean over the last 8 weeks, scaled by a payday factor, with a 90% interval."""
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h[h.date > h.date.max() - pd.Timedelta(days=56)]
    h["dow"] = h.date.dt.dayofweek
    h["pay"] = ((h.date.dt.day >= 25) | (h.date.dt.day <= 2)).astype(int)
    base = h[h.pay == 0].groupby("dow").withdrawals.mean()
    pay_factor = (h[h.pay == 1].withdrawals.mean() / h[h.pay == 0].withdrawals.mean()) if h.pay.sum() else 1.0
    fitted = h.apply(lambda r: base.get(r.dow, base.mean()) * (pay_factor if r.pay else 1), axis=1)
    resid_sd = float((h.withdrawals - fitted).std())
    rows = []
    for d in pd.date_range(start, periods=days, freq="D"):
        pay = d.day >= 25 or d.day <= 2
        f = base.get(d.dayofweek, base.mean()) * (pay_factor if pay else 1)
        rows.append({"date": str(d.date()), "forecast": round(f, -3), "low": round(max(0, f - 1.645 * resid_sd), -3),
                     "high": round(f + 1.645 * resid_sd, -3)})
    return pd.DataFrame(rows), {"pay_factor": round(float(pay_factor), 3), "resid_sd": round(resid_sd, -3)}
