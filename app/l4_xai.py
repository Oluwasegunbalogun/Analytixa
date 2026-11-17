"""L4 Explainable AI.

The credit and churn models are linear in log-odds, so each feature's
contribution (coefficient times standardised value) is an exact additive
attribution of the score, equal to SHAP values under feature independence.
Faithfulness is checked by confirming that the attributions add back to the
model's log-odds. Stability is checked by perturbing inputs slightly.
"""
import numpy as np
import pandas as pd

from . import features as F

REASON_TEXT = {
    "dti": "Debt-to-income ratio is high",
    "log_income": "Monthly income is low for the amount requested",
    "tenure_months": "Banking relationship is short",
    "utility_ontime_ratio": "Utility bills are often paid late",
    "log_airtime": "Low airtime spend gives weak alternative-data evidence",
    "mm_inflow_ratio": "Mobile money inflow is low relative to income",
    "has_bureau": "No credit bureau file",
    "bureau_centered": "Credit bureau score is weak",
    "prior_default": "Previous loan default on record",
    "informal_income": "Income comes from an informal source",
}
POSITIVE_TEXT = {
    "dti": "Debt-to-income ratio is comfortable",
    "log_income": "Income is strong relative to the request",
    "tenure_months": "Long banking relationship",
    "utility_ontime_ratio": "Utility bills are paid on time",
    "log_airtime": "Stable airtime spend",
    "mm_inflow_ratio": "Healthy mobile money inflow",
    "has_bureau": "Has a credit bureau file",
    "bureau_centered": "Strong credit bureau score",
    "prior_default": "No previous default",
    "informal_income": "Formal income source",
}


def linear_attributions(model_bundle: dict, z_row: np.ndarray) -> dict:
    coef = model_bundle["model"].coef_[0]
    intercept = float(model_bundle["model"].intercept_[0])
    contrib = coef * z_row
    logit = intercept + contrib.sum()
    direct = float(model_bundle["model"].decision_function(z_row.reshape(1, -1))[0])
    return {
        "base_logodds": round(intercept, 4),
        "contributions": {f: round(float(c), 4) for f, c in zip(model_bundle["features"], contrib)},
        "logodds": round(float(logit), 4),
        "faithful": abs(logit - direct) < 1e-6,
        "faithfulness_error": abs(float(logit - direct)),
    }


def reason_codes(attr: dict, k: int = 4) -> dict:
    items = sorted(attr["contributions"].items(), key=lambda x: -x[1])
    adverse = [{"feature": f, "impact": v, "reason": REASON_TEXT[f]} for f, v in items if v > 0.05][:k]
    support = [{"feature": f, "impact": v, "reason": POSITIVE_TEXT[f]} for f, v in reversed(items) if v < -0.05][:k]
    return {"adverse": adverse, "supporting": support}


def stability(score_fn, frame: pd.DataFrame, attr_fn, top_k: int = 3, trials: int = 20, eps: float = .02,
              seed: int = 0) -> float:
    """Share of small input perturbations that leave the top adverse reasons unchanged."""
    rng = np.random.default_rng(seed)
    base = [r["feature"] for r in reason_codes(attr_fn(frame))["adverse"][:top_k]]
    same = 0
    num = frame.select_dtypes("number").columns
    for _ in range(trials):
        f2 = frame.copy()
        f2[num] = f2[num] * (1 + rng.uniform(-eps, eps, size=f2[num].shape))
        top = [r["feature"] for r in reason_codes(attr_fn(f2))["adverse"][:top_k]]
        same += int(top == base)
    return round(same / trials, 2)


def disparate_impact(outcomes: pd.DataFrame, group_col: str, positive_col: str) -> dict:
    """Ratio of the lowest group's positive rate to the highest group's rate (four-fifths rule)."""
    rates = outcomes.groupby(group_col)[positive_col].mean()
    counts = outcomes.groupby(group_col).size()
    if len(rates) < 2 or rates.max() == 0:
        return {"ratio": 1.0, "rates": rates.round(3).to_dict(), "counts": counts.to_dict()}
    return {"ratio": round(float(rates.min() / rates.max()), 3), "rates": rates.round(3).to_dict(),
            "counts": counts.to_dict()}


def churn_drivers(attr: dict, k: int = 3) -> list:
    items = sorted(attr["contributions"].items(), key=lambda x: -x[1])
    return [F.CHURN_LABELS[f] for f, v in items if v > 0.1][:k]
