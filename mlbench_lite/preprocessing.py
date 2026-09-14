"""
Scaling / normalization utilities.

Two separate concerns, kept deliberately separate:

  * `recommend_scaling(X)`  -- READ ONLY. Inspects each column and reports what
     it would suggest. Never modifies X. Safe to call any time.

  * `build_scaler(X, mode)` -- Actually builds a transformer. Only called when
     the caller explicitly asks for it (either a fixed mode like "standard",
     or "auto", which applies exactly what `recommend_scaling` reported).

Nothing here silently changes user data. `benchmark()` only applies scaling
when the caller sets `scaling=...` (or the legacy `use_scaling=True`).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

# Techniques considered and deliberately NOT implemented here:
#
#   Batch Normalization / Layer Normalization
#   -> These are neural-network *layers* that normalize activations during
#      training (BatchNorm: across a mini-batch, LayerNorm: across the
#      features of a single sample, both with learnable scale/shift
#      parameters). They are not a static preprocessing step you apply to a
#      table before handing it to a model, and they don't mean anything for
#      the majority of models in this library (Random Forest, SVM, KNN,
#      boosting, etc. have no notion of "layers" or "batches" to normalize).
#      Implementing a real, learnable version would mean building an actual
#      neural net training loop (e.g. PyTorch), which is out of scope for a
#      scikit-learn-based benchmarking library. Not added.

SKEW_THRESHOLD = 1.0
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_RATIO_THRESHOLD = 0.05  # >5% of rows flagged as outliers -> "has outliers"


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """log1p transform. Requires non-negative input; falls back to a shift
    if negative values are present so it never raises on fit/transform."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.shift_ = 0.0
        min_val = np.nanmin(X) if X.size else 0.0
        if min_val < 0:
            self.shift_ = -min_val
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(X + self.shift_)


def _column_diagnostics(series: pd.Series) -> Dict:
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return {
            "skew": 0.0, "outlier_ratio": 0.0, "min": float(clean.min()) if len(clean) else 0.0,
            "max": float(clean.max()) if len(clean) else 0.0, "has_negative": False,
        }
    skew = float(scipy_stats.skew(clean))
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        outlier_ratio = 0.0
    else:
        lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        outlier_ratio = float(((clean < lo) | (clean > hi)).mean())
    return {
        "skew": skew,
        "outlier_ratio": outlier_ratio,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "has_negative": bool(clean.min() < 0),
    }


def recommend_scaling(X) -> pd.DataFrame:
    """Inspect each numeric column and report a suggested treatment.

    Does NOT modify X. Returns a DataFrame with one row per column:
    skew, outlier_ratio, recommendation, reason.
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
        np.asarray(X), columns=[f"col_{i}" for i in range(np.asarray(X).shape[1])]
    )
    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            rows.append({"column": col, "skew": None, "outlier_ratio": None,
                         "recommendation": "none", "reason": "non-numeric column, skipped"})
            continue
        diag = _column_diagnostics(df[col])
        skewed = abs(diag["skew"]) > SKEW_THRESHOLD
        has_outliers = diag["outlier_ratio"] > OUTLIER_RATIO_THRESHOLD

        if skewed and not diag["has_negative"]:
            rec, reason = "log", (
                f"skew={diag['skew']:.2f} (heavy-tailed), all values >= 0 "
                "-> log1p to compress the tail, then scale"
            )
        elif has_outliers:
            rec, reason = "robust", (
                f"{diag['outlier_ratio']*100:.1f}% of rows flagged as outliers "
                "(IQR rule) -> use median/IQR based scaling, less sensitive than mean/std"
            )
        elif 0 <= diag["min"] and diag["max"] <= 1:
            rec, reason = "none", "already roughly in [0, 1], scaling unlikely to help"
        else:
            rec, reason = "standard", "no strong skew or outliers -> plain standardization is fine"

        rows.append({
            "column": col, "skew": round(diag["skew"], 3),
            "outlier_ratio": round(diag["outlier_ratio"], 3),
            "recommendation": rec, "reason": reason,
        })
    return pd.DataFrame(rows)


def _simple_scaler(mode: str):
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "log": Pipeline([("log1p", Log1pTransformer()), ("scale", StandardScaler())]),
        "l2": Normalizer(norm="l2"),  # row-wise, not column-wise -- see README caveat
    }[mode]


def build_scaler(X, mode: str):
    """Build a transformer for the requested mode.

    mode: "standard" | "minmax" | "robust" | "log" | "l2" | "auto" | None
    "auto" builds a per-column ColumnTransformer from recommend_scaling(X).
    Never called unless the caller explicitly asked for scaling.
    """
    if mode is None or mode == "none":
        return None
    if mode != "auto":
        return _simple_scaler(mode)

    report = recommend_scaling(X)
    # Use positional indices as the ColumnTransformer selector, not column names --
    # names only work when the exact same DataFrame (same dtype/columns) is passed
    # at both fit and transform time. Positions work for both DataFrames and plain
    # ndarrays, which matters since callers can pass either to benchmark().
    n_cols = len(report)
    transformers = []
    passthrough_positions = []
    for pos, (_, row) in enumerate(report.iterrows()):
        rec = row["recommendation"]
        if rec == "none":
            passthrough_positions.append(pos)
            continue
        elif rec == "log":
            step = Pipeline([("log1p", Log1pTransformer()), ("scale", RobustScaler())])
        elif rec == "robust":
            step = RobustScaler()
        else:  # "standard"
            step = StandardScaler()
        transformers.append((f"col{pos}_{rec}", step, [pos]))
    if not transformers:
        return None
    return ColumnTransformer(
        transformers, remainder="passthrough" if passthrough_positions else "drop"
    )
