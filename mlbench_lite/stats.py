"""
Statistical comparison between models, built on the per-fold scores that
`benchmark(..., cv=N)` stashes in `results.attrs["raw_scores"]`.

Needs cv >= 2 (i.e. you must call benchmark with cv=N, not a single split) --
a single train/test split gives one number per model, and you can't run a
significance test on a sample size of one.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compare_models(results: pd.DataFrame, metric: Optional[str] = None, alpha: float = 0.05) -> Dict:
    raw = results.attrs.get("raw_scores")
    if not raw:
        raise ValueError(
            "No per-fold scores found on this results DataFrame. "
            "compare_models() needs benchmark(..., cv=N) with N >= 2 -- "
            "a single train/test split only gives one score per model."
        )
    if metric is None:
        metric = "r2" if any("r2" in v for v in raw.values()) else "accuracy"

    model_names = [m for m in raw if metric in raw[m]]
    if len(model_names) < 2:
        raise ValueError(f"Need at least 2 models with '{metric}' scores to compare.")

    fold_counts = {m: len(raw[m][metric]) for m in model_names}
    min_folds = min(fold_counts.values())
    if min_folds < 2:
        raise ValueError("Need at least 2 CV folds per model to run significance tests.")
    if len(set(fold_counts.values())) > 1:
        model_names = [m for m in model_names if fold_counts[m] == min_folds]

    score_matrix = np.array([raw[m][metric][:min_folds] for m in model_names])  # (n_models, n_folds)

    friedman_stat, friedman_p = scipy_stats.friedmanchisquare(*score_matrix) if len(model_names) > 2 \
        else (None, None)

    pairwise_rows = []
    pairs = list(combinations(range(len(model_names)), 2))
    n_comparisons = len(pairs)
    for i, j in pairs:
        a, b = score_matrix[i], score_matrix[j]
        if np.allclose(a, b):
            stat, p = np.nan, 1.0
        else:
            stat, p = scipy_stats.wilcoxon(a, b)
        p_bonferroni = min(p * n_comparisons, 1.0)
        pairwise_rows.append({
            "model_a": model_names[i], "model_b": model_names[j],
            "mean_a": round(float(a.mean()), 4), "mean_b": round(float(b.mean()), 4),
            "wilcoxon_stat": stat, "p_value": round(float(p), 5),
            "p_value_bonferroni": round(float(p_bonferroni), 5),
            "significant_at_alpha": bool(p_bonferroni < alpha),
        })
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("p_value")

    return {
        "metric": metric,
        "n_folds": min_folds,
        "n_models": len(model_names),
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "friedman_significant": (friedman_p is not None and friedman_p < alpha),
        "pairwise": pairwise_df,
        "note": (
            "Friedman test checks whether *any* model differs from the others "
            "across folds; a significant Friedman result is what justifies looking "
            "at the pairwise Wilcoxon table below. p_value_bonferroni corrects for "
            "running multiple pairwise tests -- use that column, not the raw p_value, "
            "when deciding significance."
        ),
    }

