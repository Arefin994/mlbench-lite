"""
Statistical comparison between models, built on the per-fold scores that
`benchmark(..., cv=N)` stashes in `results.attrs["raw_scores"]`.

Needs cv >= 2 (i.e. you must call benchmark with cv=N, not a single split) --
a single train/test split gives one number per model, and you can't run a
significance test on a sample size of one.

How to read the output -- these tests are an *exploratory* screen, not formal
inference:

  * The k-fold training sets overlap, so fold scores are not independent
    observations. Friedman/Wilcoxon assume independence, which makes the
    p-values here optimistic (too small). Treat a "significant" result as
    "worth a closer look", not as proof that one model is better.
  * Workflow: Friedman first (are *any* of the >= 3 models different?), then
    pairwise Wilcoxon signed-rank tests as post-hoc comparisons. By default
    (posthoc_only=True) no pair is flagged significant unless Friedman was.
    With exactly 2 models Friedman does not apply and the single Wilcoxon
    test is reported directly.
  * Pairwise p-values are Bonferroni-corrected (p * number_of_pairs). It is
    the simplest familywise correction and errs on the conservative side.
  * Small fold counts limit what is reachable at all: with n folds and no
    ties, the smallest exact two-sided Wilcoxon p-value is 2 / 2**n (0.0625
    at n=5, so nothing can be significant at 0.05 even before correction).
    compare_models() reports this bound and warns when significance is
    unreachable. Use cv=10 or more if you intend to test.
"""
from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

RECOMMENDED_MIN_FOLDS = 5


def _min_folds_needed(n_comparisons: int, alpha: float) -> int:
    """Smallest n with 2 / 2**n * n_comparisons < alpha."""
    return int(math.floor(math.log2(2 * n_comparisons / alpha))) + 1


def compare_models(
    results: pd.DataFrame,
    metric: Optional[str] = None,
    alpha: float = 0.05,
    posthoc_only: bool = True,
) -> Dict:
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

    pairs = list(combinations(range(len(model_names)), 2))
    n_comparisons = len(pairs)
    # exact two-sided Wilcoxon, no zero differences or ties: best case is all
    # n signs agreeing -> p = 2 * (1/2)**n
    min_attainable_p = 2.0 / 2 ** min_folds
    min_attainable_p_bonferroni = min(min_attainable_p * n_comparisons, 1.0)
    can_reach_significance = min_attainable_p_bonferroni < alpha

    if min_folds < RECOMMENDED_MIN_FOLDS:
        warnings.warn(
            f"Only {min_folds} CV folds per model. Fold-level tests on this few "
            f"observations carry very little information; use cv >= 10 if you "
            "intend to test for significance.",
            UserWarning,
            stacklevel=2,
        )
    if not can_reach_significance:
        warnings.warn(
            f"With {min_folds} folds and {n_comparisons} pairwise comparison(s), the "
            f"smallest attainable Bonferroni-corrected Wilcoxon p-value is "
            f"{min_attainable_p_bonferroni:.4g} >= alpha={alpha}, so no pair can be "
            f"flagged significant. Use at least "
            f"{_min_folds_needed(n_comparisons, alpha)} folds, or compare fewer models.",
            UserWarning,
            stacklevel=2,
        )

    friedman_applies = len(model_names) > 2
    friedman_stat, friedman_p = scipy_stats.friedmanchisquare(*score_matrix) if friedman_applies \
        else (None, None)
    friedman_significant = friedman_p is not None and friedman_p < alpha
    gated = posthoc_only and friedman_applies and not friedman_significant

    pairwise_rows = []
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
            "significant_at_alpha": bool(p_bonferroni < alpha and not gated),
            "gated_by_friedman": gated,
        })
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("p_value")

    if friedman_applies:
        workflow = (
            "Friedman test checks whether *any* model differs from the others across "
            "folds; the pairwise Wilcoxon tests are post-hoc comparisons"
            + (" and are only flagged significant when Friedman is." if posthoc_only
               else " (posthoc_only=False: pairs are flagged regardless of Friedman).")
        )
    else:
        workflow = (
            "Only 2 models, so the Friedman test does not apply; the single pairwise "
            "Wilcoxon test is reported directly."
        )
    note = (
        f"{workflow} p_value_bonferroni corrects for the {n_comparisons} pairwise "
        "test(s) -- use it, not the raw p_value. These tests are exploratory: CV "
        "training sets overlap, so fold scores are not independent and p-values are "
        f"optimistic. With {min_folds} folds the smallest attainable corrected p-value "
        f"is {min_attainable_p_bonferroni:.4g}."
    )

    return {
        "metric": metric,
        "n_folds": min_folds,
        "n_models": len(model_names),
        "n_comparisons": n_comparisons,
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "friedman_significant": friedman_significant,
        "min_attainable_p": min_attainable_p,
        "min_attainable_p_bonferroni": min_attainable_p_bonferroni,
        "can_reach_significance": can_reach_significance,
        "pairwise": pairwise_df,
        "note": note,
    }
