"""Fold-level statistical comparison with compare_models().

Runs the same three classifiers with 5 and with 10 folds to show why the fold
count matters: with 5 folds the smallest attainable two-sided Wilcoxon
p-value is 0.0625, so no pair can ever be flagged at alpha = 0.05.

    python examples/stats_example.py
"""
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlbench_lite import benchmark, compare_models

X, y = load_breast_cancer(return_X_y=True)
models = ["Logistic Regression", "Random Forest", "Gaussian Naive Bayes"]
pd.set_option("display.width", 160)

for cv in (5, 10):
    results = benchmark(X, y, cv=cv, scaling="standard", models=models)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison = compare_models(results, metric="accuracy")

    print(f"=== cv={cv} ===")
    print(f"Friedman p = {comparison['friedman_p_value']:.4g}  "
          f"(significant: {comparison['friedman_significant']})")
    print(f"smallest attainable Bonferroni p = {comparison['min_attainable_p_bonferroni']:.4g}  "
          f"(can reach significance: {comparison['can_reach_significance']})")
    print(comparison["pairwise"][[
        "model_a", "model_b", "mean_a", "mean_b",
        "p_value", "p_value_bonferroni", "significant_at_alpha",
    ]].to_string(index=False))
    for w in caught:
        print(f"warning: {w.message}")
    print()
