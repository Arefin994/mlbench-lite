"""Regression benchmark: 5-fold CV on scikit-learn's diabetes dataset.

Reports RMSE, MAE and R2 (mean over folds, with the fold std alongside).

    python examples/regression_example.py
"""
import pandas as pd
from sklearn.datasets import load_diabetes

from mlbench_lite import benchmark

X, y = load_diabetes(return_X_y=True, as_frame=True)

results = benchmark(
    X,
    y,
    task="regression",
    cv=5,
    scaling="standard",
    models=[
        "Linear Regression",
        "Ridge",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "SVR",
        "K-Nearest Neighbors Regressor",
    ],
)

pd.set_option("display.width", 140)
print(f"diabetes: {X.shape[0]} rows, {X.shape[1]} features, 5-fold CV\n")
print(results[["Model", "RMSE", "RMSE_std", "MAE", "MAE_std", "R2", "R2_std", "Train Time (s)"]]
      .to_string(index=False))
