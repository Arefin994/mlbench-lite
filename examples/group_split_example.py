"""Why `groups=` matters: replicate rows leaking across the split.

Each of 60 groups (think: one subject, one machine configuration) contributes
10 near-identical replicate rows, and each group gets a *random* label -- so
there is no real signal and an honest accuracy is ~0.5. A plain random split
puts replicates of the same group in both train and test, and models that can
memorise rows look far better than they are. With `groups=`, every group lands
entirely in train or entirely in test.

    python examples/group_split_example.py
"""
import numpy as np
import pandas as pd

from mlbench_lite import benchmark

rng = np.random.default_rng(0)
n_groups, reps, n_features = 60, 10, 5

centres = rng.normal(size=(n_groups, n_features))
group_labels = rng.integers(0, 2, size=n_groups)  # random: no learnable signal

X = np.repeat(centres, reps, axis=0) + rng.normal(scale=0.01, size=(n_groups * reps, n_features))
y = np.repeat(group_labels, reps)
groups = np.repeat(np.arange(n_groups), reps)

# real data rarely arrives sorted by group -- shuffle the rows
order = rng.permutation(len(y))
X, y, groups = X[order], y[order], groups[order]

models = ["K-Nearest Neighbors", "Random Forest", "Decision Tree", "Logistic Regression"]

naive = benchmark(X, y, cv=5, models=models)
grouped = benchmark(X, y, cv=5, groups=groups, models=models)

table = pd.merge(
    naive[["Model", "Accuracy"]].rename(columns={"Accuracy": "Accuracy (no groups)"}),
    grouped[["Model", "Accuracy"]].rename(columns={"Accuracy": "Accuracy (groups=)"}),
    on="Model",
)
print(f"{n_groups} groups x {reps} replicates, random labels (chance = 0.5), 5-fold CV\n")
print(table.to_string(index=False))
