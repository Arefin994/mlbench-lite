---
title: 'mlbench-lite: A Lightweight Benchmarking Framework for Machine Learning Systems'
tags:
  - Python
  - Machine Learning
  - Benchmarking
  - Performance Measurement
authors:
  - name: Arefin Amin
    orcid: 0000-0000-0000-0000
    affiliation: Independent Researcher
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`mlbench-lite` is a Python library for comparing tabular machine learning models on a single dataset. A user passes a feature matrix `X` and target vector `y` to `benchmark()`, and the library trains every selected model, computes task-appropriate metrics, and returns a sorted `pandas` DataFrame. Classification runs report accuracy, precision, recall, and F1; regression runs report RMSE, MAE, and R².

The library wraps scikit-learn estimators plus optional backends from XGBoost [@chen2016], LightGBM [@ke2017], CatBoost [@prokhorenkova2018], and PyTorch [@paszke2019] for deep tabular models. Model selection is controlled through whitelists (`models=`, `model_categories=`), blocklists (`exclude_models=`), or defaults that include all installed backends. Cross-validation uses scikit-learn's `cross_validate`; single-split evaluation uses `train_test_split`. Group-aware splitting via `GroupKFold` and `GroupShuffleSplit` keeps replicate rows out of both train and test sets.

Target users are researchers and engineers who need a fast baseline sweep during feature engineering or model selection, without standing up a separate benchmarking cluster. The package ships a CLI (`mlbench-lite`), an optional Textual TUI (`mlbench-lite-tui`), and helper functions for scaling recommendations (`recommend_scaling()`), global seeding (`set_seed()`), and post-hoc model comparison with Friedman and Wilcoxon tests (`compare_models()`).

# Statement of Need

Tabular model selection often starts with the same question: which algorithm family works on this dataset before tuning hyperparameters? Full benchmark suites like MLPerf [@mlperf2020] answer a different question. They standardize hardware, datasets, and submission rules for cross-vendor comparison. Running MLPerf locally requires dedicated infrastructure, reference implementations, and compliance with fixed task definitions. Custom profiling scripts solve the immediate problem but tend to copy-paste train/test splits, metric calculations, and model lists across projects.

`mlbench-lite` fills the gap between those extremes. One function call runs 40+ models with consistent splits, metrics, and random seeds. A typical sweep on a few hundred rows finishes in seconds on a laptop CPU. Optional GPU routing (`device="cpu"`, `"gpu"`, or `"auto"`) applies only to XGBoost, LightGBM, CatBoost, and the optional Torch MLP / FT-Transformer models; tree models and most scikit-learn estimators stay on CPU.

The library also handles common tabular pitfalls that ad-hoc scripts skip: label encoding for non-contiguous class labels, per-model scaling pipelines for scale-sensitive estimators only, leakage-safe group splits, and per-model error isolation so one failing estimator does not abort the run. For cross-validated runs, `compare_models()` runs Friedman and pairwise Wilcoxon tests [@friedman1937; @wilcoxon1945] on fold-level scores instead of comparing single-point means.

# High-Level Architecture

The package lives under `mlbench_lite/` with eight modules tied together by `benchmark()` in `benchmark.py`.

`benchmark.py` is the orchestrator. `_detect_task()` inspects the target dtype and cardinality to choose classification or regression. `get_available_classifiers()` and `get_available_regressors()` build a nested dict of `(name, estimator)` tuples grouped by category (Linear, Tree-based, SVM, and so on). Optional imports gate XGBoost, LightGBM, CatBoost, and PyTorch models; missing libraries are skipped without import errors. `_select_models()` applies user filters. For each model, `_maybe_pipeline()` wraps scale-sensitive estimators in a scikit-learn `Pipeline` with a scaler from `preprocessing.py`, and `_apply_device()` sets GPU parameters when requested.

Evaluation follows one of two paths. `_eval_single_split()` fits on a train set and scores on a held-out test set. `_eval_cross_validation()` calls `cross_validate` with task-specific scorers and stores per-fold raw scores in `results.attrs["raw_scores"]` for downstream statistics. Results are assembled into a DataFrame, sorted by the primary metric, and optionally written to CSV or JSON via `save_to=`.

`preprocessing.py` separates read-only diagnostics from transforms. `recommend_scaling()` computes skew and outlier ratios per column and returns a treatment suggestion. `build_scaler()` constructs `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Log1pTransformer`, or a column-wise `ColumnTransformer` for `"auto"` mode.

`stats.py` implements `compare_models()` on the fold scores stored during CV. `utils.py` exports `set_seed()` to synchronize Python, NumPy, and PyTorch RNG state. `deep_models.py` defines scikit-learn-compatible wrappers around a feedforward MLP and an FT-Transformer encoder. `datasets.py` provides `load_clover()`, a synthetic four-class dataset for demos and tests. `cli.py` and `tui.py` expose the same `benchmark()` entry point from the shell.

Execution flow: input arrays → task detection → model registry lookup → optional label encoding → split or CV loop → per-model fit/predict → metric aggregation → sorted DataFrame output.

# State of the Field

Several Python tools compare models on tabular data, but they differ in scope and overhead.

PyCaret [@pycaret2020] is a low-code AutoML framework. Its `compare_models()` function performs internal tuning and returns a leaderboard, but it pulls in a larger dependency tree and orients toward end-to-end pipelines rather than a minimal baseline sweep. LazyPredict [@lazypredict] trains many sklearn-compatible models with default hyperparameters and ranks them quickly, similar in spirit to `mlbench-lite`, but it lacks built-in group-aware splitting, per-column scaling recommendations, GPU routing for gradient boosting libraries, and fold-level significance testing.

Auto-sklearn [@feurer2019] and H2O AutoML [@h2o2023] focus on hyperparameter search and ensemble construction. They produce strong models at the cost of long runtimes and opaque search spaces. `mlbench-lite` deliberately keeps default hyperparameters fixed so runs are reproducible and fast; it is a screening tool, not an optimizer.

MLPerf [@mlperf2020] remains the reference for standardized hardware benchmarking across organizations. It does not replace a local `(X, y)` sweep during notebook iteration.

Relative to a hand-written loop over `sklearn` estimators, `mlbench-lite` adds a maintained model registry across four libraries, consistent metric columns, optional deep tabular baselines, CLI/TUI interfaces, and statistical comparison helpers. The trade-off is less flexibility than a custom script and no hyperparameter search.

# Acknowledgements

The author thanks early users who reported issues with string-labeled targets, model naming across classification and regression tasks, and CSV loading in the CLI. The library depends on scikit-learn [@pedregosa2011], pandas [@pandas2020], NumPy [@harris2020], SciPy [@virtanen2020], and optionally XGBoost [@chen2016], LightGBM [@ke2017], CatBoost [@prokhorenkova2018], and PyTorch [@paszke2019].
