# mlbench-lite

A comprehensive machine learning benchmarking library that provides an easy way to compare multiple ML models on your dataset classification **and regression**. Built with scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, and pandas for seamless integration into your ML workflow.

## 🚀 Features

- **Comprehensive Model Support**: 40+ ML models across classification and regression, from multiple libraries, plus optional deep tabular models (Torch MLP, FT-Transformer)
- **Classification & Regression**: auto-detected, or force it explicitly with `task=`
- **Flexible Model Selection**: choose specific models, categories, or exclude models
- **Cross-Validation**: single split or k-fold, with **group-aware splitting** so replicate/near-duplicate rows never leak across train and test
- **Scaling, on your terms**: a read-only column-by-column scaling *recommendation* report, plus opt-in scaling (standard / minmax / robust / log / row-norm / auto) nothing is transformed unless you ask for it
- **GPU Support**: `device="cpu"` / `"gpu"` / `"auto"` for XGBoost / LightGBM / CatBoost and the optional Deep Tabular models  `"auto"` picks a GPU if one's available and falls back to CPU otherwise
- **Reproducibility, fully**: `set_seed()` seeds Python's `random`, NumPy, and (if installed) PyTorch/CUDA in one call  `benchmark()` already does this for you internally using `random_state`
- **Statistical Significance Testing**: `compare_models()`  Friedman + pairwise Wilcoxon tests across CV folds, not just eyeballing which mean is bigger
- **Multiple ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (optional)
- **Simple API**: one function call to benchmark multiple models
- **Comprehensive Metrics**: Accuracy/Precision/Recall/F1 for classification, RMSE/MAE/R² for regression
- **Custom Dataset**: includes the `load_clover` dataset for testing
- **Pandas Output**: results returned as a clean pandas DataFrame, optionally saved straight to CSV/JSON
- **Reproducible**: consistent results with random state control
- **CLI**: everything above is also available from the command line
- **Interactive TUI**: pick your dataset, models, and options with the keyboard or mouse  no flags to remember (`mlbench-lite-tui`, optional)
- **Model Information**: get detailed info about available models, for both classification and regression

## 📦 Installation

```bash
pip install mlbench-lite
```

Two extras add optional functionality  neither is required for the rest of the library, and everything above works identically without them:

```bash
# Deep tabular models (Torch MLP, FT-Transformer) -- needs PyTorch
pip install mlbench-lite[deep]

# Interactive keyboard/mouse TUI -- needs Textual
pip install mlbench-lite[tui]

# Both
pip install mlbench-lite[all]
```

## 🎯 Quick Start

```python
from mlbench_lite import benchmark, load_clover

# Load the clover dataset
X, y = load_clover(return_X_y=True)

# Benchmark all available models
results = benchmark(X, y)
print(results)
```

**Output:**
```
                 Model           Category  Accuracy  Precision  Recall      F1
0        Random Forest  Tree-based Models    0.9500     0.9565  0.9512  0.9505
1                  SVM        SVM Models    0.9250     0.9337  0.9255  0.9254
2  Logistic Regression    Linear Models    0.9125     0.9131  0.9117  0.9115
3              XGBoost           XGBoost    0.9000     0.9024  0.9000  0.8997
4            LightGBM          LightGBM    0.8875     0.8891  0.8875  0.8873
```

Regression works the same way  just point it at a continuous target:

```python
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)

results = benchmark(X, y, task="regression")
print(results)   # RMSE / MAE / R2 columns instead of Accuracy / Precision / Recall / F1
```

## 📚 API Reference

### `benchmark(X, y, test_size=0.2, random_state=42, models=None, model_categories=None, exclude_models=None, cv=None, groups=None, task=None, use_scaling=False, scaling=None, device="cpu", save_to=None, show_std=True)`

Benchmark multiple machine learning models on a dataset.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features)
- `y` (array-like): Target values of shape (n_samples,)
- `test_size` (float, optional): Proportion of dataset for testing when `cv` is not set (default: 0.2)
- `random_state` (int, optional): Random seed for reproducibility (default: 42)
- `models` (list of str, optional): Specific models to use. If None, uses all available models for the detected/given task.
- `model_categories` (list of str, optional): Categories of models to use. If None, uses all categories.
- `exclude_models` (list of str, optional): Models to exclude from benchmarking.
- `cv` (int, optional): Number of cross-validation folds. If None, uses a single `test_size` train/test split.
- `groups` (array-like, optional): Group label per row (e.g. a shared config/subject/source id). When set, splitting uses `GroupShuffleSplit`/`GroupKFold` so rows sharing a group never end up split across train and test  prevents near-duplicate rows from inflating your score. See "Why groups matter" below.
- `task` ("classification" | "regression" | None, optional): Force the task instead of relying on auto-detection. Auto-detection warns when it's guessing on an ambiguous case (an integer-dtype target with many unique values)  pass this explicitly if you see that warning.
- `use_scaling` (bool, optional): Legacy flag, equivalent to `scaling="standard"`. Kept for backward compatibility.
- `scaling` ("standard" | "minmax" | "robust" | "log" | "l2" | "auto" | None, optional): Only applied to models known to be scale-sensitive (tree/boosting models are always left raw). `"auto"` inspects each column and picks a treatment per column  see `recommend_scaling()` below. Nothing is scaled unless this (or `use_scaling`) is explicitly set.
`RandomizedSearchCV`. Models without a defined grid still run, with defaults, unchanged.
- `device` ("cpu" | "gpu" | "auto", optional, default: "cpu"): Affects XGBoost/LightGBM/CatBoost and the optional Deep Tabular (torch) models. `"auto"` checks for an available GPU and picks accordingly; `"gpu"` attempts GPU regardless and falls back to CPU with a warning if it can't be set (same as before). Models without GPU support are unaffected either way.
- `save_to` (str, optional): Path to write results to  `.csv` or `.json`, inferred from the extension.
- `show_std` (bool, optional): Show `_std` columns when using `cv` (default: True).
- `progress_callback` (callable, optional): `callback(completed, total, model_name, status)`, called once after each model finishes (`status` is `"done"` or `"error"`). Meant for UIs  the TUI's progress bar is driven by this  has no effect on results if left as `None`.
- `cancel_check` (callable, optional): `callback() -> bool`, polled before each model starts (and, for the optional Deep Tabular torch models, between epochs too). If it returns `True`, the run stops early and returns whatever results were already gathered, with `results_df.attrs["cancelled"] = True`, instead of raising or losing the partial results. Meant for a "Cancel" button  the TUI uses this  has no effect if left as `None`.

**Reproducibility note:** `benchmark()` seeds Python's `random`, NumPy, and (if installed) PyTorch/CUDA once at the start of every run, using `random_state`  on top of passing `random_state` to every individual estimator/split as before. So a full run, sklearn models and Deep Tabular models together, is reproducible end to end given the same `random_state`.

**Returns:**
- `pandas.DataFrame`: Results, sorted best-first. For classification: `Model`, `Category`, `Accuracy`, `Precision`, `Recall`, `F1` (+ `_std` variants when `cv` is set). For regression: `Model`, `Category`, `RMSE`, `MAE`, `R2` (+ `_std` variants when `cv` is set). Always includes `Train Time (s)`, and an `Error` column for any model that failed (results for other models are unaffected).
  - `results.attrs["raw_scores"]`: per-fold scores (only populated when `cv` is set)  feed this straight into `compare_models()`.
  - `results.attrs["task"]`: `"classification"` or `"regression"`, whichever was detected/used.

### `recommend_scaling(X)`

Inspects each numeric column (skew, outlier ratio, bounds) and returns a report of what scaling treatment it would suggest. **Read-only  never modifies X.** Safe to call any time, independently of `benchmark()`.

**Returns:**
- `pandas.DataFrame`: one row per column  `column`, `skew`, `outlier_ratio`, `recommendation`, `reason`.

```python
from mlbench_lite import recommend_scaling
report = recommend_scaling(X)
print(report)
```

### `set_seed(seed, deterministic=False)`

Seeds Python's `random`, NumPy, and (if installed) PyTorch/CUDA in one call. `benchmark()` already calls this internally using `random_state`  use it directly if you want the rest of your script/notebook seeded the same way *before* calling `benchmark()`, e.g. if you're generating your own train/test split first.

**Parameters:**
- `seed` (int): The seed to use everywhere.
- `deterministic` (bool, optional, default: False): If True and PyTorch is installed, also forces cuDNN into deterministic mode. Can noticeably slow down GPU training, so it's opt-in  most people training on CPU/Colab don't need it.

```python
from mlbench_lite import set_seed

set_seed(42)
# ... your own data prep / model code, now reproducible too
```

### `compare_models(results, metric=None, alpha=0.05)`

Statistical significance testing between models, using the per-fold scores from a `cv=`-based `benchmark()` run. Requires `cv >= 2`  a single train/test split gives one number per model, which isn't enough to test significance on.

**Parameters:**
- `results` (pandas.DataFrame): output of `benchmark(..., cv=N)`
- `metric` (str, optional): which metric to compare on (defaults to `r2` for regression, `accuracy` for classification)
- `alpha` (float, optional): significance threshold (default: 0.05)

**Returns:**
- `dict`: `friedman_statistic`, `friedman_p_value`, `friedman_significant`, and a `pairwise` DataFrame with per-pair means, Wilcoxon p-values, Bonferroni-corrected p-values, and a `significant_at_alpha` flag.

```python
from mlbench_lite import benchmark, compare_models

results = benchmark(X, y, cv=5, models=["Random Forest", "Ridge", "SVR"], task="regression")
comparison = compare_models(results)
print(comparison["friedman_p_value"])
print(comparison["pairwise"])
```

### `list_available_models(task="classification")`

List all available models and their categories, for either task.

**Returns:**
- `dict`: Dictionary with model categories as keys and lists of model names as values

### `get_model_info()`

Get detailed information about available models (both classification and regression).

**Returns:**
- `pandas.DataFrame`: DataFrame with model information including category, name, and description

### `load_clover(return_X_y=False)`

Load the custom clover dataset.

**Parameters:**
- `return_X_y` (bool, default=False): If True, returns (data, target) instead of a Bunch object

**Returns:**
- `Bunch` or `tuple`: Dataset object with data, target, feature_names, target_names, and DESCR

## 💡 Code Examples

### 1. Basic Usage with All Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

results = benchmark(X, y)
print("\nBenchmark Results:")
print(results)

best_model = results.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")
```

### 2. Model Selection - Specific Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)
results = benchmark(X, y, models=['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression'])
print(results)
```

### 3. Model Selection - By Categories

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)
results = benchmark(X, y, model_categories=['Tree-based Models'])
print(results)

results = benchmark(X, y, model_categories=['Linear Models', 'SVM Models'])
print(results)
```

### 4. Exclude Specific Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)
results = benchmark(X, y, exclude_models=['Gaussian Process', 'Multi-layer Perceptron'])
print(results)
```

### 5. List Available Models

```python
from mlbench_lite import list_available_models, get_model_info

for category, model_list in list_available_models().items():
    print(f"\n{category}:")
    for model in model_list:
        print(f"  - {model}")

# Regression models, same call, different task
for category, model_list in list_available_models(task="regression").items():
    print(f"\n{category}:")
    for model in model_list:
        print(f"  - {model}")

print(get_model_info())
```

### 6. Cross-Validation Instead of a Single Split

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# 5-fold CV instead of one 80/20 split -- gives you a mean and a std per model
results = benchmark(X, y, cv=5)
print(results)
```

### 7. Regression

```python
from mlbench_lite import benchmark
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

# task="regression" isn't strictly required here (a float target auto-detects fine),
# but it's the safer thing to pass explicitly whenever you're not sure
results = benchmark(X, y, task="regression", cv=5)
print(results)
```

### 8. Why `groups` Matters (Preventing Leakage)

If some of your rows aren't fully independent  replicate samples, several rows per subject, augmented copies of the same source record  a plain random split can put near-duplicates on both sides of train/test. The model then "recognizes" its own near-duplicate at test time instead of genuinely generalizing, and your score is inflated.

```python
from mlbench_lite import benchmark

# groups: an array-like the same length as X/y, where rows sharing a group
# id are guaranteed to land entirely in train OR entirely in test, never split
groups = df["config_id"]

# without groups= -- near-duplicates can leak across the split
results_naive = benchmark(X, y, cv=5, task="regression")

# with groups= -- leakage-safe, a more honest number
results_grouped = benchmark(X, y, cv=5, groups=groups, task="regression")

print(results_naive[["Model", "R2"]])
print(results_grouped[["Model", "R2"]])   # expect this to be lower, and more trustworthy
```

### 9. Scaling  Recommendation First, Then (Optionally) Apply

```python
from mlbench_lite import benchmark, recommend_scaling

# Look before you leap: this never modifies X
report = recommend_scaling(X)
print(report)

# Apply exactly what the report suggested, per column
results = benchmark(X, y, scaling="auto")

# Or force one mode for everything scale-sensitive:
results = benchmark(X, y, scaling="robust")

# Legacy flag still works, equivalent to scaling="standard"
results = benchmark(X, y, use_scaling=True)
```

### 10. GPU Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Affects XGBoost / LightGBM / CatBoost and the optional Deep Tabular models.
# "auto" picks a GPU if one's available, otherwise CPU -- the safest default
# for scripts you'll run in different places (your laptop, Colab, a server).
results = benchmark(X, y, models=["XGBoost", "LightGBM", "CatBoost"], device="auto")
print(results)

# "gpu" still works exactly as before: attempts GPU regardless, falls back to
# CPU with a warning if the device can't be set (e.g. no GPU runtime available).
results = benchmark(X, y, models=["XGBoost"], device="gpu")
```

### 10b. Deep Tabular Models (optional  `pip install mlbench-lite[deep]`)

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Torch MLP: a plain feedforward PyTorch network -- a deep-learning-native
# baseline next to sklearn's MLPClassifier.
# FT-Transformer: feature-tokenizer + a small Transformer encoder, one of the
# strongest general-purpose architectures for tabular data specifically.
results = benchmark(
    X, y,
    models=["Torch MLP", "FT-Transformer", "Random Forest"],
    scaling="standard",   # both deep models are scale-sensitive, same as MLP/SVM
    device="auto",        # uses a GPU if available, otherwise CPU
)
print(results)
```

Both are sized to train quickly on modest hardware  a few seconds to low
tens of seconds on a free Colab CPU runtime for typical benchmark-sized
datasets, faster still on a T4. If `torch` isn't installed, these two models
simply don't appear in `list_available_models()` / `benchmark()`  nothing
else changes.

Regression works the same way, with `"Torch MLP Regressor"` / `"FT-Transformer Regressor"`:

```python
from mlbench_lite import benchmark
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
results = benchmark(X, y, task="regression",
                     models=["Torch MLP Regressor", "FT-Transformer Regressor"],
                     scaling="standard")
print(results)
```

### 10c. Progress Reporting & Cancellation (for your own UI/scripts)

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

def on_progress(completed, total, model_name, status):
    print(f"[{completed}/{total}] {model_name}: {status}")

# Stop after some condition -- e.g. a time budget, a UI cancel button, etc.
import time
start = time.time()
def should_cancel():
    return time.time() - start > 30  # bail after 30 seconds

results = benchmark(
    X, y,
    progress_callback=on_progress,
    cancel_check=should_cancel,
)
if results.attrs.get("cancelled"):
    print(f"Stopped early -- {len(results)} model(s) finished before cancellation.")
```

This is exactly what the TUI's progress bar and Cancel button are built on
underneath -- see below.

### 11. Is Model A Actually Better Than Model B?

```python
from mlbench_lite import benchmark, compare_models, load_clover

X, y = load_clover(return_X_y=True)

results = benchmark(X, y, cv=5, models=["Random Forest", "Logistic Regression", "SVM (RBF)"])
comparison = compare_models(results)

print("Any model significantly different from the others?", comparison["friedman_significant"])
print(comparison["pairwise"])
```

### 12. Saving Results

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)
benchmark(X, y, save_to="results.csv")
benchmark(X, y, save_to="results.json")
```

### 13. Using with Scikit-learn Datasets

```python
from mlbench_lite import benchmark
from sklearn.datasets import load_wine, load_breast_cancer

X, y = load_wine(return_X_y=True)
print(benchmark(X, y))

X, y = load_breast_cancer(return_X_y=True)
print(benchmark(X, y))
```

### 14. Reproducible Results

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

results1 = benchmark(X, y, random_state=123)
results2 = benchmark(X, y, random_state=123)
print(f"Results are identical: {results1.drop(columns=['Train Time (s)']).equals(results2.drop(columns=['Train Time (s)']))}")
# (Train Time (s) is wall-clock timing and will naturally differ run to run --
# exclude it when checking for reproducibility, everything else is deterministic.)
```

## 🔬 Models Included

**40+ models** across classification and regression, all using default parameters with appropriate random seeds for reproducibility.

### Classification

| Category | Models |
|---|---|
| **Linear Models** | Logistic Regression, Ridge Classifier, SGD Classifier, Perceptron, Passive Aggressive |
| **Tree-based Models** | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging Classifier |
| **SVM Models** | SVM (RBF), SVM (Linear) |
| **Neighbors** | K-Nearest Neighbors |
| **Naive Bayes** | Gaussian, Multinomial, Bernoulli |
| **Discriminant Analysis** | Linear (LDA), Quadratic (QDA) |
| **Neural Networks** | Multi-layer Perceptron |
| **Gaussian Process** | Gaussian Process Classifier |
| **Advanced Boosting** | XGBoost, LightGBM, CatBoost (if installed) |
| **Deep Tabular** | Torch MLP, FT-Transformer (if `torch` is installed  `pip install mlbench-lite[deep]`) |

### Regression

| Category | Models |
|---|---|
| **Linear Models** | Linear Regression, Ridge |
| **Tree-based Models** | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging Regressor |
| **SVM Models** | SVR |
| **Neighbors** | K-Nearest Neighbors Regressor |
| **Neural Networks** | MLP Regressor |
| **Gaussian Process** | Gaussian Process Regressor |
| **Advanced Boosting** | XGBoost, LightGBM, CatBoost Regressors (if installed) |
| **Deep Tabular** | Torch MLP Regressor, FT-Transformer Regressor (if `torch` is installed) |

Non-contiguous integer classification targets (e.g. labels that skip values) are handled automatically  `benchmark()` label-encodes the target once up front, which is also what makes XGBoost's classifier work reliably regardless of your label values.

## 🧹 Scaling Modes

| Mode | What it does | Good fit for |
|---|---|---|
| `"standard"` | Zero mean, unit variance | General default, no strong skew/outliers |
| `"minmax"` | Rescale to [0, 1] | Bounded features, neural nets |
| `"robust"` | Median/IQR based | Data with outliers |
| `"log"` | log1p, then scale | Heavy-tailed / long-tail count data |
| `"l2"` | Row-wise unit-norm (`Normalizer`) | Text/TF-IDF-style data, cosine-distance use cases  **situational**, not auto-selected, since it mixes units across columns and tends to hurt on typical mixed-unit tabular data |
| `"auto"` | Picks per-column from `recommend_scaling()` | When you don't want to decide column by column |
| `None` (default) | Off | When your models don't need it (e.g. all tree-based) |

Only applied to scale-sensitive models (linear/SVM/KNN/MLP-type models, including the optional Torch MLP / FT-Transformer)  tree and boosting models are always left raw, since scaling doesn't change how they split.

Two normalization types were deliberately **not** added: Batch Normalization and Layer Normalization. Both are neural-network *layers* that normalize activations during training with learnable parameters  not a static preprocessing step applicable to a table before fitting arbitrary scikit-learn-style models, and meaningless for the majority of models this library benchmarks (Random Forest, SVM, KNN, boosting, etc.).

### A note on scope: why Deep Tabular and not GNNs/CNNs/PINNs/SNNs

`benchmark()` is built around one contract: a flat `(X, y)` table, the same shape every sklearn-style model in this library already expects. Torch MLP and FT-Transformer fit that contract directly, so they slot in cleanly. Graph models (GCN/GraphSAGE/GAT), image models (CNN), sequence models (BiLSTM), and physics-informed/spiking nets (PINN/SNN) all need a fundamentally different input  graph structure, image grids, sequences, or a domain-specific loss  which would mean a second data pipeline bolted onto a library that's currently just about tabular benchmarking. GNNs in particular also carry a notoriously fragile dependency (`torch-geometric`, tightly version-locked to specific torch/CUDA builds) that tends to break on exactly the low-resource/Colab setups this library targets. These may show up as a separate, clearly-labeled module down the line, but they're intentionally out of scope for `benchmark()` today.

## 📊 Clover Dataset Details

The `load_clover` function provides a custom synthetic dataset:

- **Samples**: 400
- **Features**: 4
- **Classes**: 4

**Features:**
- `leaf_length`: Length of the leaf in cm
- `leaf_width`: Width of the leaf in cm
- `petiole_length`: Length of the petiole in cm
- `leaflet_count`: Number of leaflets per leaf

**Classes:**
- `white_clover`: Trifolium repens
- `red_clover`: Trifolium pratense
- `crimson_clover`: Trifolium incarnatum
- `alsike_clover`: Trifolium hybridum

## 🖥️ Command Line Interface

```bash
# Basic run on the built-in clover dataset
mlbench-lite

# Cross-validation on a different built-in dataset
mlbench-lite --dataset iris --cv 5

# Regression (uses the diabetes dataset)
mlbench-lite --task regression

# Specific models, with scaling
mlbench-lite --models "Random Forest" "SVM (RBF)" --scaling

# Full scaling control + a look at what it recommends first
mlbench-lite --task regression --scaling-mode auto --show-recommendations

# Auto-detect GPU (falls back to CPU if none available) -- works for
# boosting libraries and the optional Deep Tabular models alike
mlbench-lite --device auto --categories "Deep Tabular"

# Your own dataset instead of a built-in one -- target defaults to the last column
mlbench-lite --data my_data.csv --target label

# Same, but only using specific feature columns
mlbench-lite --data my_data.csv --target label --feature-columns col_a col_b col_c

# See every available model name
mlbench-lite --list-models
```

| Flag | Purpose |
|---|---|
| `--dataset {clover,iris,wine,breast_cancer}` | Built-in dataset to run on (ignored if `--data` is set) |
| `--data PATH` | Your own CSV/TSV file instead of a built-in dataset |
| `--target COLUMN` | Target column in `--data` (default: the last column) |
| `--feature-columns COL [COL ...]` | Specific feature columns from `--data` (default: every column except `--target`) |
| `--task {classification,regression}` | Force task type (regression uses the diabetes dataset when no `--data` is given) |
| `--cv N` | k-fold cross-validation instead of a single split |
| `--test-size FRAC` | Test fraction when not using `--cv` |
| `--scaling` | Shorthand for `--scaling-mode standard` |
| `--scaling-mode {standard,minmax,robust,log,l2,auto}` | Full scaling control |
| `--show-recommendations` | Print `recommend_scaling()`'s report before running (read-only) |
| `--device {cpu,gpu,auto}` | GPU routing for XGBoost/LightGBM/CatBoost and the optional Deep Tabular models. `auto` uses a GPU if available, otherwise CPU. |
| `--save-to PATH` | Save results to `.csv`/`.json` |
| `--no-std` | Hide `_std` columns in CV output |
| `--models NAME [NAME ...]` | Whitelist specific models |
| `--categories CAT [CAT ...]` | Whitelist specific categories |
| `--exclude NAME [NAME ...]` | Exclude specific models |
| `--random-state SEED` | Random seed |
| `--list-models` | Print all available model names and exit |

> **Note:** classification and regression models have different names (e.g. `"Random Forest"` vs `"Random Forest Regressor"`). If `--models`/`--categories` produces "No results produced", run `mlbench-lite --list-models --task <classification\|regression>` to see the names valid for that task  the CLI's error message points you here automatically.

## 🖱️ Interactive TUI (optional  `pip install mlbench-lite[tui]`)

Prefer picking things with the keyboard or mouse instead of remembering flags? Launch the interactive TUI:

```bash
mlbench-lite-tui
```

It's a separate entry point from the flag-based `mlbench-lite` command (both can be installed at once  `pip install mlbench-lite[all]`), and calls the exact same `benchmark()` / `recommend_scaling()` underneath, so anything that works from the Python API or the CLI works here too:

1. **Dataset source**  built-in demo dataset, or your own CSV/TSV file
   - For a built-in dataset: pick one with arrow keys or a click, same as before
   - For your own file: set the source to "My own CSV/TSV file", type or paste the path, click **"Load columns"**  a **target column (y)** picker appears (defaulting to the last column) and a **feature column (X)** checklist appears (defaulting to everything else, all selected). Changing the target column automatically removes it from the feature list.
2. **Task**  auto-detect, or force classification/regression (the model list below updates to match)
3. **Models**  a checkbox list, grouped by category; toggle with `Space` or a click, all selected by default
4. **Scaling recommendation**  click **"Show scaling recommendation for this dataset"** any time to see `recommend_scaling()`'s per-column report inline (works against whichever dataset  built-in or your CSV  is currently selected). Purely informational, same as the CLI's `--show-recommendations`; pick a Scaling mode below to actually apply something.
5. **Options**  CV folds, scaling mode, device (`auto`/`cpu`/`gpu`), random seed
6. **Run**  press `r` or click "Run Benchmark"; a status line and progress bar update live as each model finishes (or errors), showing e.g. `[3/12] ✓ Random Forest (done)`
7. **Cancel**  press `c` or click "Cancel" at any point during a run to stop gracefully. It finishes whatever model is currently training (Deep Tabular models can also stop mid-training, between epochs, so this doesn't hang waiting on a slow one) and then shows whatever results already finished, instead of losing them
8. **Results**  a sortable-looking results table appears inline; click "Save results to CSV" to export

If `textual` isn't installed, running `mlbench-lite-tui` prints a short message telling you to install the `[tui]` extra  it won't crash, and it doesn't affect the regular `mlbench-lite` CLI at all.

## 🛠️ Requirements

### **Core Dependencies**
- Python >= 3.8
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0 *(new  needed for `recommend_scaling` and `compare_models`)*

### **Optional Dependencies (for additional models)**
- xgboost >= 1.5.0 (for XGBoost models)
- lightgbm >= 3.2.0 (for LightGBM models)
- catboost >= 1.0.0 (for CatBoost models)
- torch >= 2.0.0 (for Deep Tabular models  `pip install mlbench-lite[deep]`)

### **Optional Dependency (for the interactive TUI)**
- textual >= 0.47.0 (`pip install mlbench-lite[tui]`)

**Note**: The library works with just the core dependencies. Optional dependencies are automatically installed when you install the package, but models/features from unavailable libraries will be skipped gracefully  `import mlbench_lite` never fails because `torch` or `textual` are missing, and neither does anything else in the library.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=mlbench_lite

# Quick functionality test
python -c "from mlbench_lite import benchmark, load_clover; X, y = load_clover(return_X_y=True); results = benchmark(X, y); print(results)"

# Quick check that the Deep Tabular models are wired up (needs [deep] installed)
python -c "from mlbench_lite import benchmark, load_clover; X, y = load_clover(return_X_y=True); print(benchmark(X, y, models=['Torch MLP', 'FT-Transformer']))"
```

## 🚀 Development

### Setup Development Environment

```bash
git clone https://github.com/Arefin994/mlbench-lite.git
cd mlbench-lite
pip install -e ".[dev]"
```

### Code Quality

```bash
black mlbench_lite tests
flake8 mlbench_lite tests
mypy mlbench_lite
```

### Building for Distribution

```bash
python -m build
twine upload dist/*
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Changelog

### 3.3.0
- **NEW**: `benchmark(..., progress_callback=..., cancel_check=...)`  optional hooks for UIs. `progress_callback(completed, total, model_name, status)` fires after each model; `cancel_check()` is polled before each model (and between epochs for the optional Deep Tabular torch models) to support graceful early stopping with partial results (`results_df.attrs["cancelled"]`)
- **NEW**: TUI  a live progress bar and per-model status line during a run, plus a **Cancel** button (`c` or click) that stops gracefully and still shows whatever finished
- Both are purely additive  `benchmark()` behaves identically to 3.2.1 when these two new parameters are left as `None` (the default)

### 3.2.1
- **NEW**: `--data PATH` / `--target COLUMN` / `--feature-columns COL [COL ...]`  benchmark your own CSV/TSV file from the CLI, not just built-in datasets
- **NEW**: TUI  pick your own CSV/TSV file as the dataset source, with an interactive target-column (y) picker and feature-column (X) checklist
- **NEW**: TUI  "Show scaling recommendation for this dataset" button surfaces `recommend_scaling()`'s report inline, for either a built-in dataset or your loaded CSV
- **FIXED**: task auto-detection crashed on string-labeled classification targets (e.g. a CSV column of `"yes"`/`"no"`)  now handled correctly
- **IMPROVED**: "No results produced" now explains that classification/regression models have different names and points to `--list-models --task <...>`, instead of a generic message

### 3.2.0
- **NEW**: Deep Tabular model category  Torch MLP and FT-Transformer (classification + regression), optional (`pip install mlbench-lite[deep]`), gated behind `torch` the same way XGBoost/LightGBM/CatBoost are  `import mlbench_lite` and every existing feature work identically without it
- **NEW**: `device="auto"`  checks for an available GPU and picks accordingly; `"cpu"`/`"gpu"` behave exactly as before
- **NEW**: Interactive TUI  `mlbench-lite-tui` (optional, `pip install mlbench-lite[tui]`), a keyboard/mouse-driven way to pick a dataset, models, and options and run a benchmark, built on the same `benchmark()` call as the CLI/API
- **NEW**: `set_seed(seed, deterministic=False)`  seeds Python's `random`, NumPy, and (if installed) PyTorch/CUDA in one call; `benchmark()` now calls this internally using `random_state`, so a full run (sklearn + Deep Tabular models together) is reproducible end to end
- **FIXED**: `mlbench-lite` console command wasn't actually registered as an installable entry point in `pyproject.toml`  added `[project.scripts]` for both `mlbench-lite` and the new `mlbench-lite-tui`
- Fully backward compatible  every new parameter/model is additive and off/absent unless explicitly requested or installed

### 3.1.0
- **NEW**: `groups=` parameter for leakage-safe splitting (`GroupShuffleSplit` / `GroupKFold`)
- **NEW**: `task=` parameter to override auto-detection, with a warning when detection is ambiguous
- **NEW**: Regressor parity with the classifier model list (SVR, KNN, MLP, Extra Trees, Bagging, AdaBoost, Gaussian Process regressors)
- **NEW**: `scaling=` parameter  `"standard" | "minmax" | "robust" | "log" | "l2" | "auto"`
- **NEW**: `recommend_scaling(X)`  read-only per-column scaling diagnostic report
- **NEW**: `device=`  GPU routing for XGBoost/LightGBM/CatBoost
- **NEW**: `save_to=`  write results directly to CSV/JSON
- **NEW**: `compare_models()`  Friedman + pairwise Wilcoxon significance testing across CV folds
- **NEW**: CLI flags for all of the above
- **FIXED**: XGBoost classifier crashing on non-contiguous integer labels
- Fully backward compatible  `use_scaling=True` still works as an alias for `scaling="standard"`; every new parameter defaults to off

### 3.0.0
- Added `cv=` (cross-validation) and `use_scaling=` support
- Added regression support (`get_available_regressors`, RMSE/MAE/R² metrics)
- *(Note: this release shipped without README documentation for these features  the 3.1.0 update above fills that gap.)*

### 2.0.0
- **MAJOR UPDATE**: Added 20+ machine learning models
- **NEW**: Flexible model selection (specific models, categories, exclusions)
- **NEW**: Support for XGBoost, LightGBM, and CatBoost
- **NEW**: Model information and listing functions
- **NEW**: Comprehensive model categories (Linear, Tree-based, SVM, etc.)
- **IMPROVED**: Enhanced API with more parameters
- **IMPROVED**: Better error handling and graceful degradation
- **IMPROVED**: Updated documentation with extensive examples

### 0.1.0
- Initial release
- Basic benchmarking functionality
- Support for Logistic Regression, Random Forest, and SVM
- Comprehensive metrics (Accuracy, Precision, Recall, F1)
- Custom clover dataset
- Full test coverage
- PyPI ready

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Arefin994/mlbench-lite/issues) page
2. Create a new issue with detailed information
3. Include code examples and error messages

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Uses [pandas](https://pandas.pydata.org/) for data handling
- Inspired by the need for simple ML benchmarking tools

## Citation

If you use `mlbench-lite` in your research, please cite:

```bibtex
@software{mlbench_lite2026,
  author = {Amin, Arefin},
  title = {mlbench-lite: A Lightweight Benchmarking Framework for Machine Learning Systems},
  year = {2026},
  url = {https://github.com/Arefin994/mlbench-lite},
  version = {3.3.0}
}
```
