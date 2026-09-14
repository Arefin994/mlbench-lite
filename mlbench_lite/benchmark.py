from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from .deep_models import (
    TORCH_AVAILABLE,
    FTTransformerClassifier,
    FTTransformerRegressor,
    TorchMLPClassifier,
    TorchMLPRegressor,
)
from .preprocessing import build_scaler
from .utils import set_seed

warnings.filterwarnings("ignore")
try:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import (
        LogisticRegression,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    SKLEARN_CLF_AVAILABLE = True
except ImportError:
    SKLEARN_CLF_AVAILABLE = False
try:
    from sklearn.ensemble import (
        AdaBoostRegressor,
        BaggingRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    SKLEARN_REG_AVAILABLE = True
except ImportError:
    SKLEARN_REG_AVAILABLE = False
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

_SCALE_SENSITIVE = {
    "Logistic Regression",
    "SGD Classifier",
    "Perceptron",
    "Passive Aggressive",
    "SVM (RBF)",
    "SVM (Linear)",
    "K-Nearest Neighbors",
    "Multi-layer Perceptron",
    "Ridge Classifier",
    "Linear Regression",
    "Ridge",
    # regressor additions (parity with the classifier scale-sensitive set)
    "SVR",
    "K-Nearest Neighbors Regressor",
    "MLP Regressor",
    # deep tabular models (torch) -- also benefit from scaling
    "Torch MLP",
    "Torch MLP Regressor",
    "FT-Transformer",
    "FT-Transformer Regressor",
}


def _detect_task(y: np.ndarray, explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in ("classification", "regression"):
            raise ValueError("task must be 'classification', 'regression', or None")
        return explicit
    target_type = type_of_target(y)
    if target_type in ("continuous", "continuous-multioutput"):
        return "regression"
    n_unique = len(np.unique(y))
    y_arr = np.asarray(y)
    # only numeric dtypes can be "int-like" -- string/object labels (the common
    # case for a CSV target column) always mean classification, and previously
    # crashed here trying np.round() on non-numeric data
    is_numeric_dtype = np.issubdtype(y_arr.dtype, np.number)
    is_int_like = is_numeric_dtype and (
        np.issubdtype(y_arr.dtype, np.integer) or np.allclose(y_arr, np.round(y_arr))
    )
    if is_int_like and n_unique > 20:
        warnings.warn(
            f"Target has an integer dtype with {n_unique} unique values -- auto-detected as "
            "'classification', but this may actually be a bounded numeric quantity better suited "
            "to regression (e.g. a count or an ordinal score). If that's the case, pass task='regression' "
            "explicitly. Proceeding with classification.",
            UserWarning,
            stacklevel=3,
        )
    return "classification"


def _maybe_pipeline(name: str, model, scaler_mode: Optional[str], X):
    if scaler_mode is None or name not in _SCALE_SENSITIVE:
        return model
    scaler = build_scaler(X, scaler_mode)
    if scaler is None:
        return model
    return Pipeline([("scaler", scaler), ("model", model)])


_TORCH_MODEL_NAMES = {
    "Torch MLP",
    "Torch MLP Regressor",
    "FT-Transformer",
    "FT-Transformer Regressor",
}


def _gpu_is_available() -> bool:
    """Best-effort GPU check across the libraries this project can offload to.
    Only used to resolve device='auto' -- explicit 'gpu' keeps its old
    behavior (attempt it regardless, matching pre-existing semantics)."""
    if TORCH_AVAILABLE:
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except Exception:
            pass
    return False


def _resolve_device(device: str) -> str:
    """Resolve 'cpu' | 'gpu' | 'auto' to 'cpu' or 'gpu'.

    'auto' is new and additive: it checks for an available GPU and picks
    accordingly. 'cpu' and 'gpu' behave exactly as before -- 'gpu' still
    means "try, and fall back with a warning if it doesn't work", same as
    prior versions.
    """
    if device == "auto":
        return "gpu" if _gpu_is_available() else "cpu"
    return device


def _apply_device(name: str, model, device: str):
    resolved = _resolve_device(device)
    if resolved == "cpu":
        return model
    try:
        if name.startswith("XGBoost"):
            model.set_params(device="cuda")
        elif name.startswith("LightGBM"):
            model.set_params(device="gpu")
        elif name.startswith("CatBoost"):
            model.set_params(task_type="GPU")
        elif name in _TORCH_MODEL_NAMES:
            model.set_params(device="cuda")
    except Exception as exc:
        warnings.warn(
            f"[mlbench-lite] Could not set device='gpu' for {name}: {exc}. Falling back to CPU.",
            UserWarning,
            stacklevel=3,
        )
    return model


def _select_models(
    available: Dict,
    models: Optional[List[str]],
    model_categories: Optional[List[str]],
    exclude_models: Optional[List[str]],
) -> Dict:
    if models is not None:
        selected: Dict = {}
        for category, model_list in available.items():
            for name, model in model_list:
                if name in models:
                    selected.setdefault(category, []).append((name, model))
    elif model_categories is not None:
        selected = {cat: available[cat] for cat in model_categories if cat in available}
    else:
        selected = dict(available)
    if exclude_models:
        for category in list(selected.keys()):
            selected[category] = [
                (n, m) for n, m in selected[category] if n not in exclude_models
            ]
            if not selected[category]:
                del selected[category]
    return selected


def _eval_single_split(
    model, X_train, X_test, y_train, y_test, task: str
) -> Tuple[Dict, float]:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 4)
    y_pred = model.predict(X_test)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "RMSE": round(rmse, 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "R2": round(float(r2_score(y_test, y_pred)), 4),
        }
    else:
        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(
                precision_score(y_test, y_pred, average="macro", zero_division=0), 4
            ),
            "Recall": round(
                recall_score(y_test, y_pred, average="macro", zero_division=0), 4
            ),
            "F1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        }
    return metrics, train_time


def _eval_cross_validation(
    model, X, y, cv, task: str, groups=None
) -> Tuple[Dict, float, Dict]:
    if task == "regression":
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
    else:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        }
    t0 = time.time()
    cv_res = cross_validate(
        model, X, y, cv=cv, scoring=scoring, groups=groups, error_score="raise"
    )
    total_time = round(time.time() - t0, 4)
    if task == "regression":
        metrics = {
            "RMSE": round(float(-np.mean(cv_res["test_rmse"])), 4),
            "RMSE_std": round(float(np.std(cv_res["test_rmse"])), 4),
            "MAE": round(float(-np.mean(cv_res["test_mae"])), 4),
            "MAE_std": round(float(np.std(cv_res["test_mae"])), 4),
            "R2": round(float(np.mean(cv_res["test_r2"])), 4),
            "R2_std": round(float(np.std(cv_res["test_r2"])), 4),
        }
        raw = {
            "rmse": (-cv_res["test_rmse"]).tolist(),
            "mae": (-cv_res["test_mae"]).tolist(),
            "r2": cv_res["test_r2"].tolist(),
        }
    else:
        metrics = {
            "Accuracy": round(float(np.mean(cv_res["test_accuracy"])), 4),
            "Accuracy_std": round(float(np.std(cv_res["test_accuracy"])), 4),
            "Precision": round(float(np.mean(cv_res["test_precision"])), 4),
            "Precision_std": round(float(np.std(cv_res["test_precision"])), 4),
            "Recall": round(float(np.mean(cv_res["test_recall"])), 4),
            "Recall_std": round(float(np.std(cv_res["test_recall"])), 4),
            "F1": round(float(np.mean(cv_res["test_f1"])), 4),
            "F1_std": round(float(np.std(cv_res["test_f1"])), 4),
        }
        raw = {
            "accuracy": cv_res["test_accuracy"].tolist(),
            "precision": cv_res["test_precision"].tolist(),
            "recall": cv_res["test_recall"].tolist(),
            "f1": cv_res["test_f1"].tolist(),
        }
    return metrics, total_time, raw


def get_available_classifiers(random_state: int = 42) -> Dict:
    models: Dict = {}
    if not SKLEARN_CLF_AVAILABLE:
        return models
    models["Linear Models"] = [
        (
            "Logistic Regression",
            LogisticRegression(random_state=random_state, max_iter=1000),
        ),
        ("Ridge Classifier", RidgeClassifier(random_state=random_state)),
        ("SGD Classifier", SGDClassifier(random_state=random_state)),
        ("Perceptron", Perceptron(random_state=random_state)),
        ("Passive Aggressive", PassiveAggressiveClassifier(random_state=random_state)),
    ]
    models["Tree-based Models"] = [
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ("Random Forest", RandomForestClassifier(random_state=random_state)),
        ("Extra Trees", ExtraTreesClassifier(random_state=random_state)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=random_state)),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
        ("Bagging Classifier", BaggingClassifier(random_state=random_state)),
    ]
    models["SVM Models"] = [
        ("SVM (RBF)", SVC(random_state=random_state)),
        ("SVM (Linear)", LinearSVC(random_state=random_state)),
    ]
    models["Neighbors"] = [
        ("K-Nearest Neighbors", KNeighborsClassifier()),
    ]
    models["Naive Bayes"] = [
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Multinomial Naive Bayes", MultinomialNB()),
        ("Bernoulli Naive Bayes", BernoulliNB()),
    ]
    models["Discriminant Analysis"] = [
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ]
    models["Neural Networks"] = [
        (
            "Multi-layer Perceptron",
            MLPClassifier(random_state=random_state, max_iter=1000),
        ),
    ]
    models["Gaussian Process"] = [
        ("Gaussian Process", GaussianProcessClassifier(random_state=random_state)),
    ]
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = [
            (
                "XGBoost",
                xgb.XGBClassifier(
                    random_state=random_state, eval_metric="logloss", verbosity=0
                ),
            ),
        ]
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = [
            ("LightGBM", lgb.LGBMClassifier(random_state=random_state, verbose=-1)),
        ]
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = [
            (
                "CatBoost",
                cb.CatBoostClassifier(random_state=random_state, verbose=False),
            ),
        ]
    if TORCH_AVAILABLE:
        models["Deep Tabular"] = [
            ("Torch MLP", TorchMLPClassifier(random_state=random_state)),
            ("FT-Transformer", FTTransformerClassifier(random_state=random_state)),
        ]
    return models


def get_available_regressors(random_state: int = 42) -> Dict:
    models: Dict = {}
    if not SKLEARN_REG_AVAILABLE:
        return models
    models["Linear Models"] = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
    ]
    models["Tree-based Models"] = [
        ("Decision Tree Regressor", DecisionTreeRegressor(random_state=random_state)),
        ("Random Forest Regressor", RandomForestRegressor(random_state=random_state)),
        ("Extra Trees Regressor", ExtraTreesRegressor(random_state=random_state)),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=random_state),
        ),
        ("AdaBoost Regressor", AdaBoostRegressor(random_state=random_state)),
        ("Bagging Regressor", BaggingRegressor(random_state=random_state)),
    ]
    models["SVM Models"] = [
        ("SVR", SVR()),
    ]
    models["Neighbors"] = [
        ("K-Nearest Neighbors Regressor", KNeighborsRegressor()),
    ]
    models["Neural Networks"] = [
        ("MLP Regressor", MLPRegressor(random_state=random_state, max_iter=1000)),
    ]
    models["Gaussian Process"] = [
        (
            "Gaussian Process Regressor",
            GaussianProcessRegressor(random_state=random_state),
        ),
    ]
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = [
            (
                "XGBoost Regressor",
                xgb.XGBRegressor(random_state=random_state, verbosity=0),
            ),
        ]
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = [
            (
                "LightGBM Regressor",
                lgb.LGBMRegressor(random_state=random_state, verbose=-1),
            ),
        ]
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = [
            (
                "CatBoost Regressor",
                cb.CatBoostRegressor(random_state=random_state, verbose=False),
            ),
        ]
    if TORCH_AVAILABLE:
        models["Deep Tabular"] = [
            ("Torch MLP Regressor", TorchMLPRegressor(random_state=random_state)),
            (
                "FT-Transformer Regressor",
                FTTransformerRegressor(random_state=random_state),
            ),
        ]
    return models


def get_available_models() -> Dict:
    return get_available_classifiers()


def benchmark(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    models: Optional[List[str]] = None,
    model_categories: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    cv: Optional[int] = None,
    groups=None,
    task: Optional[str] = None,
    use_scaling: bool = False,
    scaling: Optional[str] = None,
    device: str = "cpu",
    save_to: Optional[str] = None,
    show_std: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """
    New (all additive, nothing below changes existing behavior when unused):

    groups : array-like, optional
        Group label per row (e.g. a config id shared by replicate/near-duplicate
        rows). When set, splitting uses GroupShuffleSplit (single split) or
        GroupKFold (cv) so rows sharing a group never appear in both train and
        test -- prevents near-duplicate rows from leaking across the split.
    task : "classification" | "regression" | None
        Force the task instead of relying on auto-detection. Auto-detection
        warns when it's guessing on an ambiguous integer target with many
        unique values (see _detect_task) -- pass this explicitly if you hit
        that warning.
    scaling : "standard" | "minmax" | "robust" | "log" | "l2" | "auto" | None
        Only applied to models known to be scale-sensitive (tree/boosting
        models are left untouched either way). "auto" inspects each column
        (see `preprocessing.recommend_scaling`) and picks a treatment per
        column. `use_scaling=True` (legacy) is equivalent to scaling="standard".
        Nothing is scaled unless one of these is explicitly set.
    device : "cpu" | "gpu" | "auto"
        Affects XGBoost/LightGBM/CatBoost and the optional torch-based Deep
        Tabular models (Torch MLP, FT-Transformer). "auto" (new) checks for
        an available GPU and picks accordingly; "gpu" keeps the old
        behavior of attempting GPU regardless and falling back to CPU with
        a warning if it can't be set. Models that don't support GPU
        acceleration (most of sklearn) are unaffected either way.
    save_to : str, optional
        Path to write results to (.csv or .json inferred from extension).
    progress_callback : callable(completed, total, model_name, status), optional
        Called once after each model finishes (or is skipped by cancellation),
        where `status` is "done", "error", or "cancelled". Meant for UIs (the
        TUI uses this to drive its progress bar) -- has no effect on results
        or timing if left as None.
    cancel_check : callable() -> bool, optional
        Polled before each model starts (and, for the optional Deep Tabular
        torch models, between epochs too) -- if it returns True, the run
        stops early and returns whatever results were already gathered,
        with `results_df.attrs["cancelled"] = True`, instead of raising or
        losing the partial results. Meant for UIs offering a "Cancel"
        button; has no effect if left as None.

    Reproducibility: this seeds Python's `random`, NumPy, and (if installed)
    PyTorch/CUDA once at the start of the run, in addition to passing
    `random_state` to every individual estimator/split as before -- so a
    full run (sklearn models and any Deep Tabular models together) is
    reproducible given the same `random_state`, not just the sklearn part.
    """
    set_seed(random_state)
    y = np.asarray(y)
    detected_task = _detect_task(y, task)
    scaler_mode = (
        scaling if scaling is not None else ("standard" if use_scaling else None)
    )

    label_encoder = None
    if detected_task == "classification":
        # Encode labels to a contiguous 0..k-1 range once, up front. This is what
        # XGBoost requires internally (it rejects non-contiguous integer labels);
        # doing it here for *all* classifiers is harmless (macro-averaged metrics
        # are invariant to a consistent relabeling) and fixes the XGBoost failure
        # without a model-specific special case.
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    available = (
        get_available_regressors(random_state=random_state)
        if detected_task == "regression"
        else get_available_classifiers(random_state=random_state)
    )
    selected = _select_models(available, models, model_categories, exclude_models)
    if not selected:
        warnings.warn(
            "No models matched the selection criteria.  "
            "Check model names / categories.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    X_train = X_test = y_train = y_test = None
    cv_splits = cv
    if cv is None:
        if groups is not None:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_arr = (
                X.iloc[train_idx] if hasattr(X, "iloc") else np.asarray(X)[train_idx]
            )
            X_test = X.iloc[test_idx] if hasattr(X, "iloc") else np.asarray(X)[test_idx]
            X_train, y_train, y_test = X_arr, y[train_idx], y[test_idx]
        else:
            stratify = y if detected_task == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
    elif groups is not None:
        cv_splits = list(GroupKFold(n_splits=cv).split(X, y, groups))

    raw_scores: Dict[str, Dict] = {}
    results = []
    total_models = sum(len(model_list) for model_list in selected.values())
    completed = 0
    cancelled = False
    for category, model_list in selected.items():
        if cancelled:
            break
        for model_name, base_model in model_list:
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            row: Dict = {"Model": model_name, "Category": category}
            model = _apply_device(model_name, base_model, device)
            fitted_model = _maybe_pipeline(model_name, model, scaler_mode, X)
            if cancel_check is not None and model_name in _TORCH_MODEL_NAMES:
                # Deep Tabular models can also stop mid-training, between epochs,
                # instead of only at the model boundary -- they're usually the
                # slowest models in a run, so this is where a "Cancel" button
                # actually needs to bite.
                target = fitted_model
                if hasattr(target, "steps"):  # unwrap if inside a scaling Pipeline
                    target = target.named_steps.get("model", target)
                try:
                    target._cancel_check = cancel_check
                except Exception:
                    pass
            try:
                if cv is None:
                    metrics, train_time = _eval_single_split(
                        fitted_model, X_train, X_test, y_train, y_test, detected_task
                    )
                else:
                    metrics, train_time, raw = _eval_cross_validation(
                        fitted_model,
                        X,
                        y,
                        cv=cv_splits,
                        task=detected_task,
                        groups=None,
                    )
                    raw_scores[model_name] = raw
                row.update(metrics)
                row["Train Time (s)"] = train_time
                status = "done"
            except Exception as exc:
                err_msg = str(exc)
                warnings.warn(
                    f"[mlbench-lite] {model_name} failed: {err_msg}",
                    UserWarning,
                    stacklevel=2,
                )
                row["Train Time (s)"] = None
                row["Error"] = err_msg
                status = "error"
            results.append(row)
            completed += 1
            if progress_callback is not None:
                try:
                    progress_callback(completed, total_models, model_name, status)
                except Exception:
                    pass  # a broken progress callback should never take down the benchmark run
    results_df = pd.DataFrame(results)
    if results_df.empty:
        warnings.warn("No models produced results.", UserWarning, stacklevel=2)
        results_df.attrs["cancelled"] = cancelled
        return results_df
    if cv is not None and not show_std:
        std_cols = [c for c in results_df.columns if c.endswith("_std")]
        results_df = results_df.drop(columns=std_cols)
    sort_col = "R2" if detected_task == "regression" else "Accuracy"
    if sort_col in results_df.columns:
        results_df = results_df.sort_values(sort_col, ascending=False).reset_index(
            drop=True
        )
    results_df.attrs["raw_scores"] = (
        raw_scores  # per-fold scores, for stats.compare_models
    )
    results_df.attrs["task"] = detected_task
    results_df.attrs["cancelled"] = cancelled
    if save_to is not None:
        if save_to.endswith(".json"):
            results_df.to_json(save_to, orient="records", indent=2)
        else:
            results_df.to_csv(save_to, index=False)
    return results_df


def list_available_models(task: str = "classification") -> Dict:
    getter = (
        get_available_regressors if task == "regression" else get_available_classifiers
    )
    return {
        category: [name for name, _ in model_list]
        for category, model_list in getter().items()
    }


def get_model_info() -> pd.DataFrame:
    info = [
        {
            "Category": "Linear Models",
            "Model": "Logistic Regression",
            "Description": "Linear model for classification using logistic function",
        },
        {
            "Category": "Linear Models",
            "Model": "Ridge Classifier",
            "Description": "Linear classifier with L2 regularization",
        },
        {
            "Category": "Linear Models",
            "Model": "SGD Classifier",
            "Description": "Linear classifier using Stochastic Gradient Descent",
        },
        {
            "Category": "Linear Models",
            "Model": "Perceptron",
            "Description": "Simple linear classifier",
        },
        {
            "Category": "Linear Models",
            "Model": "Passive Aggressive",
            "Description": "Online learning algorithm for classification",
        },
        {
            "Category": "Tree-based Models",
            "Model": "Decision Tree",
            "Description": "Non-parametric supervised learning method",
        },
        {
            "Category": "Tree-based Models",
            "Model": "Random Forest",
            "Description": "Ensemble of decision trees with bagging",
        },
        {
            "Category": "Tree-based Models",
            "Model": "Extra Trees",
            "Description": "Extremely randomized trees ensemble",
        },
        {
            "Category": "Tree-based Models",
            "Model": "Gradient Boosting",
            "Description": "Boosting ensemble method using gradient descent",
        },
        {
            "Category": "Tree-based Models",
            "Model": "AdaBoost",
            "Description": "Adaptive boosting ensemble method",
        },
        {
            "Category": "Tree-based Models",
            "Model": "Bagging Classifier",
            "Description": "Bootstrap aggregating ensemble method",
        },
        {
            "Category": "SVM Models",
            "Model": "SVM (RBF)",
            "Description": "Support Vector Machine with RBF kernel",
        },
        {
            "Category": "SVM Models",
            "Model": "SVM (Linear)",
            "Description": "Support Vector Machine with linear kernel",
        },
        {
            "Category": "Neighbors",
            "Model": "K-Nearest Neighbors",
            "Description": "Instance-based learning algorithm",
        },
        {
            "Category": "Naive Bayes",
            "Model": "Gaussian Naive Bayes",
            "Description": "Naive Bayes classifier for Gaussian features",
        },
        {
            "Category": "Naive Bayes",
            "Model": "Multinomial Naive Bayes",
            "Description": "Naive Bayes classifier for multinomial features",
        },
        {
            "Category": "Naive Bayes",
            "Model": "Bernoulli Naive Bayes",
            "Description": "Naive Bayes classifier for binary features",
        },
        {
            "Category": "Discriminant Analysis",
            "Model": "Linear Discriminant Analysis",
            "Description": "Linear dimensionality reduction and classification",
        },
        {
            "Category": "Discriminant Analysis",
            "Model": "Quadratic Discriminant Analysis",
            "Description": "Quadratic classifier with Gaussian assumptions",
        },
        {
            "Category": "Neural Networks",
            "Model": "Multi-layer Perceptron",
            "Description": "Feedforward artificial neural network",
        },
        {
            "Category": "Gaussian Process",
            "Model": "Gaussian Process",
            "Description": "Probabilistic classifier using Gaussian processes",
        },
        {
            "Category": "Linear Models (Reg)",
            "Model": "Linear Regression",
            "Description": "Ordinary least-squares linear regression",
        },
        {
            "Category": "Linear Models (Reg)",
            "Model": "Ridge",
            "Description": "Linear regression with L2 regularization",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "Decision Tree Regressor",
            "Description": "Non-parametric supervised learning method for regression",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "Random Forest Regressor",
            "Description": "Ensemble of decision trees for regression",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "Extra Trees Regressor",
            "Description": "Extremely randomized trees ensemble for regression",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "Gradient Boosting Regressor",
            "Description": "Gradient boosted trees for regression",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "AdaBoost Regressor",
            "Description": "Adaptive boosting ensemble method for regression",
        },
        {
            "Category": "Tree-based Models (Reg)",
            "Model": "Bagging Regressor",
            "Description": "Bootstrap aggregating ensemble method for regression",
        },
        {
            "Category": "SVM Models (Reg)",
            "Model": "SVR",
            "Description": "Support Vector Machine for regression",
        },
        {
            "Category": "Neighbors (Reg)",
            "Model": "K-Nearest Neighbors Regressor",
            "Description": "Instance-based learning algorithm for regression",
        },
        {
            "Category": "Neural Networks (Reg)",
            "Model": "MLP Regressor",
            "Description": "Feedforward artificial neural network for regression",
        },
        {
            "Category": "Gaussian Process (Reg)",
            "Model": "Gaussian Process Regressor",
            "Description": "Probabilistic regressor using Gaussian processes",
        },
    ]
    if XGBOOST_AVAILABLE:
        info += [
            {
                "Category": "XGBoost",
                "Model": "XGBoost",
                "Description": "Extreme gradient boosting for classification",
            },
            {
                "Category": "XGBoost",
                "Model": "XGBoost Regressor",
                "Description": "Extreme gradient boosting for regression",
            },
        ]
    if LIGHTGBM_AVAILABLE:
        info += [
            {
                "Category": "LightGBM",
                "Model": "LightGBM",
                "Description": "Light gradient boosting machine for classification",
            },
            {
                "Category": "LightGBM",
                "Model": "LightGBM Regressor",
                "Description": "Light gradient boosting machine for regression",
            },
        ]
    if CATBOOST_AVAILABLE:
        info += [
            {
                "Category": "CatBoost",
                "Model": "CatBoost",
                "Description": "Categorical boosting framework for classification",
            },
            {
                "Category": "CatBoost",
                "Model": "CatBoost Regressor",
                "Description": "Categorical boosting framework for regression",
            },
        ]
    if TORCH_AVAILABLE:
        info += [
            {
                "Category": "Deep Tabular",
                "Model": "Torch MLP",
                "Description": "Feedforward PyTorch neural network for classification (GPU-capable)",
            },
            {
                "Category": "Deep Tabular",
                "Model": "Torch MLP Regressor",
                "Description": "Feedforward PyTorch neural network for regression (GPU-capable)",
            },
            {
                "Category": "Deep Tabular",
                "Model": "FT-Transformer",
                "Description": "Feature Tokenizer + Transformer for tabular classification (GPU-capable)",
            },
            {
                "Category": "Deep Tabular",
                "Model": "FT-Transformer Regressor",
                "Description": "Feature Tokenizer + Transformer for tabular regression (GPU-capable)",
            },
        ]
    return pd.DataFrame(info)
