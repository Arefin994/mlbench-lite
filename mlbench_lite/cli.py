from __future__ import annotations

import argparse
import sys


def _load_demo_dataset(name: str, task: str):
    if task == "regression":
        from sklearn.datasets import load_diabetes
        return load_diabetes(return_X_y=True)
    name = name.lower()
    if name == "iris":
        from sklearn.datasets import load_iris
        return load_iris(return_X_y=True)
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer(return_X_y=True)
    if name == "wine":
        from sklearn.datasets import load_wine
        return load_wine(return_X_y=True)
    from mlbench_lite.datasets import load_clover
    return load_clover(return_X_y=True)
def _load_custom_dataset(path: str, target: str | None, feature_columns: list[str] | None):
    """Load a user-supplied CSV/TSV into (X, y) using pandas, the same
    library already used for benchmark() output -- no new dependency."""
    import pandas as pd
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)          # <-- add this branch
    else:
        sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 2:
        raise ValueError(f"'{path}' needs at least 2 columns (features + target); found {df.shape[1]}.")
    target_col = target or df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in '{path}'. "
                          f"Available columns: {list(df.columns)}")
    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature column(s) not found in '{path}': {missing}")
        feature_cols = [c for c in feature_columns if c != target_col]
    else:
        feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols], df[target_col]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlbench-lite",
        description="Quick ML model comparison — mlbench-lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  mlbench-lite
  mlbench-lite --dataset iris --cv 5
  mlbench-lite --task regression
  mlbench-lite --models "Random Forest" "SVM (RBF)" --scaling
  mlbench-lite --task regression --scaling-mode auto --show-recommendations
  mlbench-lite --device auto --categories "Deep Tabular"
  mlbench-lite --data my_data.csv --target label
  mlbench-lite --data my_data.csv --target label --feature-columns col_a col_b col_c

Prefer picking things with arrow keys / mouse instead of flags? Try the
interactive TUI (requires `pip install mlbench-lite[tui]`):
  mlbench-lite-tui
        """,
    )
    p.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help="Path to your own CSV/TSV file to benchmark on, instead of a built-in "
             "dataset. Overrides --dataset when set.",
    )
    p.add_argument(
        "--target",
        default=None,
        metavar="COLUMN",
        help="Target column name in --data (default: the last column).",
    )
    p.add_argument(
        "--feature-columns",
        nargs="+",
        default=None,
        dest="feature_columns",
        metavar="COL",
        help="Specific feature columns to use from --data (default: every column "
             "except --target).",
    )
    p.add_argument(
        "--dataset",
        default="clover",
        choices=["clover", "iris", "wine", "breast_cancer"],
        help="Built-in dataset to benchmark on (default: clover)",
    )
    p.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help="Force task type. 'regression' uses the diabetes dataset.",
    )
    p.add_argument(
        "--cv",
        type=int,
        default=None,
        metavar="FOLDS",
        help="Number of cross-validation folds (omit to use train/test split)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        dest="test_size",
        metavar="FRAC",
        help="Test fraction when not using CV (default: 0.2)",
    )
    p.add_argument(
        "--scaling",
        action="store_true",
        help="Wrap scale-sensitive models in a StandardScaler pipeline "
             "(shorthand for --scaling-mode standard)",
    )
    p.add_argument(
        "--scaling-mode",
        default=None,
        dest="scaling_mode",
        choices=["standard", "minmax", "robust", "log", "l2", "auto"],
        metavar="MODE",
        help="Scaling mode for scale-sensitive models. Overrides --scaling if both are given.",
    )
    p.add_argument(
        "--show-recommendations",
        action="store_true",
        dest="show_recommendations",
        help="Print a per-column scaling recommendation report before running (read-only, "
             "doesn't require --scaling-mode)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="Device for XGBoost/LightGBM/CatBoost and the optional Deep Tabular "
             "(torch) models. 'auto' uses a GPU if one is available, otherwise "
             "CPU. Default: cpu (unchanged from previous versions).",
    )
    p.add_argument(
        "--save-to",
        default=None,
        dest="save_to",
        metavar="PATH",
        help="Save results to a .csv or .json file",
    )
    p.add_argument(
        "--no-std",
        action="store_true",
        dest="no_std",
        help="Hide std columns when using cross-validation",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Whitelist specific model names",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CAT",
        help="Whitelist specific model categories",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Exclude specific models by name",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        metavar="SEED",
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        dest="list_models",
        help="Print all available model names and exit",
    )
    return p
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list_models:
        from mlbench_lite.benchmark import list_available_models
        task_for_listing = args.task or "classification"
        print(f"\nAvailable models ({task_for_listing}):\n")
        for category, names in list_available_models(task=task_for_listing).items():
            print(f"  [{category}]")
            for n in names:
                print(f"    - {n}")
        print()
        sys.exit(0)
    from mlbench_lite.benchmark import benchmark
    task = args.task
    print("\n" + "=" * 62)
    print("  mlbench-lite  ·  quick model comparison")
    print("=" * 62)
    if args.data:
        dataset_name = args.data
    else:
        dataset_name = "diabetes" if task == "regression" else args.dataset
    scaling_mode = args.scaling_mode or ("standard" if args.scaling else None)
    print(f"  Dataset  : {dataset_name}")
    print(f"  Task     : {task or 'auto-detect'}")
    print(f"  CV folds : {args.cv or 'single split'}")
    print(f"  Scaling  : {scaling_mode or 'off'}")
    print(f"  Device   : {args.device}")
    print("=" * 62 + "\n")
    if args.data:
        try:
            X, y = _load_custom_dataset(args.data, args.target, args.feature_columns)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"Error loading --data: {exc}\n")
            sys.exit(1)
    else:
        X, y = _load_demo_dataset(args.dataset, task or "classification")
    if args.show_recommendations:
        from mlbench_lite.preprocessing import recommend_scaling
        print("Scaling recommendations (informational only, not applied unless "
              "--scaling / --scaling-mode is set):\n")
        print(recommend_scaling(X).to_string(index=False))
        print()
    results = benchmark(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        models=args.models,
        model_categories=args.categories,
        exclude_models=args.exclude,
        cv=args.cv,
        task=task,
        scaling=scaling_mode,
        device=args.device,
        save_to=args.save_to,
        show_std=not args.no_std,
    )
    if results.empty:
        detected_task = results.attrs.get("task", task or "classification")
        print("No results produced.\n")
        if args.models or args.categories:
            print(f"This usually means the model/category name(s) you gave don't exist "
                  f"for the detected task ('{detected_task}') -- classification and "
                  f"regression models have different names (e.g. 'Random Forest' vs "
                  f"'Random Forest Regressor'). Run:\n"
                  f"  mlbench-lite --list-models --task {detected_task}\n"
                  f"to see the valid names, or drop --models/--categories to see everything.\n")
        else:
            print("Check --exclude, --models, and --categories for typos or conflicts.\n")
        sys.exit(1)
    with_index = results.copy()
    with_index.index = with_index.index + 1
    with_index.index.name = "Rank"
    print(with_index.to_string())
    print()
    task_detected = "regression" if "RMSE" in results.columns else "classification"
    sort_col = "R2" if task_detected == "regression" else "Accuracy"
    if sort_col in results.columns:
        best = results.iloc[0]
        print(f"  🏆  Best model : {best['Model']}  ({sort_col} = {best[sort_col]})\n")
    if args.save_to:
        print(f"  💾  Results saved to {args.save_to}\n")
if __name__ == "__main__":
    main()
