from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from mlbench_lite import (
    benchmark,
    compare_models,
    get_model_info,
    list_available_models,
    load_clover,
    recommend_scaling,
    set_seed,
)
@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    return X, y
@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=42,
    )
    return X, y.astype(float)
class TestClassificationBasic:
    def test_returns_dataframe(self, clf_data):
        X, y = clf_data
        assert isinstance(benchmark(X, y), pd.DataFrame)
    def test_expected_columns(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y)
        for col in ("Model", "Category", "Accuracy", "Precision", "Recall", "F1"):
            assert col in results.columns, f"Missing column: {col}"
    def test_train_time_column_present(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y)
        assert "Train Time (s)" in results.columns
    def test_sorted_by_accuracy_descending(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y)
        accs = results["Accuracy"].dropna().tolist()
        assert accs == sorted(accs, reverse=True)
    def test_expected_models_present(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y)
        names = set(results["Model"])
        for m in ("Logistic Regression", "Random Forest", "SVM (RBF)"):
            assert m in names
    def test_metrics_in_unit_interval(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y)
        for col in ("Accuracy", "Precision", "Recall", "F1"):
            vals = results[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all()
    def test_reproducibility(self, clf_data):
        X, y = clf_data
        r1 = benchmark(X, y, random_state=7)
        r2 = benchmark(X, y, random_state=7)
        metric_cols = [c for c in r1.columns if c != "Train Time (s)"]
        pd.testing.assert_frame_equal(r1[metric_cols], r2[metric_cols])
    def test_different_random_states_differ(self, clf_data):
        X, y = clf_data
        r1 = benchmark(X, y, random_state=1)
        r2 = benchmark(X, y, random_state=999)
        assert not r1.equals(r2)
    def test_custom_test_size(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, test_size=0.3)
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
    def test_specific_models_filter(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, models=["Random Forest", "Logistic Regression"])
        assert set(results["Model"]) == {"Random Forest", "Logistic Regression"}
        assert len(results) == 2
    def test_model_categories_filter(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, model_categories=["Linear Models"])
        assert all(results["Category"] == "Linear Models")
        assert len(results) >= 5
    def test_exclude_models(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, exclude_models=["Gaussian Process"])
        assert "Gaussian Process" not in set(results["Model"])
    def test_small_dataset_does_not_crash(self):
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=0)
        results = benchmark(X, y)
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
class TestCrossValidation:
    def test_cv_returns_dataframe(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, cv=3, models=["Logistic Regression", "Random Forest"])
        assert isinstance(results, pd.DataFrame)
    def test_cv_std_columns_present(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, cv=3, models=["Logistic Regression"])
        for col in ("Accuracy_std", "Precision_std", "Recall_std", "F1_std"):
            assert col in results.columns
    def test_cv_std_columns_hidden_when_show_std_false(self, clf_data):
        X, y = clf_data
        results = benchmark(
            X, y, cv=3, models=["Logistic Regression"], show_std=False
        )
        std_cols = [c for c in results.columns if c.endswith("_std")]
        assert std_cols == []
    def test_cv_mean_accuracy_in_unit_interval(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, cv=3, models=["Random Forest"])
        acc = results["Accuracy"].iloc[0]
        assert 0.0 <= acc <= 1.0
    def test_cv_train_time_present(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, cv=3, models=["Logistic Regression"])
        assert "Train Time (s)" in results.columns
        assert results["Train Time (s)"].iloc[0] is not None
class TestRegression:
    def test_regression_detected_from_float_y(self, reg_data):
        X, y = reg_data
        results = benchmark(X, y, models=["Linear Regression", "Ridge"])
        assert "RMSE" in results.columns
        assert "MAE" in results.columns
        assert "R2" in results.columns
    def test_regression_no_classification_columns(self, reg_data):
        X, y = reg_data
        results = benchmark(X, y, models=["Linear Regression"])
        for col in ("Accuracy", "Precision", "Recall", "F1"):
            assert col not in results.columns
    def test_regression_sorted_by_r2_descending(self, reg_data):
        X, y = reg_data
        results = benchmark(
            X, y, models=["Linear Regression", "Ridge", "Random Forest Regressor"]
        )
        r2s = results["R2"].dropna().tolist()
        assert r2s == sorted(r2s, reverse=True)
    def test_regression_cv(self, reg_data):
        X, y = reg_data
        results = benchmark(
            X, y, cv=3, models=["Linear Regression", "Random Forest Regressor"]
        )
        assert "RMSE" in results.columns
        assert "R2_std" in results.columns
    def test_regression_train_time_present(self, reg_data):
        X, y = reg_data
        results = benchmark(X, y, models=["Linear Regression"])
        assert "Train Time (s)" in results.columns
class TestScaling:
    def test_use_scaling_returns_dataframe(self, clf_data):
        X, y = clf_data
        results = benchmark(
            X, y, use_scaling=True, models=["Logistic Regression", "SVM (RBF)"]
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
    def test_scaling_does_not_break_non_sensitive_models(self, clf_data):
        X, y = clf_data
        results = benchmark(
            X, y, use_scaling=True, models=["Random Forest", "Decision Tree"]
        )
        assert len(results) == 2
class TestTrainingTime:
    def test_train_time_is_non_negative(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, models=["Random Forest"])
        t = results["Train Time (s)"].iloc[0]
        assert t is not None
        assert t >= 0
    def test_cv_train_time_reflects_all_folds(self, clf_data):
        X, y = clf_data
        r_single = benchmark(X, y, models=["Random Forest"])
        r_cv = benchmark(X, y, cv=3, models=["Random Forest"])
        assert r_cv["Train Time (s)"].iloc[0] >= r_single["Train Time (s)"].iloc[0]
class TestErrorHandling:
    def test_failed_model_adds_error_column(self):
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=0
        )
        X = X - X.max()
        results = benchmark(X, y, models=["Multinomial Naive Bayes"])
        assert "Error" in results.columns
        err_val = results.loc[results["Model"] == "Multinomial Naive Bayes", "Error"]
        assert err_val.notna().any()
    def test_empty_selection_returns_empty_dataframe(self, clf_data):
        X, y = clf_data
        results = benchmark(X, y, models=["NonExistentModel"])
        assert isinstance(results, pd.DataFrame)
        assert results.empty
    def test_warning_on_empty_selection(self, clf_data):
        X, y = clf_data
        with pytest.warns(UserWarning, match="No models matched"):
            benchmark(X, y, models=["NonExistentModel"])
class TestUtilities:
    def test_list_available_models_returns_dict(self):
        result = list_available_models()
        assert isinstance(result, dict)
        assert "Linear Models" in result
        assert "Tree-based Models" in result
        assert "Random Forest" in result["Tree-based Models"]
    def test_get_model_info_returns_dataframe(self):
        info = get_model_info()
        assert isinstance(info, pd.DataFrame)
        assert {"Category", "Model", "Description"}.issubset(info.columns)
        assert len(info) >= 20
    def test_get_model_info_contains_regression_models(self):
        info = get_model_info()
        assert "Linear Regression" in info["Model"].values
        assert "Ridge" in info["Model"].values
class TestCLI:
    def test_cli_main_importable(self):
        from mlbench_lite.cli import main
        assert callable(main)


class TestLoadClover:
    def test_returns_bunch_by_default(self):
        data = load_clover()
        assert hasattr(data, "data")
        assert hasattr(data, "target")
        assert data.data.shape == (400, 4)

    def test_return_X_y(self):
        X, y = load_clover(return_X_y=True)
        assert X.shape == (400, 4)
        assert len(y) == 400


class TestRecommendScaling:
    def test_returns_dataframe(self, clf_data):
        X, _ = clf_data
        report = recommend_scaling(X)
        assert isinstance(report, pd.DataFrame)
        assert {"column", "recommendation"}.issubset(report.columns)

    def test_does_not_modify_input(self, clf_data):
        X, _ = clf_data
        original = X.copy()
        recommend_scaling(X)
        np.testing.assert_array_equal(X, original)


class TestCompareModels:
    def test_compare_models_with_cv(self, clf_data):
        X, y = clf_data
        results = benchmark(
            X, y, cv=3, models=["Random Forest", "Logistic Regression"]
        )
        comparison = compare_models(results)
        assert "friedman_p_value" in comparison
        assert "pairwise" in comparison
        assert len(comparison["pairwise"]) == 1


class TestSetSeed:
    def test_set_seed_is_callable(self):
        set_seed(42)
        set_seed(42, deterministic=False)
