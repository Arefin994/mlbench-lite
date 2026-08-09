# mlbench-lite

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/mlbench-lite?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/mlbench-lite)

A comprehensive machine learning benchmarking library that provides an easy way to compare multiple ML models on your dataset. Built with scikit-learn, XGBoost, LightGBM, CatBoost, and pandas for seamless integration into your ML workflow.

## 🚀 Features

- **Comprehensive Model Support**: 20+ ML models from multiple libraries
- **Flexible Model Selection**: Choose specific models, categories, or exclude models
- **Multiple ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Simple API**: One function call to benchmark multiple models
- **Comprehensive Metrics**: Returns Accuracy, Precision, Recall, and F1 scores
- **Custom Dataset**: Includes the `load_clover` dataset for testing
- **Easy Integration**: Works seamlessly with scikit-learn datasets
- **Pandas Output**: Results returned as a clean pandas DataFrame
- **Reproducible**: Consistent results with random state control
- **Model Information**: Get detailed info about available models

## 📦 Installation

```bash
pip install mlbench-lite
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

## 📚 API Reference

### `benchmark(X, y, test_size=0.2, random_state=42, models=None, model_categories=None, exclude_models=None)`

Benchmark multiple machine learning models on a dataset.

**Parameters:**
- `X` (array-like): Training vectors of shape (n_samples, n_features)
- `y` (array-like): Target values of shape (n_samples,)
- `test_size` (float, optional): Proportion of dataset for testing (default: 0.2)
- `random_state` (int, optional): Random seed for reproducibility (default: 42)
- `models` (list of str, optional): Specific models to use. If None, uses all available models.
- `model_categories` (list of str, optional): Categories of models to use. If None, uses all categories.
- `exclude_models` (list of str, optional): Models to exclude from benchmarking.

**Returns:**
- `pandas.DataFrame`: Results with columns:
  - `Model`: Name of the model
  - `Category`: Category of the model
  - `Accuracy`: Accuracy score
  - `Precision`: Precision score (macro-averaged)
  - `Recall`: Recall score (macro-averaged)
  - `F1`: F1 score (macro-averaged)

### `list_available_models()`

List all available models and their categories.

**Returns:**
- `dict`: Dictionary with model categories as keys and lists of model names as values

### `get_model_info()`

Get detailed information about available models.

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

# Load the clover dataset
X, y = load_clover(return_X_y=True)
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

# Benchmark all available models
results = benchmark(X, y)
print("\nBenchmark Results:")
print(results)

# Get the best model
best_model = results.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")
```

### 2. Model Selection - Specific Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Benchmark only specific models
results = benchmark(X, y, models=['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression'])
print("Selected Models Results:")
print(results)
```

### 3. Model Selection - By Categories

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Benchmark only tree-based models
results = benchmark(X, y, model_categories=['Tree-based Models'])
print("Tree-based Models Results:")
print(results)

# Benchmark multiple categories
results = benchmark(X, y, model_categories=['Linear Models', 'SVM Models'])
print("\nLinear and SVM Models Results:")
print(results)
```

### 4. Exclude Specific Models

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Exclude slow models
results = benchmark(X, y, exclude_models=['Gaussian Process', 'Multi-layer Perceptron'])
print("Results without slow models:")
print(results)
```

### 5. List Available Models

```python
from mlbench_lite import list_available_models, get_model_info

# List all available models by category
models = list_available_models()
print("Available Models by Category:")
for category, model_list in models.items():
    print(f"\n{category}:")
    for model in model_list:
        print(f"  - {model}")

# Get detailed model information
model_info = get_model_info()
print("\nDetailed Model Information:")
print(model_info)
```

### 6. Advanced Model Selection

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Complex selection: specific models from specific categories, excluding some
results = benchmark(
    X, y,
    models=['Random Forest', 'XGBoost', 'SVM (RBF)', 'Logistic Regression'],
    exclude_models=['SVM (Linear)']
)
print("Custom Selection Results:")
print(results)
```

### 7. Using with Scikit-learn Datasets

```python
from mlbench_lite import benchmark
from sklearn.datasets import load_wine, load_breast_cancer

# Test with Wine dataset
print("=== Wine Dataset ===")
X, y = load_wine(return_X_y=True)
results = benchmark(X, y)
print(results)

# Test with Breast Cancer dataset
print("\n=== Breast Cancer Dataset ===")
X, y = load_breast_cancer(return_X_y=True)
results = benchmark(X, y)
print(results)
```

### 8. Custom Test Size

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Use 30% of data for testing
results = benchmark(X, y, test_size=0.3)
print("Results with 30% test size:")
print(results)

# Use 10% of data for testing
results = benchmark(X, y, test_size=0.1)
print("\nResults with 10% test size:")
print(results)
```

### 9. Reproducible Results

```python
from mlbench_lite import benchmark, load_clover

X, y = load_clover(return_X_y=True)

# Set random seed for reproducible results
results1 = benchmark(X, y, random_state=123)
results2 = benchmark(X, y, random_state=123)

print("Results with random_state=123:")
print(results1)
print(f"\nResults are identical: {results1.equals(results2)}")

# Different random state produces different results
results3 = benchmark(X, y, random_state=456)
print(f"\nDifferent random state produces different results: {not results1.equals(results3)}")
```

### 10. Working with Synthetic Data

```python
from mlbench_lite import benchmark
from sklearn.datasets import make_classification

# Create synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_classes=4,
    random_state=42
)

print(f"Synthetic dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

results = benchmark(X, y)
print("\nBenchmark Results:")
print(results)
```

### 11. Analyzing Results

```python
from mlbench_lite import benchmark, load_clover
import pandas as pd

X, y = load_clover(return_X_y=True)
results = benchmark(X, y)

# Display results with better formatting
print("Detailed Results:")
print("=" * 60)
for idx, row in results.iterrows():
    print(f"{row['Model']:20} | Acc: {row['Accuracy']:.4f} | "
          f"Prec: {row['Precision']:.4f} | Rec: {row['Recall']:.4f} | "
          f"F1: {row['F1']:.4f}")

# Find models with accuracy > 0.9
high_accuracy = results[results['Accuracy'] > 0.9]
print(f"\nModels with accuracy > 0.9: {len(high_accuracy)}")

# Calculate average metrics
avg_metrics = results[['Accuracy', 'Precision', 'Recall', 'F1']].mean()
print(f"\nAverage metrics across all models:")
for metric, value in avg_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### 12. Comparing Different Datasets

```python
from mlbench_lite import benchmark, load_clover
from sklearn.datasets import load_wine, load_breast_cancer

datasets = [
    ("Clover", load_clover(return_X_y=True)),
    ("Wine", load_wine(return_X_y=True)),
    ("Breast Cancer", load_breast_cancer(return_X_y=True))
]

print("Dataset Comparison:")
print("=" * 80)

for name, (X, y) in datasets:
    print(f"\n{name} Dataset:")
    print(f"  Shape: {X.shape}, Classes: {len(set(y))}")
    
    results = benchmark(X, y)
    best_acc = results.iloc[0]['Accuracy']
    best_model = results.iloc[0]['Model']
    
    print(f"  Best Model: {best_model} (Accuracy: {best_acc:.4f})")
    
    # Show top 2 models
    print("  Top 2 Models:")
    for idx, row in results.head(2).iterrows():
        print(f"    {row['Model']}: {row['Accuracy']:.4f}")
```

## 🔬 Models Included

The library includes **20+ machine learning models** from multiple categories:

### **Linear Models**
- **Logistic Regression**: Linear model for classification using logistic function
- **Ridge Classifier**: Linear classifier with L2 regularization
- **SGD Classifier**: Linear classifier using Stochastic Gradient Descent
- **Perceptron**: Simple linear classifier
- **Passive Aggressive**: Online learning algorithm for classification

### **Tree-based Models**
- **Decision Tree**: Non-parametric supervised learning method
- **Random Forest**: Ensemble of decision trees with bagging
- **Extra Trees**: Extremely randomized trees ensemble
- **Gradient Boosting**: Boosting ensemble method using gradient descent
- **AdaBoost**: Adaptive boosting ensemble method
- **Bagging Classifier**: Bootstrap aggregating ensemble method

### **SVM Models**
- **SVM (RBF)**: Support Vector Machine with RBF kernel
- **SVM (Linear)**: Support Vector Machine with linear kernel

### **Neighbors**
- **K-Nearest Neighbors**: Instance-based learning algorithm

### **Naive Bayes**
- **Gaussian Naive Bayes**: Naive Bayes classifier for Gaussian features
- **Multinomial Naive Bayes**: Naive Bayes classifier for multinomial features
- **Bernoulli Naive Bayes**: Naive Bayes classifier for binary features

### **Discriminant Analysis**
- **Linear Discriminant Analysis**: Linear dimensionality reduction and classification
- **Quadratic Discriminant Analysis**: Quadratic classifier with Gaussian assumptions

### **Neural Networks**
- **Multi-layer Perceptron**: Feedforward artificial neural network

### **Gaussian Process**
- **Gaussian Process**: Probabilistic classifier using Gaussian processes

### **Advanced Gradient Boosting**
- **XGBoost**: Extreme gradient boosting framework (if installed)
- **LightGBM**: Light gradient boosting machine (if installed)
- **CatBoost**: Categorical boosting framework (if installed)

All models use their default parameters with appropriate random seeds for reproducibility.

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

## 🛠️ Requirements

### **Core Dependencies**
- Python >= 3.8
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0

### **Optional Dependencies (for additional models)**
- xgboost >= 1.5.0 (for XGBoost models)
- lightgbm >= 3.2.0 (for LightGBM models)
- catboost >= 1.0.0 (for CatBoost models)
- scikit-optimize >= 0.9.0 (for advanced optimization)

**Note**: The library works with just the core dependencies. Optional dependencies are automatically installed when you install the package, but models from unavailable libraries will be skipped gracefully.

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=mlbench_lite

# Quick functionality test
python -c "from mlbench_lite import benchmark, load_clover; X, y = load_clover(return_X_y=True); results = benchmark(X, y); print(results)"
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
# Format code
black mlbench_lite tests

# Lint code
flake8 mlbench_lite tests

# Type checking
mypy mlbench_lite
```

### Building for Distribution

```bash
# Build package
python -m build

# Upload to PyPI
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

### 2.0.0 (2024-01-XX)
- **MAJOR UPDATE**: Added 20+ machine learning models
- **NEW**: Flexible model selection (specific models, categories, exclusions)
- **NEW**: Support for XGBoost, LightGBM, and CatBoost
- **NEW**: Model information and listing functions
- **NEW**: Comprehensive model categories (Linear, Tree-based, SVM, etc.)
- **IMPROVED**: Enhanced API with more parameters
- **IMPROVED**: Better error handling and graceful degradation
- **IMPROVED**: Updated documentation with extensive examples

### 0.1.0 (2024-01-XX)
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
