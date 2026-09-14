from .benchmark import (
    benchmark,
    get_model_info,
    list_available_models,
    get_available_classifiers,
    get_available_regressors,
)
from .datasets import load_clover
from .preprocessing import recommend_scaling, build_scaler
from .stats import compare_models
from .utils import set_seed
from .deep_models import TORCH_AVAILABLE
__version__ = "3.3.0"
__author__ = "Arefin Amin"
__email__ = "arefinamin994@gmail.com"
__all__ = [
    "benchmark",
    "load_clover",
    "list_available_models",
    "get_model_info",
    "get_available_classifiers",
    "get_available_regressors",
    "recommend_scaling",
    "build_scaler",
    "compare_models",
    "set_seed",
    "TORCH_AVAILABLE",
]
