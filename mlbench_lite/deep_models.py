"""
Optional deep-learning models for tabular data: a plain PyTorch MLP and a
(simplified) FT-Transformer.

Why only these two, and why they're safe to bolt onto benchmark():
  * Both take the exact same (X, y) tabular contract as every sklearn model
    already in this library -- no new data format, no new eval path.
  * PyTorch is NOT a hard dependency (see pyproject.toml `[deep]` extra).
    Exactly like xgboost/lightgbm/catboost, this module degrades gracefully:
    `TORCH_AVAILABLE` is False and the estimator classes are set to None when
    torch isn't installed, so `import mlbench_lite` never breaks for people
    who don't have torch. benchmark.py only adds the "Deep Tabular" category
    when TORCH_AVAILABLE is True.
  * Defaults are kept small on purpose (few layers, small embedding dims,
    modest epoch counts) so they train in a reasonable time on a Colab
    free-tier CPU/T4, not just on a workstation GPU.

Not included here: GNNs (GCN/GraphSAGE/GAT), CNNs, BiLSTM, PINN, SNN -- see
the project notes for why those don't fit this module's (X, y) tabular
contract without a much bigger, separate data pipeline.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _MLPNet(nn.Module):
        def __init__(self, n_features: int, hidden_dim: int, n_layers: int,
                     dropout: float, n_outputs: int):
            super().__init__()
            layers = []
            in_dim = n_features
            for _ in range(max(n_layers, 1)):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_outputs))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class _FeatureTokenizer(nn.Module):
        """Numeric feature tokenizer: emb_i = x_i * W_i + b_i, per-feature."""

        def __init__(self, n_features: int, emb_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, emb_dim))
            self.bias = nn.Parameter(torch.empty(n_features, emb_dim))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.zeros_(self.bias)

        def forward(self, x):  # x: (batch, n_features)
            return x.unsqueeze(-1) * self.weight + self.bias  # (batch, n_features, emb_dim)

    class _FTTransformerNet(nn.Module):
        """Simplified FT-Transformer: tokenize numeric features, prepend a CLS
        token, run a small Transformer encoder, read out the CLS token."""

        def __init__(self, n_features: int, emb_dim: int, n_layers: int,
                     n_heads: int, dropout: float, n_outputs: int):
            super().__init__()
            self.tokenizer = _FeatureTokenizer(n_features, emb_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 2,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(emb_dim)
            self.head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, n_outputs)
            )

        def forward(self, x):
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            out = self.encoder(tokens)
            cls_out = self.norm(out[:, 0])
            return self.head(cls_out)

    class _TorchTabularMixin:
        """Shared fit/predict machinery for both torch estimators below.

        Deliberately hand-rolled mini-batching (no DataLoader / no
        multiprocessing workers) so that a given `random_state` produces a
        byte-for-byte reproducible shuffling order regardless of platform --
        matters for the library's `random_state` reproducibility guarantee
        (see tests/test_benchmark.py::test_reproducibility).
        """

        def _resolve_device(self) -> "torch.device":
            wants_gpu = str(self.device).lower() in ("cuda", "gpu")
            if wants_gpu:
                if torch.cuda.is_available():
                    return torch.device("cuda")
                warnings.warn(
                    f"[mlbench-lite] GPU requested for {self.__class__.__name__} "
                    "but CUDA is not available; falling back to CPU.",
                    UserWarning, stacklevel=4,
                )
            return torch.device("cpu")

        def _fit_common(self, X, y_arr, n_outputs: int, is_classifier: bool):
            X = np.asarray(X, dtype=np.float32)
            torch.manual_seed(self.random_state)
            device = self._resolve_device()
            net = self._build_net(X.shape[1], n_outputs).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            if is_classifier:
                loss_fn = nn.CrossEntropyLoss()
                y_t = torch.tensor(y_arr, dtype=torch.long)
            else:
                loss_fn = nn.MSELoss()
                y_t = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1)
            X_t = torch.tensor(X, dtype=torch.float32)

            n = X_t.shape[0]
            rng = np.random.RandomState(self.random_state)
            best_loss, best_state, patience_left = float("inf"), None, self.patience

            for epoch in range(self.epochs):
                cancel_check = getattr(self, "_cancel_check", None)
                if cancel_check is not None and cancel_check():
                    break
                perm = rng.permutation(n)
                net.train()
                epoch_loss = 0.0
                for start in range(0, n, self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    xb, yb = X_t[idx].to(device), y_t[idx].to(device)
                    opt.zero_grad()
                    loss = loss_fn(net(xb), yb)
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item() * len(idx)
                epoch_loss /= n
                if self.verbose:
                    print(f"[{self.__class__.__name__}] epoch {epoch + 1}/{self.epochs} "
                          f"loss={epoch_loss:.4f}")
                if epoch_loss < best_loss - 1e-5:
                    best_loss = epoch_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.net_ = net
            self.device_ = device
            return self

        def _predict_raw(self, X) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
            with torch.no_grad():
                out = self.net_(X_t)
            return out.cpu().numpy()

    class TorchMLPClassifier(_TorchTabularMixin, BaseEstimator, ClassifierMixin):
        """Small feedforward PyTorch classifier -- a deep-learning-native
        baseline sitting next to sklearn's MLPClassifier."""

        def __init__(self, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1,
                     lr: float = 1e-3, epochs: int = 60, batch_size: int = 32,
                     weight_decay: float = 1e-5, random_state: int = 42,
                     device: str = "cpu", verbose: bool = False, patience: int = 10):
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.device = device
            self.verbose = verbose
            self.patience = patience

        def _build_net(self, n_features, n_outputs):
            return _MLPNet(n_features, self.hidden_dim, self.n_layers, self.dropout, n_outputs)

        def fit(self, X, y):
            y_arr = np.asarray(y)
            self.classes_ = np.unique(y_arr)
            self._fit_common(X, y_arr, len(self.classes_), is_classifier=True)
            return self

        def predict(self, X):
            logits = self._predict_raw(X)
            return self.classes_[logits.argmax(axis=1)]

        def predict_proba(self, X):
            logits = self._predict_raw(X)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

    class TorchMLPRegressor(_TorchTabularMixin, BaseEstimator, RegressorMixin):
        """Small feedforward PyTorch regressor."""

        def __init__(self, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1,
                     lr: float = 1e-3, epochs: int = 60, batch_size: int = 32,
                     weight_decay: float = 1e-5, random_state: int = 42,
                     device: str = "cpu", verbose: bool = False, patience: int = 10):
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.device = device
            self.verbose = verbose
            self.patience = patience

        def _build_net(self, n_features, n_outputs):
            return _MLPNet(n_features, self.hidden_dim, self.n_layers, self.dropout, n_outputs)

        def fit(self, X, y):
            self._fit_common(X, np.asarray(y, dtype=float), 1, is_classifier=False)
            return self

        def predict(self, X):
            return self._predict_raw(X).ravel()

    class FTTransformerClassifier(_TorchTabularMixin, BaseEstimator, ClassifierMixin):
        """Simplified FT-Transformer for tabular classification. Numeric
        feature tokenizer + a small Transformer encoder + CLS-token head."""

        def __init__(self, hidden_dim: int = 32, n_layers: int = 2, n_heads: int = 4,
                     dropout: float = 0.1, lr: float = 1e-3, epochs: int = 40,
                     batch_size: int = 32, weight_decay: float = 1e-5,
                     random_state: int = 42, device: str = "cpu",
                     verbose: bool = False, patience: int = 10):
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.device = device
            self.verbose = verbose
            self.patience = patience

        def _build_net(self, n_features, n_outputs):
            return _FTTransformerNet(n_features, self.hidden_dim, self.n_layers,
                                      self.n_heads, self.dropout, n_outputs)

        def fit(self, X, y):
            y_arr = np.asarray(y)
            self.classes_ = np.unique(y_arr)
            self._fit_common(X, y_arr, len(self.classes_), is_classifier=True)
            return self

        def predict(self, X):
            logits = self._predict_raw(X)
            return self.classes_[logits.argmax(axis=1)]

        def predict_proba(self, X):
            logits = self._predict_raw(X)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

    class FTTransformerRegressor(_TorchTabularMixin, BaseEstimator, RegressorMixin):
        """Simplified FT-Transformer for tabular regression."""

        def __init__(self, hidden_dim: int = 32, n_layers: int = 2, n_heads: int = 4,
                     dropout: float = 0.1, lr: float = 1e-3, epochs: int = 40,
                     batch_size: int = 32, weight_decay: float = 1e-5,
                     random_state: int = 42, device: str = "cpu",
                     verbose: bool = False, patience: int = 10):
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.device = device
            self.verbose = verbose
            self.patience = patience

        def _build_net(self, n_features, n_outputs):
            return _FTTransformerNet(n_features, self.hidden_dim, self.n_layers,
                                      self.n_heads, self.dropout, n_outputs)

        def fit(self, X, y):
            self._fit_common(X, np.asarray(y, dtype=float), 1, is_classifier=False)
            return self

        def predict(self, X):
            return self._predict_raw(X).ravel()

else:
    TorchMLPClassifier = None
    TorchMLPRegressor = None
    FTTransformerClassifier = None
    FTTransformerRegressor = None
