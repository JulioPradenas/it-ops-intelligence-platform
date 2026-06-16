"""Detectores de anomalías sobre ventanas horarias de tickets.

IsolationForestDetector: baseline rápido con scikit-learn.
AutoencoderDetector: MLP PyTorch entrenado en ventanas normales.

Ambos implementan AnomalyDetector (ABC) con fit / score / predict.
La columna 0 de X debe ser ticket_count (ver FEATURE_COLS en features.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector(ABC):
    """Protocolo común: fit entrena, score devuelve anomalía (mayor = peor), predict umbraliza."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        scores = self.score(X)
        threshold = np.percentile(scores, percentile)
        return (scores >= threshold).astype(bool)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest con StandardScaler interno. Reproducible vía seed."""

    def __init__(self, n_estimators: int = 100, seed: int = 42) -> None:
        self._scaler = StandardScaler()
        self._model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: más negativo = más anómalo; invertimos para consistencia.
        return -self._model.decision_function(self._scaler.transform(X))

    def save(self, path: Path | str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> IsolationForestDetector:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class _MLP(nn.Module):
    """Autoencoder MLP: input_dim → 32 → 16 → 8 → 16 → 32 → input_dim."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder MLP en PyTorch.

    Se entrena solo en ventanas con ticket_count < percentil 95 para que el
    error de reconstrucción sea bajo en la distribución base y alto en bursts.
    Columna 0 de X debe ser ticket_count (contrato con FEATURE_COLS).
    """

    def __init__(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._seed = seed
        self._scaler = StandardScaler()
        self._model: _MLP | None = None

    def fit(self, X: np.ndarray) -> None:
        self._input_dim = X.shape[1]
        torch.manual_seed(self._seed)
        normal_mask = X[:, 0] < np.percentile(X[:, 0], 95)
        Xs = self._scaler.fit_transform(X[normal_mask])

        self._model = _MLP(Xs.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()
        data = torch.FloatTensor(Xs)

        gen = torch.Generator()
        gen.manual_seed(self._seed)
        self._model.train()
        for _ in range(self._epochs):
            perm = torch.randperm(len(data), generator=gen)
            for i in range(0, len(data), self._batch_size):
                batch = data[perm[i : i + self._batch_size]]
                optimizer.zero_grad()
                loss = criterion(self._model(batch), batch)
                loss.backward()
                optimizer.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score()")
        tensor = torch.FloatTensor(self._scaler.transform(X))
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
        return ((tensor - recon) ** 2).mean(dim=1).numpy()

    def save(self, path: Path | str, weights_path: Path | str) -> None:
        import pickle

        import torch  # noqa: PLC0415
        if self._model is None:
            raise RuntimeError("Call fit() before save()")
        torch.save(self._model.state_dict(), str(weights_path))
        state = {k: v for k, v in self.__dict__.items() if k != "_model"}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str, weights_path: Path | str) -> AutoencoderDetector:
        import pickle

        import torch  # noqa: PLC0415
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._model = _MLP(state["_input_dim"])
        obj._model.load_state_dict(
            torch.load(str(weights_path), map_location="cpu", weights_only=True)
        )
        obj._model.eval()
        return obj
