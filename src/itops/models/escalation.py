"""Clasificador de escalación de tickets con LightGBM.

EscalationModel encapsula: feature engineering, split temporal 80/20,
entrenamiento LightGBM y optimización del threshold por costo asimétrico
(FN:FP = 5:1 — un falso negativo cuesta 5 veces más que un falso positivo).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from itops.data.features import build_ticket_features

_COST_FN: int = 5
_COST_FP: int = 1


class EscalationModel:
    """Clasificador LightGBM con threshold optimizado por costo asimétrico."""

    def __init__(self, seed: int = 42, n_estimators: int = 500) -> None:
        self._seed = seed
        self._n_estimators = n_estimators
        self._lgbm_model: lgb.LGBMClassifier | None = None
        self.threshold_: float = 0.5
        self.eval_metrics_: dict = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Entrena el modelo sobre df (se ordena por created_at internamente)."""
        df_sorted = df.sort_values("created_at").reset_index(drop=True)
        X = build_ticket_features(df_sorted)
        y = df_sorted["escalated"].to_numpy()

        split = int(len(df_sorted) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y[:split], y[split:]

        self._lgbm_model = lgb.LGBMClassifier(
            n_estimators=self._n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=self._seed,
            verbose=-1,
            n_jobs=1,
        )
        self._lgbm_model.fit(X_train, y_train)

        val_proba = self._lgbm_model.predict_proba(X_val)[:, 1]

        thresholds = np.arange(0.01, 1.0, 0.01)
        costs = np.array([
            _COST_FN * ((y_val == 1) & (val_proba < t)).sum()
            + _COST_FP * ((y_val == 0) & (val_proba >= t)).sum()
            for t in thresholds
        ])
        self.threshold_ = float(thresholds[np.argmin(costs)])

        preds_opt = (val_proba >= self.threshold_).astype(int)
        preds_05 = (val_proba >= 0.5).astype(int)

        fn_opt = ((y_val == 1) & (preds_opt == 0)).sum()
        fp_opt = ((y_val == 0) & (preds_opt == 1)).sum()
        fn_05 = ((y_val == 1) & (preds_05 == 0)).sum()
        fp_05 = ((y_val == 0) & (preds_05 == 1)).sum()

        self.eval_metrics_ = {
            "auc_roc": float(roc_auc_score(y_val, val_proba)),
            "pr_auc": float(average_precision_score(y_val, val_proba)),
            "f1": float(f1_score(y_val, preds_opt, zero_division=0)),
            "precision": float(precision_score(y_val, preds_opt, zero_division=0)),
            "recall": float(recall_score(y_val, preds_opt, zero_division=0)),
            "cost_optimal": int(_COST_FN * fn_opt + _COST_FP * fp_opt),
            "cost_at_05": int(_COST_FN * fn_05 + _COST_FP * fp_05),
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._lgbm_model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        return self._lgbm_model.predict_proba(build_ticket_features(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold_).astype(bool)

    def save(self, path: Path | str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> EscalationModel:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
