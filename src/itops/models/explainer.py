"""Wrapper SHAP para explicabilidad por ticket del EscalationModel.

Usa TreeExplainer (exacto, rápido con LightGBM) y devuelve SHAP values
de la clase positiva (escalation=True) para cada ticket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from itops.data.features import TICKET_FEATURE_COLS, build_ticket_features
from itops.models.escalation import EscalationModel


class ShapExplainer:
    """SHAP TreeExplainer sobre un EscalationModel entrenado."""

    def __init__(self, model: EscalationModel) -> None:
        if model._lgbm_model is None:
            raise RuntimeError("Model must be fitted before creating ShapExplainer")
        self._explainer = shap.TreeExplainer(model._lgbm_model)

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values de la clase positiva (escalation=True) por ticket."""
        features = build_ticket_features(df)
        sv = self._explainer.shap_values(features)
        values = sv[1] if isinstance(sv, list) else sv
        if isinstance(values, np.ndarray) and values.ndim == 3:
            values = values[:, :, 1]
        return pd.DataFrame(values, columns=TICKET_FEATURE_COLS, index=df.index)

    def top_features(self, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """Top-N features por |SHAP value| para cada ticket.

        Columnas: feature_1, shap_1, feature_2, shap_2, ..., feature_N, shap_N.
        """
        shap_df = self.explain(df)
        rows = []
        for _, row in shap_df.iterrows():
            top_n = row.abs().nlargest(n)
            entry: dict = {}
            for i, feat in enumerate(top_n.index, 1):
                entry[f"feature_{i}"] = feat
                entry[f"shap_{i}"] = float(row[feat])
            rows.append(entry)
        return pd.DataFrame(rows, index=df.index)
