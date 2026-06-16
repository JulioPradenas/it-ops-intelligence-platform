"""Tests de integración para la API REST (modelos mockeados)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from itops.api.main import app
from itops.llm.narrative import Narrative


class MockEscalationModel:
    threshold_ = 0.4
    eval_metrics_ = {
        "auc_roc": 0.85, "pr_auc": 0.75, "f1": 0.72,
        "precision": 0.70, "recall": 0.75, "cost_optimal": 100, "cost_at_05": 150,
    }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([0.75] * len(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([True] * len(df))


class MockIFDetector:
    def score(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.5] * len(X))

    def predict(self, X: np.ndarray, percentile: float = 97.0) -> np.ndarray:
        return np.array([False] * len(X))


class MockAEDetector:
    def score(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.3] * len(X))


class MockShapExplainer:
    def top_features(self, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        rows = [
            {
                "feature_1": "response_time_minutes", "shap_1": 0.8,
                "feature_2": "priority_initial", "shap_2": 0.6,
                "feature_3": "num_reassignments", "shap_3": 0.4,
            }
        ] * len(df)
        return pd.DataFrame(rows, index=df.index)


class MockNarrativeGenerator:
    def generate(self, ticket_context: dict, top_features: list[dict]) -> Narrative:
        return Narrative(
            summary="Ticket de alto riesgo detectado.",
            recommendation="Escalar inmediatamente al equipo de soporte nivel 2.",
            confidence=0.8,
            provider="mock",
        )


_TICKET_PAYLOAD = {
    "ticket_id": "T-001",
    "created_at": "2024-01-15T10:00:00",
    "category": "Network",
    "subcategory": "Connectivity",
    "priority_initial": "High",
    "customer_tier": "Gold",
    "description": "Production server unreachable since 09:00",
    "response_time_minutes": 15,
    "num_comments": 3,
    "num_reassignments": 1,
    "business_hours": True,
    "assigned_team": "Network-Ops",
}


@pytest.fixture
def client():
    with (
        patch("itops.api.main.EscalationModel.load", return_value=MockEscalationModel()),
        patch("itops.api.main.IsolationForestDetector.load", return_value=MockIFDetector()),
        patch("itops.api.main.AutoencoderDetector.load", return_value=MockAEDetector()),
        patch("itops.api.main.ShapExplainer", return_value=MockShapExplainer()),
        patch("itops.api.main.NarrativeGenerator", return_value=MockNarrativeGenerator()),
    ):
        with TestClient(app) as c:
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_anomaly_endpoint(client):
    resp = client.post("/anomaly", json={"tickets": [_TICKET_PAYLOAD]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_windows"] >= 1
    window = data["anomalies"][0]
    assert isinstance(window["if_score"], float)
    assert isinstance(window["is_anomaly"], bool)


def test_predict_escalation(client):
    resp = client.post("/predict_escalation", json={"ticket": _TICKET_PAYLOAD})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["predicted_escalation"], bool)


def test_explain_endpoint(client):
    resp = client.post("/explain", json={"ticket": _TICKET_PAYLOAD})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top_features"]) == 3
    assert data["narrative"]["summary"] != ""


def test_invalid_ticket_returns_422(client):
    resp = client.post("/predict_escalation", json={"ticket": {"description": "Missing fields"}})
    assert resp.status_code == 422
