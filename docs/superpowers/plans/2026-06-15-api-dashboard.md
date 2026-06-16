# API y Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the three trained models as a FastAPI REST API and build a 4-view Streamlit dashboard that reads pre-computed parquet for demo reliability.

**Architecture:** Models are serialized to disk via `scripts/train_all.py` and loaded at API startup via lifespan context manager. The Streamlit dashboard reads `data/processed/dashboard_data.parquet` directly — no API dependency — allowing offline demos. Integration tests inject mock models into `app.state` by patching class methods so the lifespan runs cleanly without files on disk.

**Tech Stack:** FastAPI 0.111+, uvicorn 0.30+, Pydantic v2, httpx 0.27+, Streamlit 1.35+, seaborn, matplotlib, LightGBM, PyTorch, SHAP, Anthropic SDK.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add fastapi, uvicorn, httpx, streamlit to main deps; move matplotlib, seaborn to main deps |
| `src/itops/config.py` | Modify | Add `MODELS_DIR` constant |
| `src/itops/models/anomaly.py` | Modify | Add `self._input_dim` to `AutoencoderDetector.fit()`; add `save/load` to both detectors |
| `src/itops/models/escalation.py` | Modify | Add `save/load` to `EscalationModel` |
| `tests/unit/test_serialization.py` | Create | TDD for save/load round-trips on all three models |
| `scripts/train_all.py` | Create | Train + serialize models + generate `dashboard_data.parquet` + `model_metrics.json` |
| `src/itops/api/schemas.py` | Implement | All Pydantic request/response schemas |
| `tests/integration/test_api.py` | Create | 5 integration tests with mock models injected via lifespan patch |
| `src/itops/api/main.py` | Implement | FastAPI app + lifespan (loads models into `app.state`) |
| `src/itops/api/routes.py` | Implement | 4 endpoints: `/health`, `/anomaly`, `/predict_escalation`, `/explain` |
| `src/itops/dashboard/streamlit_app.py` | Implement | 4 views: Operaciones, Compliance, Estratégica, Cómo lo hice |

---

## Task 1: Dependencies + config

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/itops/config.py`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

The current `dependencies` block ends before `[project.optional-dependencies]`. Replace the dependencies list and move matplotlib/seaborn to main deps so the dashboard runs without `--extra notebook`:

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "faker>=25.0",
    "pyarrow>=16.0",
    "scikit-learn>=1.5",
    "torch>=2.3",
    "nbformat>=5.9",
    "lightgbm>=4.3",
    "shap>=0.45",
    "anthropic>=0.30",
    "transformers>=4.40",
    "pydantic>=2.5",
    "fastapi>=0.111",
    "uvicorn>=0.30",
    "httpx>=0.27",
    "streamlit>=1.35",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]
```

Remove `matplotlib>=3.8` and `seaborn>=0.13` from `[project.optional-dependencies] notebook` since they're now in main deps:

```toml
[project.optional-dependencies]
notebook = [
    "jupyter>=1.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "pandas-stubs>=2.2",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
uv sync
```

Expected: packages installed, no errors.

- [ ] **Step 3: Add MODELS_DIR to config.py**

Current `src/itops/config.py` ends at line 16. Add one line:

```python
"""Configuración centralizada de rutas y constantes del proyecto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROCESSED_DIR / "models"

RAW_TICKETS_CSV: Path = RAW_DIR / "tickets_synthetic.csv"
SEEDED_ANOMALIES_JSON: Path = RAW_DIR / "seeded_anomalies.json"

RANDOM_SEED: int = 42
```

- [ ] **Step 4: Verify config import works**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
uv run python -c "from itops.config import MODELS_DIR; print(MODELS_DIR)"
```

Expected: prints `...data/processed/models`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/itops/config.py uv.lock
git commit -m "feat(phase5): add API/dashboard deps and MODELS_DIR config"
```

---

## Task 2: Model serialization (TDD)

**Files:**
- Create: `tests/unit/test_serialization.py`
- Modify: `src/itops/models/anomaly.py`
- Modify: `src/itops/models/escalation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_serialization.py`:

```python
"""Tests TDD para serialización de modelos a disco."""

from __future__ import annotations

import numpy as np
import pytest

from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.data.synthesizer import SynthConfig, generate
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel


@pytest.fixture(scope="module")
def small_df():
    return generate(SynthConfig(n_tickets=3_000, seed=42)).tickets


@pytest.fixture(scope="module")
def X_small(small_df):
    return build_hourly_features(small_df)[FEATURE_COLS].values


def test_escalation_model_save_load(tmp_path, small_df):
    model = EscalationModel(seed=42, n_estimators=50)
    model.fit(small_df)
    path = tmp_path / "escalation.pkl"
    model.save(path)
    loaded = EscalationModel.load(path)

    np.testing.assert_array_equal(model.predict_proba(small_df), loaded.predict_proba(small_df))
    assert loaded.threshold_ == model.threshold_
    assert loaded.eval_metrics_["auc_roc"] == model.eval_metrics_["auc_roc"]


def test_isolation_forest_save_load(tmp_path, X_small):
    detector = IsolationForestDetector(seed=42)
    detector.fit(X_small)
    path = tmp_path / "if_detector.pkl"
    detector.save(path)
    loaded = IsolationForestDetector.load(path)

    np.testing.assert_array_equal(detector.score(X_small), loaded.score(X_small))


def test_autoencoder_input_dim_stored(X_small):
    detector = AutoencoderDetector(epochs=2, seed=42)
    detector.fit(X_small)
    assert hasattr(detector, "_input_dim")
    assert detector._input_dim == X_small.shape[1]


def test_autoencoder_save_load(tmp_path, X_small):
    detector = AutoencoderDetector(epochs=2, seed=42)
    detector.fit(X_small)
    path = tmp_path / "ae_detector.pkl"
    weights_path = tmp_path / "ae_weights.pt"
    detector.save(path, weights_path)
    loaded = AutoencoderDetector.load(path, weights_path)

    np.testing.assert_allclose(detector.score(X_small), loaded.score(X_small), rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest tests/unit/test_serialization.py -v
```

Expected: 4 tests FAIL with `AttributeError: 'EscalationModel' object has no attribute 'save'` (or similar).

- [ ] **Step 3: Add save/load to EscalationModel**

In `src/itops/models/escalation.py`, add `from pathlib import Path` to imports, then add these methods at the end of the `EscalationModel` class (after `predict`):

```python
    def save(self, path: Path | str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "EscalationModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
```

- [ ] **Step 4: Fix AutoencoderDetector.fit() to store _input_dim**

In `src/itops/models/anomaly.py`, add `from pathlib import Path` to imports. In `AutoencoderDetector.fit()`, add `self._input_dim = X.shape[1]` as the first line inside the method body (before `torch.manual_seed`). The updated `fit` method:

```python
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
```

- [ ] **Step 5: Add save/load to IsolationForestDetector and AutoencoderDetector**

In `src/itops/models/anomaly.py`, add save/load to `IsolationForestDetector` (after the `score` method):

```python
    def save(self, path: Path | str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "IsolationForestDetector":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
```

Add save/load to `AutoencoderDetector` (after the `score` method):

```python
    def save(self, path: Path | str, weights_path: Path | str) -> None:
        import pickle
        import torch
        torch.save(self._model.state_dict(), str(weights_path))
        state = {k: v for k, v in self.__dict__.items() if k != "_model"}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str, weights_path: Path | str) -> "AutoencoderDetector":
        import pickle
        import torch
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
```

- [ ] **Step 6: Run serialization tests to verify they pass**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest tests/unit/test_serialization.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -q
```

Expected: 49 tests passing (45 existing + 4 new).

- [ ] **Step 8: Commit**

```bash
git add src/itops/models/anomaly.py src/itops/models/escalation.py tests/unit/test_serialization.py
git commit -m "feat(phase5): add save/load serialization to all three models"
```

---

## Task 3: FastAPI schemas + integration tests (failing)

**Files:**
- Implement: `src/itops/api/schemas.py`
- Create: `tests/integration/test_api.py`

- [ ] **Step 1: Implement schemas.py**

Replace the stub at `src/itops/api/schemas.py` with:

```python
"""Modelos Pydantic v2 de request/response para la API REST."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from itops.llm.narrative import Narrative


class TicketIn(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str
    subcategory: str = "unknown"
    priority_initial: str
    customer_tier: str
    description: str
    response_time_minutes: int
    num_comments: int
    num_reassignments: int
    business_hours: bool
    assigned_team: str


class AnomalyRequest(BaseModel):
    tickets: list[TicketIn]


class AnomalyWindow(BaseModel):
    date: str
    hour: int
    category: str
    ticket_count: int
    if_score: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    anomalies: list[AnomalyWindow]
    total_windows: int


class EscalationRequest(BaseModel):
    ticket: TicketIn


class EscalationResponse(BaseModel):
    ticket_id: str
    risk_score: float
    predicted_escalation: bool
    threshold: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class ExplainRequest(BaseModel):
    ticket: TicketIn


class ExplainResponse(BaseModel):
    ticket_id: str
    risk_score: float
    top_features: list[ShapFeature]
    narrative: Narrative


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
```

- [ ] **Step 2: Write integration tests**

Create `tests/integration/test_api.py`:

```python
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
```

- [ ] **Step 3: Run integration tests to verify they fail**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest tests/integration/test_api.py -v
```

Expected: all 5 tests FAIL — `ImportError` or `AttributeError` because `itops.api.main` is still a stub (no `app` defined).

- [ ] **Step 4: Commit schemas + failing tests**

```bash
git add src/itops/api/schemas.py tests/integration/test_api.py
git commit -m "feat(phase5): add API schemas and integration tests (failing)"
```

---

## Task 4: FastAPI routes + main (make integration tests pass)

**Files:**
- Implement: `src/itops/api/main.py`
- Implement: `src/itops/api/routes.py`

- [ ] **Step 1: Implement routes.py**

Replace the stub at `src/itops/api/routes.py` with:

```python
"""Endpoints de la API REST de IT Ops Intelligence."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request

from itops.api.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    AnomalyWindow,
    EscalationRequest,
    EscalationResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ShapFeature,
)
from itops.data.features import FEATURE_COLS, build_hourly_features

router = APIRouter()


def _ticket_to_df(ticket) -> pd.DataFrame:
    """Convierte un TicketIn a DataFrame de una fila para build_ticket_features."""
    return pd.DataFrame([{
        "ticket_id": ticket.ticket_id,
        "created_at": pd.Timestamp(ticket.created_at),
        "category": ticket.category,
        "subcategory": ticket.subcategory,
        "priority_initial": ticket.priority_initial,
        "customer_tier": ticket.customer_tier,
        "description": ticket.description,
        "response_time_minutes": ticket.response_time_minutes,
        "num_comments": ticket.num_comments,
        "num_reassignments": ticket.num_reassignments,
        "business_hours": ticket.business_hours,
        "assigned_team": ticket.assigned_team,
        "escalated": False,
    }])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=getattr(request.app.state, "models_loaded", False),
    )


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly(body: AnomalyRequest, request: Request) -> AnomalyResponse:
    df = pd.DataFrame([{
        "ticket_id": t.ticket_id,
        "created_at": pd.Timestamp(t.created_at),
        "category": t.category,
        "subcategory": t.subcategory,
        "priority_initial": t.priority_initial,
        "customer_tier": t.customer_tier,
        "description": t.description,
        "response_time_minutes": t.response_time_minutes,
        "num_comments": t.num_comments,
        "num_reassignments": t.num_reassignments,
        "business_hours": t.business_hours,
        "assigned_team": t.assigned_team,
        "escalated": False,
    } for t in body.tickets])

    hourly_feat = build_hourly_features(df)
    X = hourly_feat[FEATURE_COLS].values
    if_scores = request.app.state.if_detector.score(X)
    is_anomaly = request.app.state.if_detector.predict(X, percentile=97.0)

    windows = [
        AnomalyWindow(
            date=str(row["date"]),
            hour=int(row["hour"]),
            category=str(row["category"]),
            ticket_count=int(row["ticket_count"]),
            if_score=float(if_scores[i]),
            is_anomaly=bool(is_anomaly[i]),
        )
        for i, (_, row) in enumerate(hourly_feat.iterrows())
    ]
    return AnomalyResponse(anomalies=windows, total_windows=len(windows))


@router.post("/predict_escalation", response_model=EscalationResponse)
async def predict_escalation(body: EscalationRequest, request: Request) -> EscalationResponse:
    df = _ticket_to_df(body.ticket)
    risk_score = float(request.app.state.escalation_model.predict_proba(df)[0])
    predicted = bool(request.app.state.escalation_model.predict(df)[0])
    return EscalationResponse(
        ticket_id=body.ticket.ticket_id,
        risk_score=risk_score,
        predicted_escalation=predicted,
        threshold=float(request.app.state.escalation_model.threshold_),
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    df = _ticket_to_df(body.ticket)
    risk_score = float(request.app.state.escalation_model.predict_proba(df)[0])

    shap_row = request.app.state.shap_explainer.top_features(df, n=3).iloc[0]
    top_features_api = [
        ShapFeature(
            feature=str(shap_row[f"feature_{i}"]),
            shap_value=float(shap_row[f"shap_{i}"]),
        )
        for i in range(1, 4)
    ]

    ticket_context = {
        "ticket_id": body.ticket.ticket_id,
        "category": body.ticket.category,
        "priority": body.ticket.priority_initial,
        "customer_tier": body.ticket.customer_tier,
        "risk_score": risk_score,
        "description_snippet": body.ticket.description[:200],
    }
    top_features_llm = [
        {"feature": f.feature, "shap": f.shap_value}
        for f in top_features_api
    ]
    narrative = request.app.state.narrative_gen.generate(ticket_context, top_features_llm)

    return ExplainResponse(
        ticket_id=body.ticket.ticket_id,
        risk_score=risk_score,
        top_features=top_features_api,
        narrative=narrative,
    )
```

- [ ] **Step 2: Implement main.py**

Replace the stub at `src/itops/api/main.py` with:

```python
"""FastAPI application con lifespan para IT Ops Intelligence."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from itops.api.routes import router
from itops.config import MODELS_DIR
from itops.llm.narrative import NarrativeGenerator
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel
from itops.models.explainer import ShapExplainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.escalation_model = EscalationModel.load(MODELS_DIR / "escalation_model.pkl")
    app.state.if_detector = IsolationForestDetector.load(MODELS_DIR / "if_detector.pkl")
    app.state.ae_detector = AutoencoderDetector.load(
        MODELS_DIR / "ae_detector.pkl", MODELS_DIR / "ae_weights.pt"
    )
    app.state.shap_explainer = ShapExplainer(app.state.escalation_model)
    app.state.narrative_gen = NarrativeGenerator()
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False


app = FastAPI(title="IT Ops Intelligence API", lifespan=lifespan)
app.include_router(router)
```

- [ ] **Step 3: Run integration tests to verify they pass**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest tests/integration/test_api.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 4: Run full suite to confirm no regressions**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -q
```

Expected: 54 tests passing (49 + 5 integration tests).

- [ ] **Step 5: Commit**

```bash
git add src/itops/api/main.py src/itops/api/routes.py
git commit -m "feat(phase5): implement FastAPI app with 4 endpoints and lifespan"
```

---

## Task 5: Streamlit dashboard (4 views)

**Files:**
- Implement: `src/itops/dashboard/streamlit_app.py`

This task has no unit tests (dashboard is UI). Verification is by reading the parquet in a smoke-test import.

- [ ] **Step 1: Implement streamlit_app.py**

Replace the stub at `src/itops/dashboard/streamlit_app.py` with:

```python
"""Dashboard Streamlit con 4 vistas sobre datos ITSM pre-computados."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from itops.config import PROCESSED_DIR

PARQUET_PATH = PROCESSED_DIR / "dashboard_data.parquet"
METRICS_PATH = PROCESSED_DIR / "model_metrics.json"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def view_operaciones(df: pd.DataFrame) -> None:
    st.header("Operaciones")

    col1, col2, col3, col4 = st.columns(4)
    pct_escalated = df["escalated"].mean() * 100
    pct_anomaly = df["is_anomaly"].fillna(False).mean() * 100
    mttr = df["response_time_minutes"].mean()
    col1.metric("Total tickets", f"{len(df):,}")
    col2.metric("% Escalados", f"{pct_escalated:.1f}%")
    col3.metric("% Anomalías", f"{pct_anomaly:.1f}%")
    col4.metric("MTTR medio (min)", f"{mttr:.0f}")

    st.subheader("Heatmap de anomalías por hora y categoría")
    df_heat = df.copy()
    df_heat["hour"] = pd.to_datetime(df_heat["created_at"]).dt.hour
    heat_data = df_heat.pivot_table(
        index="hour", columns="category",
        values="is_anomaly", aggfunc="mean", fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".1%")
    ax.set_title("Tasa de anomalías por hora y categoría")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 tickets por riesgo")
    top10 = (
        df.nlargest(10, "risk_score")[
            ["ticket_id", "category", "customer_tier", "risk_score", "escalated"]
        ].rename(columns={
            "ticket_id": "Ticket", "category": "Categoría", "customer_tier": "Tier",
            "risk_score": "Riesgo", "escalated": "Escalado",
        })
    )
    st.dataframe(top10, use_container_width=True)


def view_compliance(df: pd.DataFrame) -> None:
    st.header("Compliance")

    st.subheader("Tickets predichos como escalación")
    pred_esc = df[df["predicted_escalation"]][
        ["ticket_id", "category", "customer_tier", "risk_score", "priority_initial", "assigned_team"]
    ]
    st.dataframe(pred_esc, use_container_width=True)

    st.subheader("Tendencia mensual de escalaciones por tier")
    df_monthly = df.copy()
    df_monthly["month"] = pd.to_datetime(df_monthly["created_at"]).dt.to_period("M").astype(str)
    trend = (
        df_monthly[df_monthly["escalated"]]
        .groupby(["month", "customer_tier"])
        .size()
        .unstack(fill_value=0)
    )
    st.line_chart(trend)

    st.subheader("Tiempo de respuesta por prioridad")
    fig, ax = plt.subplots(figsize=(8, 4))
    priorities = sorted(df["priority_initial"].dropna().unique())
    ax.boxplot(
        [df[df["priority_initial"] == p]["response_time_minutes"].dropna() for p in priorities],
        labels=priorities,
    )
    ax.set_xlabel("Prioridad")
    ax.set_ylabel("Tiempo de respuesta (min)")
    ax.set_title("Distribución tiempo de respuesta por prioridad")
    st.pyplot(fig)
    plt.close(fig)


def view_estrategica(df: pd.DataFrame, metrics: dict) -> None:
    st.header("Estratégica")

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
    col2.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    col3.metric("Threshold óptimo", f"{metrics.get('threshold', 0):.2f}")

    st.subheader("Evolución % escalación semanal")
    df_weekly = df.copy()
    df_weekly["week"] = pd.to_datetime(df_weekly["created_at"]).dt.to_period("W").astype(str)
    weekly_rate = df_weekly.groupby("week")["escalated"].mean() * 100
    st.area_chart(weekly_rate.rename("% escalación"))

    st.subheader("Costo estimado evitado")
    tp = int((df["escalated"] & df["predicted_escalation"]).sum())
    st.metric("Verdaderos positivos detectados", f"{tp:,}")
    st.metric("Costo estimado evitado (USD)", f"${tp * 500:,.0f}")
    st.caption("Asunción: cada escalación detectada a tiempo evita $500 en costo operacional.")


def view_como_lo_hice() -> None:
    st.header("Cómo lo hice")
    st.caption(
        "Esta vista está diseñada para que recruiters y colegas entiendan el proyecto sin leer código."
    )

    with st.expander("Problema de negocio"):
        st.write(
            "Las operaciones IT reciben miles de tickets diariamente. Sin visibilidad temprana "
            "de qué tickets van a escalar, los equipos reaccionan en lugar de anticipar — "
            "aumentando tiempos de resolución, costos y frustración del cliente. "
            "Este proyecto construye un sistema de inteligencia operacional que detecta anomalías "
            "en el volumen de incidentes, predice qué tickets escalarán y genera explicaciones "
            "en lenguaje natural para los equipos de soporte."
        )

    with st.expander("Arquitectura"):
        st.markdown(
            "**Fase 1-2 — Datos y anomalías:** 50k tickets sintéticos con patrones realistas. "
            "Isolation Forest (baseline) y Autoencoder MLP en PyTorch para ventanas horarias.\n\n"
            "**Fase 3 — Predicción de escalación:** LightGBM con split temporal 80/20 y threshold "
            "optimizado por costo asimétrico (FN:FP = 5:1). SHAP TreeExplainer por ticket.\n\n"
            "**Fase 4 — Narrativas LLM:** Claude Haiku genera resúmenes en español con fallback a "
            "flan-t5-small para entornos offline. SQLite para caché de deduplicación.\n\n"
            "**Fase 5 — API y Dashboard:** FastAPI expone los modelos. Streamlit consume parquet "
            "pre-computado — sin dependencia de la API para demos."
        )

    with st.expander("Stack técnico"):
        st.table(pd.DataFrame({
            "Capa": [
                "Datos", "ML Anomalías", "ML Escalación",
                "Explicabilidad", "LLM", "API", "Dashboard",
            ],
            "Herramienta": [
                "pandas, Faker",
                "scikit-learn (Isolation Forest), PyTorch (Autoencoder MLP)",
                "LightGBM",
                "SHAP TreeExplainer",
                "Claude Haiku (primario), flan-t5-small (fallback offline)",
                "FastAPI, uvicorn, Pydantic v2",
                "Streamlit, seaborn, matplotlib",
            ],
        }))

    with st.expander("Decisiones clave"):
        st.markdown(
            "- **LightGBM sobre XGBoost** — soporte nativo de categóricas; `n_jobs=1` evita "
            "el conflicto OpenMP con PyTorch en macOS.\n"
            "- **Threshold por costo asimétrico** — FN:FP = 5:1. Un ticket que escala sin "
            "detectarse cuesta 5× más que una falsa alarma.\n"
            "- **Claude + fallback HF** — calidad de narrativas con Haiku; flan-t5-small "
            "permite ejecutar en CI/offline sin API key.\n"
            "- **Dashboard independiente de la API** — el parquet pre-computado permite demos "
            "sin levantar el servidor; más robusto para presentaciones."
        )

    with st.expander("Métricas del modelo"):
        metrics = load_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 'N/A'):.3f}")
            col2.metric("PR-AUC", f"{metrics.get('pr_auc', 'N/A'):.3f}")
            col1.metric("F1 Score", f"{metrics.get('f1', 'N/A'):.3f}")
            col2.metric("Threshold óptimo", f"{metrics.get('threshold', 'N/A'):.2f}")
        else:
            st.info("Ejecuta `scripts/train_all.py` para generar las métricas.")

    with st.expander("Código fuente"):
        st.markdown(
            "Repositorio completo: "
            "[JulioPradenas/it-ops-intelligence-platform]"
            "(https://github.com/JulioPradenas/it-ops-intelligence-platform)"
        )


def main() -> None:
    st.set_page_config(page_title="IT Ops Intelligence", layout="wide")
    st.title("IT Operations Intelligence Platform")

    vista = st.sidebar.radio(
        "Vista", ["Operaciones", "Compliance", "Estratégica", "Cómo lo hice"]
    )

    if vista == "Cómo lo hice":
        view_como_lo_hice()
        return

    df = load_data()
    metrics = load_metrics()

    if vista == "Operaciones":
        view_operaciones(df)
    elif vista == "Compliance":
        view_compliance(df)
    elif vista == "Estratégica":
        view_estrategica(df, metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the import**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
uv run python -c "from itops.dashboard.streamlit_app import main; print('OK')"
```

Expected: prints `OK` with no import errors.

- [ ] **Step 3: Commit**

```bash
git add src/itops/dashboard/streamlit_app.py
git commit -m "feat(phase5): implement Streamlit dashboard with 4 views"
```

---

## Task 6: scripts/train_all.py (generate artifacts)

**Files:**
- Create: `scripts/train_all.py`

This script trains all models, serializes them, and generates `dashboard_data.parquet` + `model_metrics.json`. Verification is by running the script and inspecting the outputs.

- [ ] **Step 1: Verify raw data exists**

```bash
ls -lh "/Users/julio/Desktop/IT Operations Intelligence Platform/data/raw/"
```

Expected: `tickets_synthetic.csv` present. If not, run `uv run python scripts/generate_data.py` first.

- [ ] **Step 2: Create scripts/train_all.py**

```python
"""Entrena todos los modelos, los serializa y genera dashboard_data.parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itops.config import MODELS_DIR, PROCESSED_DIR, RAW_TICKETS_CSV
from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    df = pd.read_csv(RAW_TICKETS_CSV, parse_dates=["created_at"])
    print(f"  {len(df):,} tickets cargados")

    # --- Escalation model ---
    print("Entrenando EscalationModel...")
    escalation_model = EscalationModel(seed=42)
    escalation_model.fit(df)
    escalation_model.save(MODELS_DIR / "escalation_model.pkl")
    print(f"  AUC-ROC:  {escalation_model.eval_metrics_['auc_roc']:.4f}")
    print(f"  PR-AUC:   {escalation_model.eval_metrics_['pr_auc']:.4f}")
    print(f"  Threshold: {escalation_model.threshold_:.3f}")

    # --- Anomaly detectors ---
    print("Construyendo features horarias...")
    hourly_feat = build_hourly_features(df)
    X = hourly_feat[FEATURE_COLS].values

    print("Entrenando IsolationForestDetector...")
    if_detector = IsolationForestDetector(seed=42)
    if_detector.fit(X)
    if_detector.save(MODELS_DIR / "if_detector.pkl")

    print("Entrenando AutoencoderDetector...")
    ae_detector = AutoencoderDetector(seed=42)
    ae_detector.fit(X)
    ae_detector.save(MODELS_DIR / "ae_detector.pkl", MODELS_DIR / "ae_weights.pt")

    # --- Pre-compute predictions ---
    print("Pre-computando predicciones de escalación...")
    df_sorted = df.sort_values("created_at").reset_index(drop=True)
    risk_scores = escalation_model.predict_proba(df_sorted)
    predicted_escalation = escalation_model.predict(df_sorted)

    print("Pre-computando anomalías por ventana...")
    hourly_all = build_hourly_features(df_sorted)
    X_all = hourly_all[FEATURE_COLS].values
    if_scores = if_detector.score(X_all)
    ae_scores = ae_detector.score(X_all)
    is_anomaly = if_detector.predict(X_all, percentile=97.0)

    hourly_all["if_score"] = if_scores
    hourly_all["ae_score"] = ae_scores
    hourly_all["is_anomaly"] = is_anomaly

    # Build ticket-level data and join with window anomaly scores
    dashboard_df = df_sorted[
        ["ticket_id", "created_at", "category", "customer_tier", "escalated",
         "response_time_minutes", "priority_initial", "assigned_team"]
    ].copy()
    dashboard_df["risk_score"] = risk_scores
    dashboard_df["predicted_escalation"] = predicted_escalation.astype(bool)
    dashboard_df["_date"] = dashboard_df["created_at"].dt.date
    dashboard_df["_hour"] = dashboard_df["created_at"].dt.hour

    window_data = hourly_all[
        ["date", "hour", "category", "if_score", "ae_score", "is_anomaly"]
    ].rename(columns={"date": "_date", "hour": "_hour"})

    merged = dashboard_df.merge(
        window_data, on=["_date", "_hour", "category"], how="left"
    ).drop(columns=["_date", "_hour"])

    parquet_path = PROCESSED_DIR / "dashboard_data.parquet"
    merged.to_parquet(parquet_path, index=False)
    print(f"  Parquet guardado: {parquet_path} ({len(merged):,} filas, {merged.shape[1]} columnas)")

    # --- Save metrics JSON ---
    metrics = {**escalation_model.eval_metrics_, "threshold": escalation_model.threshold_}
    metrics_path = PROCESSED_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Métricas guardadas: {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run train_all.py**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run python scripts/train_all.py
```

Expected output (values approximate):
```
Cargando datos...
  50,000 tickets cargados
Entrenando EscalationModel...
  AUC-ROC:  0.8xxx
  PR-AUC:   0.7xxx
  Threshold: 0.xxx
Construyendo features horarias...
Entrenando IsolationForestDetector...
Entrenando AutoencoderDetector...
Pre-computando predicciones de escalación...
Pre-computando anomalías por ventana...
  Parquet guardado: .../dashboard_data.parquet (50000 filas, 13 columnas)
  Métricas guardadas: .../model_metrics.json
Done.
```

- [ ] **Step 4: Verify artifacts exist**

```bash
ls -lh "/Users/julio/Desktop/IT Operations Intelligence Platform/data/processed/models/"
ls -lh "/Users/julio/Desktop/IT Operations Intelligence Platform/data/processed/dashboard_data.parquet"
ls -lh "/Users/julio/Desktop/IT Operations Intelligence Platform/data/processed/model_metrics.json"
```

Expected: `escalation_model.pkl`, `if_detector.pkl`, `ae_detector.pkl`, `ae_weights.pt` in models/; parquet and metrics JSON present.

- [ ] **Step 5: Verify parquet schema**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/dashboard_data.parquet')
print(df.shape)
print(df.columns.tolist())
print(df[['risk_score','predicted_escalation','if_score','is_anomaly']].describe())
"
```

Expected: 50000 rows, columns include `risk_score`, `predicted_escalation`, `if_score`, `ae_score`, `is_anomaly`.

- [ ] **Step 6: Run full test suite one more time**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -q
```

Expected: 54 tests passing.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_all.py
git commit -m "feat(phase5): add train_all.py - trains models and generates dashboard parquet"
```

---

## Task 7: Final verification + commit + push

**Files:** None (verification and push only)

- [ ] **Step 1: Run the complete test suite**

```bash
cd "/Users/julio/Desktop/IT Operations Intelligence Platform"
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -v --tb=short
```

Expected: 54 tests PASS, 0 failures.

- [ ] **Step 2: Verify API starts (smoke test with models on disk)**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run uvicorn itops.api.main:app --port 8000 &
sleep 4
curl -s http://localhost:8000/health | python3 -m json.tool
kill %1
```

Expected: `{"status": "ok", "models_loaded": true}`

- [ ] **Step 3: Commit all remaining files and push**

```bash
git add src/itops/api/__init__.py src/itops/dashboard/__init__.py
git status
git commit -m "feat(phase5): complete API + dashboard implementation

- FastAPI with 4 endpoints: /health, /anomaly, /predict_escalation, /explain
- Streamlit dashboard with 4 views (Operaciones, Compliance, Estratégica, Cómo lo hice)
- Model serialization (save/load) for EscalationModel, IsolationForestDetector, AutoencoderDetector
- scripts/train_all.py generates models + dashboard_data.parquet + model_metrics.json
- 54 unit + integration tests passing"

git push origin main
```

Expected: push succeeds to `https://github.com/JulioPradenas/it-ops-intelligence-platform`.
