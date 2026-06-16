# Fase 5 — API y Dashboard: Design Spec

**Fecha:** 2026-06-15
**Fase del plan:** 5 de 6
**Estado:** aprobado

---

## Contexto

Con los tres modelos implementados (anomaly detection, escalation predictor, LLM narratives), el siguiente paso es exponer el sistema completo como API REST y construir un dashboard Streamlit de 4 vistas para que equipos de operaciones e interesados puedan interactuar con las predicciones sin tocar código.

**Arquitectura elegida:** API + dashboard independientes. `train_all.py` entrena, serializa modelos a disco y pre-computa predicciones en un parquet. La API carga los modelos al arrancar. El dashboard lee el parquet directamente — no requiere la API corriendo para hacer un demo.

---

## Serialización de modelos

### Métodos save/load a agregar

**`EscalationModel`** (`src/itops/models/escalation.py`):
```python
def save(self, path: Path | str) -> None:
    """Serializa el modelo completo (LGBMClassifier + threshold + eval_metrics)."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(self, f)

@classmethod
def load(cls, path: Path | str) -> "EscalationModel":
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
```

**`IsolationForestDetector`** (`src/itops/models/anomaly.py`):
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

**`AutoencoderDetector`** (`src/itops/models/anomaly.py`) — PREREQUISITO: `fit()` debe guardar `self._input_dim = X.shape[1]` para que `load()` pueda reconstruir el MLP:
```python
def save(self, path: Path | str, weights_path: Path | str) -> None:
    """Serializa scaler + config con pickle; pesos MLP con torch.save."""
    import pickle, torch
    torch.save(self._model.state_dict(), weights_path)
    state = {k: v for k, v in self.__dict__.items() if k != "_model"}
    with open(path, "wb") as f:
        pickle.dump(state, f)

@classmethod
def load(cls, path: Path | str, weights_path: Path | str) -> "AutoencoderDetector":
    import pickle, torch
    with open(path, "rb") as f:
        state = pickle.load(f)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    input_dim = state["_input_dim"]
    obj._model = _MLP(input_dim)
    obj._model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    obj._model.eval()
    return obj
```

### `scripts/train_all.py`

Pasos en orden:

1. Cargar `RAW_TICKETS_CSV`
2. Entrenar `EscalationModel(seed=42)` → `save(MODELS_DIR / "escalation_model.pkl")`
3. `build_hourly_features(df)` → entrenar `IsolationForestDetector(seed=42)` → `save(MODELS_DIR / "if_detector.pkl")`
4. Entrenar `AutoencoderDetector(seed=42)` → `save(MODELS_DIR / "ae_detector.pkl", MODELS_DIR / "ae_weights.pt")`
5. Pre-computar predicciones:
   - `risk_score = escalation_model.predict_proba(df)` (1D array)
   - `predicted_escalation = escalation_model.predict(df)` (bool array)
   - `hourly_feat = build_hourly_features(df)` → `X = hourly_feat[FEATURE_COLS].values`
   - `if_score = if_detector.score(X)` (per window)
   - `ae_score = ae_detector.score(X)` (per window)
   - `is_anomaly = if_detector.predict(X, percentile=97.0)` (bool per window)
6. Merge ticket-level y window-level → guardar `PROCESSED_DIR / "dashboard_data.parquet"`

`MODELS_DIR = PROCESSED_DIR / "models"` — creado si no existe.

**Dashboard parquet schema:**

| Columna | Origen |
|---|---|
| `ticket_id` | directo |
| `created_at` | directo |
| `category` | directo |
| `customer_tier` | directo |
| `escalated` | directo (target real) |
| `risk_score` | EscalationModel.predict_proba |
| `predicted_escalation` | EscalationModel.predict |
| `response_time_minutes` | directo |
| `priority_initial` | directo |
| `assigned_team` | directo |

Columnas de anomalía (join por `date + hour + category`):

| Columna | Origen |
|---|---|
| `if_score` | IsolationForestDetector.score |
| `ae_score` | AutoencoderDetector.score |
| `is_anomaly` | IsolationForestDetector.predict(p97) |

---

## FastAPI (`src/itops/api/`)

### Schemas (`schemas.py`)

```python
class TicketIn(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str
    priority_initial: str
    customer_tier: str
    description: str
    response_time_minutes: int
    num_comments: int
    num_reassignments: int
    business_hours: bool
    assigned_team: str

class AnomalyRequest(BaseModel):
    tickets: list[TicketIn]          # mínimo 1 ticket

class AnomalyWindow(BaseModel):
    date: str                        # ISO date
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
    top_features: list[ShapFeature]   # top 3
    narrative: Narrative               # from itops.llm.narrative import Narrative

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
```

### App con lifespan (`main.py`)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.escalation_model = EscalationModel.load(MODELS_DIR / "escalation_model.pkl")
    app.state.if_detector = IsolationForestDetector.load(MODELS_DIR / "if_detector.pkl")
    app.state.ae_detector = AutoencoderDetector.load(MODELS_DIR / "ae_detector.pkl",
                                                      MODELS_DIR / "ae_weights.pt")
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False

app = FastAPI(title="IT Ops Intelligence API", lifespan=lifespan)
app.include_router(router)
```

### Endpoints (`routes.py`)

```
GET  /health               → HealthResponse
POST /anomaly              → tickets[] → build_hourly_features → IF score → AnomalyResponse
POST /predict_escalation   → ticket → EscalationModel.predict_proba/predict → EscalationResponse
POST /explain              → ticket → predict_proba + ShapExplainer.top_features(n=3)
                              + NarrativeGenerator.generate → ExplainResponse
```

`/explain` usa `NarrativeGenerator` sin `api_key` explícito — usa `ANTHROPIC_API_KEY` del entorno, con fallback HF automático.

---

## Dashboard Streamlit (`src/itops/dashboard/streamlit_app.py`)

Navegación: `st.sidebar.radio("Vista", ["Operaciones", "Compliance", "Estratégica", "Cómo lo hice"])`

Todas las vistas cargan `@st.cache_data` sobre el parquet — sin modelos en memoria.

### Vista 1 — Operaciones
- Métricas: `st.metric` para total tickets, % escalados, % anomalías, MTTR medio
- Heatmap anomalías por `(hour_of_day, category)` — `seaborn.heatmap` vía `st.pyplot`
- Tabla top-10 tickets por `risk_score` con columnas ticket_id, categoría, tier, score, escalado real

### Vista 2 — Compliance
- Tickets predichos como escalación: tabla filtrable
- Tendencia mensual de escalaciones por `customer_tier` — `st.line_chart`
- Distribución `response_time_minutes` por `priority_initial` — boxplot vía `st.pyplot`

### Vista 3 — Estratégica
- KPIs del modelo: AUC-ROC, PR-AUC, threshold óptimo (leídos del parquet metadata o constantes)
- Evolución % escalación semanal — `st.area_chart`
- Costo estimado evitado: `true_positives × $500` con nota de asunción visible

### Vista 4 — Cómo lo hice
Secciones con `st.expander`:
- **Problema de negocio** — descripción del ITSM challenge
- **Arquitectura** — descripción textual de las 4 fases
- **Stack técnico** — tabla: Capa | Herramienta
- **Decisiones clave** — 4 bullet points con las ADR más importantes
- **Métricas del modelo** — AUC-ROC, PR-AUC, cobertura de anomalías
- **Código fuente** — link a `https://github.com/JulioPradenas/it-ops-intelligence-platform`

---

## Tests de integración (`tests/integration/test_api.py`)

Fixture `client` inyecta modelos mock en `app.state` via override de lifespan:

```python
@pytest.fixture
def client():
    app.state.escalation_model = MockEscalationModel()
    app.state.if_detector = MockIFDetector()
    app.state.ae_detector = MockAEDetector()
    app.state.models_loaded = True
    with TestClient(app) as c:
        yield c
```

| Test | Verifica |
|---|---|
| `test_health_ok` | 200, `models_loaded=True` |
| `test_anomaly_endpoint` | 200, `total_windows >= 1`, tipos correctos |
| `test_predict_escalation` | 200, `risk_score` ∈ [0,1], `predicted_escalation` bool |
| `test_explain_endpoint` | 200, `len(top_features) == 3`, `narrative.summary` no vacío |
| `test_invalid_ticket_returns_422` | ticket sin `ticket_id` → 422 |

---

## Dependencias nuevas

```toml
"fastapi>=0.111",
"uvicorn>=0.30",
"httpx>=0.27",
"streamlit>=1.35",
```

---

## Archivos a crear/modificar

| Archivo | Acción |
|---|---|
| `src/itops/models/anomaly.py` | Agregar `save/load` a `IsolationForestDetector` y `AutoencoderDetector` |
| `src/itops/models/escalation.py` | Agregar `save/load` a `EscalationModel` |
| `scripts/train_all.py` | Crear: entrenar + serializar + generar parquet |
| `src/itops/api/schemas.py` | Implementar schemas Pydantic |
| `src/itops/api/routes.py` | Implementar 4 endpoints |
| `src/itops/api/main.py` | FastAPI app con lifespan |
| `src/itops/dashboard/streamlit_app.py` | 4 vistas con sidebar |
| `tests/integration/test_api.py` | Crear con 5 tests |
| `pyproject.toml` | Agregar fastapi, uvicorn, httpx, streamlit |

**No se tocan:** synthesizer.py, features.py, anomaly.py (lógica ML), escalation.py (lógica ML), explainer.py, narrative.py, tests existentes (45 pasando).
