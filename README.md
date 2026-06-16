# IT Operations Intelligence Platform

[![CI](https://github.com/JulioPradenas/it-ops-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/JulioPradenas/it-ops-intelligence-platform/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JulioPradenas/it-ops-intelligence-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/JulioPradenas/it-ops-intelligence-platform)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)

Plataforma end-to-end de inteligencia operacional para datos ITSM. Combina detección de anomalías, scoring de riesgo de escalación y narrativas generadas con LLM — expuesto via FastAPI y visualizado en un dashboard Streamlit por audiencia.

---

## Problema de negocio

Los equipos de IT Operations reciben miles de tickets diariamente. Sin visibilidad temprana de cuáles van a escalar, los equipos reaccionan en lugar de anticipar — aumentando tiempos de resolución, costos y frustración del cliente.

Este proyecto construye un sistema que:
1. **Detecta anomalías** en el volumen horario de incidentes (¿hay un spike inusual ahora?)
2. **Predice escalaciones** por ticket antes de que ocurran (¿este ticket va a escalar?)
3. **Explica en lenguaje natural** qué factores están impulsando el riesgo (¿por qué?)

---

## Arquitectura

```
data/raw/
  tickets_synthetic.csv     ← 50k tickets con patrones realistas (Faker)

src/itops/
  data/
    synthesizer.py          ← generación de datos sintéticos
    features.py             ← pipeline de features (ticket-level + window-level)
  models/
    anomaly.py              ← IsolationForest + Autoencoder PyTorch
    escalation.py           ← LightGBM con threshold por costo asimétrico
    explainer.py            ← SHAP TreeExplainer
  llm/
    narrative.py            ← NarrativeGenerator (Claude → Groq → Template)
    prompts.py              ← build_escalation_prompt()
  api/
    main.py                 ← FastAPI app (lifespan + model loading)
    routes.py               ← /health, /anomaly, /predict_escalation, /explain
    schemas.py              ← Pydantic v2 schemas
  dashboard/
    streamlit_app.py        ← 4 vistas: Operaciones, Compliance, Estratégica, Cómo lo hice

scripts/
  generate_data.py          ← genera tickets_synthetic.csv
  train_all.py              ← entrena modelos + guarda parquet + MLflow tracking
```

**Flujo:** `generate_data.py` → `train_all.py` → modelos serializados + parquet → API/Dashboard

---

## Stack técnico

| Capa | Herramienta |
|---|---|
| Datos | pandas, Faker |
| Anomalías | scikit-learn (Isolation Forest), PyTorch (Autoencoder MLP) |
| Escalación | LightGBM, SHAP TreeExplainer |
| LLM | Claude Haiku (primario), Groq llama-3.1-8b-instant (fallback), Template (offline) |
| API | FastAPI, uvicorn, Pydantic v2 |
| Dashboard | Streamlit, seaborn, matplotlib |
| MLOps | MLflow (tracking), pytest-cov, ruff, mypy |
| Infra | Docker multi-stage, GitHub Actions CI |

---

## Resultados

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.863 |
| PR-AUC | 0.419 |
| F1 Score | 0.52 |
| Threshold óptimo (FN:FP = 5:1) | 0.690 |
| Tests | 55 pasando |
| Coverage | ≥ 85% (excl. UI) |

---

## Cómo ejecutar localmente

**Requisitos:** Python 3.11, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/JulioPradenas/it-ops-intelligence-platform.git
cd it-ops-intelligence-platform

# Instalar dependencias
uv sync --extra dev

# Variables de entorno (opcional — activa Groq como fallback LLM)
cp .env.example .env
# Editar .env con tu GROQ_API_KEY

# Generar datos sintéticos
uv run python scripts/generate_data.py

# Entrenar modelos (genera parquet + MLflow run)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run python scripts/train_all.py

# Levantar API
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run uvicorn itops.api.main:app --reload

# Levantar dashboard (nueva terminal)
uv run streamlit run src/itops/dashboard/streamlit_app.py

# Ver experimentos MLflow
uv run mlflow ui
```

**Con Docker:**
```bash
docker compose up
# API en http://localhost:8000
# Dashboard en http://localhost:8501
```

**Tests:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest tests/ --cov=itops -q
```

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servidor y modelos cargados |
| POST | `/anomaly` | Detecta ventanas horarias anómalas |
| POST | `/predict_escalation` | Probabilidad de escalación por ticket |
| POST | `/explain` | SHAP top features + narrativa LLM |

Documentación interactiva: `http://localhost:8000/docs`

---

## Decisiones técnicas clave

**LightGBM sobre XGBoost** — soporte nativo de categóricas y `n_jobs=1` evita el conflicto OpenMP con PyTorch en macOS.

**Threshold por costo asimétrico (FN:FP = 5:1)** — un ticket que escala sin detectarse cuesta 5× más que una falsa alarma. El threshold de 0.69 optimiza este trade-off en lugar de F1 simple.

**Claude → Groq → Template** — calidad con Haiku cuando hay API key; Groq (llama-3.1-8b-instant) como fallback gratuito y rápido; template determinista para CI y entornos offline.

**Dashboard independiente de la API** — el parquet pre-computado permite demos sin levantar el servidor, más robusto para presentaciones. La API solo se necesita para narrativas LLM en tiempo real.

**SQLite para caché de narrativas** — cero infraestructura, persiste entre sesiones. Solo se cachean entradas con `confidence > 0.0` para evitar servir respuestas fallidas.

Los ADRs completos están en [`docs/decisions/`](docs/decisions/).

---

## Limitaciones

- **Datos sintéticos:** los patrones (5% escalación, anomalías sembradas) son controlados. Un modelo entrenado con datos reales ITSM probablemente requeriría feature engineering adicional.
- **Autoencoder:** el modelo MLP no captura dependencias temporales explícitas (sin LSTM/Transformer). Es un baseline razonable para el tamaño del demo.
- **Narrativas:** sin API keys ambas (Claude y Groq), la narrativa template es funcional pero menos fluida que el LLM.
- **Escalabilidad:** SQLite y pickle son adecuados para el volumen del demo. Producción requeriría PostgreSQL y un model registry como MLflow Model Registry o BentoML.
