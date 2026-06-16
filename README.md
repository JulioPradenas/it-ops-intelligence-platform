# IT Operations Intelligence Platform

Plataforma de inteligencia operacional para datos ITSM: detección de anomalías,
scoring de riesgo de escalación de tickets y explicaciones en lenguaje natural
con LLM. Ver el plan completo en [`PLAN_IT_Ops_Intelligence.md`](PLAN_IT_Ops_Intelligence.md).

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Dataset sintético + EDA | en progreso |
| 2 | Detección de anomalías | pendiente |
| 3 | Predictor de escalación | pendiente |
| 4 | Capa LLM de narrativas | pendiente |
| 5 | API + dashboard | pendiente |
| 6 | MLOps + documentación | pendiente |

## Quick start

```bash
uv venv --python 3.11
uv pip install -e ".[dev,notebook]"

# Genera data/raw/tickets_synthetic.csv (50k tickets, ~5% escalados)
python scripts/generate_data.py

# EDA
jupyter lab notebooks/01_eda.ipynb

# Calidad
ruff check .
mypy
pytest --cov=itops
```

## Estructura

Ver sección 4 de [`PLAN_IT_Ops_Intelligence.md`](PLAN_IT_Ops_Intelligence.md). El código
de aplicación vive en `src/itops/`; el dataset y su schema en `data/`.
