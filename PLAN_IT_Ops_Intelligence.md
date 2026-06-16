# IT Operations Intelligence Platform

> Plataforma de inteligencia operacional para datos ITSM. Combina detección de anomalías, scoring de riesgo de escalación de tickets y explicabilidad en lenguaje natural mediante LLM. Diseñado para demostrar experiencia hands-on con el stack y los dominios típicos de roles BI & Data Scientist en operaciones de IT.

---

## 1. Contexto y motivación

### Pregunta de negocio

Un equipo de operaciones de IT recibe cientos de tickets diarios. La mayoría se resuelve sin escalar, pero un porcentaje pequeño (típicamente 3-7%) escala a prioridad crítica y genera costos operacionales altos: SLA incumplidos, equipos multidisciplinarios convocados de urgencia, impacto en clientes finales.

**La pregunta concreta:** ¿podemos predecir qué tickets van a escalar a crítico antes de que lo hagan, detectar patrones anómalos en el flujo de tickets, y explicar en lenguaje natural por qué un ticket fue marcado como riesgo, para que el equipo de operaciones actúe a tiempo?

### Por qué este proyecto

Este proyecto está diseñado para cerrar gaps específicos del perfil de candidato y cubrir múltiples requisitos de roles BI & Data Scientist en empresas con foco operacional. En un solo repositorio se demuestra: anomaly detection, predictive analytics, risk scoring, LLM aplicado, MLOps end-to-end y comunicación con stakeholders no técnicos.

### Mapeo con job description típico

| Requisito del cargo | Cómo lo cubre el proyecto |
|---|---|
| Anomaly identification | Isolation Forest + Autoencoder sobre patrones de tickets |
| Predictive analytics + Risk scoring | Clasificador de escalación con LightGBM |
| LLM applications (LangChain, Hugging Face) | Capa de explicación en lenguaje natural |
| ITSM data integration (ServiceNow-like) | Schema sintético realista de tickets |
| Deep learning (PyTorch) | Autoencoder para anomaly detection |
| Feature engineering + cross-validation | Pipeline completo con validación temporal |
| Root cause analysis | SHAP por ticket: factores que contribuyeron al riesgo |
| MLOps pipelines | MLflow, FastAPI, CI/CD, tests automatizados |
| Power BI / Tableau dashboards | Dashboard de operaciones conectado al pipeline |
| Linux environments | Docker + scripts bash para reproducibilidad |

---

## 2. Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS (Storage)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ tickets.csv │→ │  SQLite DB  │→ │  Marts agregados        │ │
│  │   (raw)     │  │ (operacional)│  │ (predicciones + alertas)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAPA DE MODELOS (ML / DL / LLM)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐    │
│  │  Anomaly     │ │  Escalation  │ │  LLM Narrative       │    │
│  │  Detection   │ │  Predictor   │ │  Generator           │    │
│  │  (Isolation  │ │  (LightGBM   │ │  (Claude API /       │    │
│  │   Forest +   │ │   + SHAP)    │ │   Hugging Face)      │    │
│  │   Autoenc.)  │ │              │ │                      │    │
│  └──────────────┘ └──────────────┘ └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAPA DE SERVING (FastAPI)                           │
│   POST /anomaly         POST /predict_escalation                 │
│   POST /explain         GET  /health                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAPA DE PRESENTACIÓN                                │
│  Power BI dashboard / Streamlit fallback                         │
│  - KPIs operacionales                                            │
│  - Anomalías detectadas hoy                                      │
│  - Top tickets en riesgo de escalación                           │
│  - Explicaciones LLM por ticket                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              MLOPS (transversal)                                 │
│  MLflow (tracking) · pytest · GitHub Actions (CI/CD) · Docker    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Stack técnico

| Capa | Herramientas |
|---|---|
| Lenguaje | Python 3.11 |
| Gestión de dependencias | uv (recomendado) o pip |
| Datos | pandas, SQLAlchemy, SQLite |
| ML clásico | scikit-learn, LightGBM |
| Deep Learning | PyTorch (Autoencoder para anomaly detection) |
| Explicabilidad | SHAP |
| LLM | Claude API (anthropic) o Hugging Face transformers |
| API | FastAPI, Pydantic v2 |
| Tracking | MLflow |
| Dashboard | Power BI (principal) + Streamlit (fallback en CI/web) |
| Tests | pytest, pytest-cov |
| Calidad | ruff, mypy, pre-commit |
| Infraestructura | Docker, GitHub Actions |

---

## 4. Estructura del repositorio

```
it-ops-intelligence/
├── README.md                        # Documentación principal
├── pyproject.toml                   # Dependencias y configuración
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                   # Tests + lint + typecheck
├── .pre-commit-config.yaml
├── data/
│   ├── raw/
│   │   └── tickets_synthetic.csv
│   ├── processed/
│   │   └── tickets_features.parquet
│   └── README.md                    # Explicación del schema
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploración inicial
│   ├── 02_anomaly_detection.ipynb   # Iteración del modelo anomaly
│   ├── 03_escalation_model.ipynb    # Iteración del clasificador
│   └── 04_llm_explanations.ipynb    # Prototipo de narrativas
├── src/
│   └── itops/
│       ├── __init__.py
│       ├── config.py                # Configuración centralizada
│       ├── data/
│       │   ├── __init__.py
│       │   ├── synthesizer.py       # Generador del dataset sintético
│       │   ├── loader.py            # Carga desde CSV / SQLite
│       │   └── features.py          # Feature engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── anomaly.py           # Isolation Forest + Autoencoder
│       │   ├── escalation.py        # Clasificador de escalación
│       │   └── explainer.py         # SHAP wrapper
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── narrative.py         # Generación de explicaciones
│       │   └── prompts.py           # Templates de prompts
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py              # FastAPI app
│       │   ├── routes.py
│       │   └── schemas.py           # Pydantic models
│       ├── dashboard/
│       │   └── streamlit_app.py     # Dashboard fallback
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── mlflow_utils.py
├── tests/
│   ├── unit/
│   │   ├── test_synthesizer.py
│   │   ├── test_features.py
│   │   ├── test_anomaly.py
│   │   ├── test_escalation.py
│   │   └── test_explainer.py
│   ├── integration/
│   │   └── test_api.py
│   └── conftest.py
├── powerbi/
│   ├── IT_Ops_Intelligence.pbix     # Archivo Power BI
│   └── README.md                    # Cómo conectar
├── scripts/
│   ├── generate_data.py             # Genera dataset sintético
│   ├── train_all.py                 # Pipeline de entrenamiento end-to-end
│   └── deploy_local.sh
└── docs/
    ├── architecture.md
    ├── decisions/                   # ADRs (Architecture Decision Records)
    │   ├── 001_synthetic_data.md
    │   ├── 002_autoencoder_for_anomaly.md
    │   ├── 003_lightgbm_over_xgboost.md
    │   └── 004_llm_provider.md
    └── api_reference.md
```

---

## 5. Plan de implementación por fases

> **Estimación total: 10-12 días de trabajo concentrado.** Cada fase es funcional por sí sola — si tienes que parar al final de cualquier fase, el proyecto sigue siendo defendible en una entrevista.

### Fase 1 — Dataset sintético + EDA (días 1-2)

**Objetivo:** generar un dataset realista de tickets ITSM y entender sus patrones.

**Entregables:**
- `src/itops/data/synthesizer.py` con generador parametrizable
- `data/raw/tickets_synthetic.csv` (~50,000 tickets, ~5% escalados)
- `notebooks/01_eda.ipynb` con análisis exploratorio completo
- `data/README.md` documentando el schema

**Schema del dataset (mínimo viable):**

```
ticket_id              VARCHAR  # ID único
created_at             TIMESTAMP
closed_at              TIMESTAMP (nullable)
category               VARCHAR  # network, hardware, software, access, other
subcategory            VARCHAR  # más granular
priority_initial       VARCHAR  # low, medium, high
priority_final         VARCHAR  # low, medium, high, critical
assigned_team          VARCHAR  # team_a, team_b, ...
assignee_id            VARCHAR
customer_tier          VARCHAR  # basic, standard, premium, enterprise
description            TEXT     # texto libre del ticket
response_time_minutes  INT      # tiempo a primera respuesta
num_comments           INT      # comentarios en primera hora
num_reassignments      INT      # veces que cambió de equipo
business_hours         BOOLEAN  # creado en horario laboral
escalated              BOOLEAN  # target principal
hours_to_escalation    FLOAT    # nullable, solo si escalated
```

**Patrones realistas a inyectar en el sintetizador:**
- Estacionalidad horaria (más tickets 9-11am y 14-16pm)
- Estacionalidad semanal (peak los lunes)
- Tickets de clientes enterprise escalan 3x más que basic
- Tickets reasignados >2 veces tienen 50% probabilidad de escalar
- Categoría network tiene tasa de escalación más alta
- Anomalías sembradas: días con burst de tickets de la misma categoría (simula incidente sistémico)

**Tests a escribir:**
- El generador produce el ratio de escalación esperado (±2%)
- El schema cumple con los tipos definidos
- Las anomalías sembradas son detectables visualmente

---

### Fase 2 — Detección de anomalías (días 3-4)

**Objetivo:** detectar patrones anómalos en el flujo de tickets — ejemplo: aumento súbito de tickets de una categoría, picos en horario no laboral, clusters geográficos o por equipo.

**Entregables:**
- `src/itops/models/anomaly.py` con dos modelos:
  - `IsolationForestDetector`: baseline rápido y explicable
  - `AutoencoderDetector`: modelo en PyTorch para capturar patrones complejos
- `notebooks/02_anomaly_detection.ipynb` con comparación
- `docs/decisions/002_autoencoder_for_anomaly.md` con la justificación

**Decisiones técnicas a documentar:**
- Por qué dos modelos (Isolation Forest como baseline obligatorio antes de complejidad)
- Cómo se define "anómalo" operacionalmente (umbral por percentil del score)
- Granularidad de detección (por hora, por día, por categoría)

**Tests:**
- El autoencoder se entrena sin errores en datos sintéticos
- El score de anomalía es reproducible con la misma semilla
- Las anomalías inyectadas en el dataset son detectadas

---

### Fase 3 — Predictor de escalación (días 5-6)

**Objetivo:** clasificador que predice qué tickets escalarán a crítico en las próximas N horas, con explicabilidad por SHAP.

**Entregables:**
- `src/itops/models/escalation.py` con LightGBM
- `src/itops/models/explainer.py` con wrapper SHAP
- `notebooks/03_escalation_model.ipynb` con comparación de modelos
- Métricas evaluadas: AUC-ROC, Precision-Recall, F1 ponderado, matriz de confusión por categoría
- Umbral optimizado por **costo asimétrico del error** (falso negativo cuesta más que falso positivo en escalaciones)

**Feature engineering crítico:**
- Tiempo desde creación hasta primera respuesta
- Velocidad de comentarios en primera hora
- Tier del cliente
- Categoría + subcategoría
- Volumen de tickets activos del mismo equipo
- Hora del día y día de la semana
- Features de texto (longitud descripción, presencia de keywords críticas como "down", "production")

**Validación:**
- Validación cruzada temporal (NUNCA aleatoria con datos de tickets)
- Métricas reportadas en holdout temporal

**Tests:**
- El modelo se entrena sin error
- SHAP genera explicaciones por ticket
- Las predicciones son reproducibles
- El umbral optimizado mejora el costo total vs umbral 0.5

---

### Fase 4 — Capa LLM para narrativas (días 7-8)

**Objetivo:** generar explicaciones en lenguaje natural para cada alerta o predicción de alto riesgo, accesibles para equipos de operaciones no técnicos.

**Entregables:**
- `src/itops/llm/narrative.py` con generador de narrativas
- `src/itops/llm/prompts.py` con templates parametrizables
- `notebooks/04_llm_explanations.ipynb` con ejemplos
- `docs/decisions/004_llm_provider.md` justificando la elección

**Funcionalidades:**
- Dado un ticket con score de riesgo y top-3 features SHAP, generar:
  - Resumen ejecutivo en 2 frases
  - Recomendación de acción concreta
  - Nivel de confianza calibrado
- Caché de respuestas para tickets similares (evita costos)
- Fallback a templates fijos si la API LLM falla

**Decisiones técnicas:**
- Provider primario: Claude API (anthropic) por contexto largo y calidad
- Fallback: Hugging Face transformers (modelo small) para offline/CI
- Estructura del prompt: contexto del ticket → factores de riesgo → instrucción de salida

**Tests:**
- El generador funciona con datos mockeados (sin llamar API real en CI)
- El fallback se activa correctamente cuando la API no responde
- Las narrativas no contienen información sensible (PII)

---

### Fase 5 — API y dashboard (días 9-10)

**Objetivo:** exponer el sistema completo como API REST y construir un dashboard que un equipo de operaciones pueda usar todos los días.

**Entregables:**
- `src/itops/api/` con FastAPI app y 4 endpoints:
  - `POST /anomaly`: dado un batch de tickets, identifica anómalos
  - `POST /predict_escalation`: dado un ticket, predice escalación con score
  - `POST /explain`: dado un ticket, devuelve SHAP + narrativa LLM
  - `GET /health`: health check
- `powerbi/IT_Ops_Intelligence.pbix` con dashboard conectado a SQLite
- `src/itops/dashboard/streamlit_app.py` como fallback navegable en web

**Dashboard — 3 vistas por audiencia:**

**Vista Operaciones (gerente del equipo):**
- KPIs del día: total tickets, % escalados, MTTR
- Heatmap de anomalías por categoría y hora
- Top 10 tickets en riesgo activo con score y explicación

**Vista Compliance (SLA y cumplimiento):**
- Tickets en riesgo de SLA
- Tendencia mensual de escalaciones por tier de cliente
- Distribución de tiempo de respuesta

**Vista Estratégica (gerencia alta):**
- KPIs agregados mensuales
- Evolución del % de escalación
- Costo estimado de incidentes evitados

**Tests:**
- Tests de integración del API (mock model, sin DB)
- Tests E2E con modelos reales (skippean en CI)
- Validación de schemas Pydantic

---

### Fase 6 — MLOps, documentación y polish (días 11-12)

**Objetivo:** infraestructura de calidad y documentación que haga el proyecto defendible en entrevista técnica.

**Entregables:**
- MLflow tracking de experimentos en los tres modelos
- GitHub Actions con: lint (ruff), typecheck (mypy), tests (pytest), coverage report
- Dockerfile multi-stage funcional
- README.md completo con:
  - Pregunta de negocio
  - Arquitectura visual
  - Decisiones técnicas y trade-offs
  - Cómo ejecutar localmente
  - Resultados y limitaciones
  - Stack técnico completo
- ADRs (Architecture Decision Records) en `docs/decisions/`
- Coverage objetivo: ≥85%

**Checklist final:**
- [ ] README profesional con badges de CI y coverage
- [ ] Todos los notebooks con narrativa clara, no solo código
- [ ] Tests pasando en CI
- [ ] Docker compose corre localmente sin errores
- [ ] Dashboard Power BI exportado a PDF para portfolio
- [ ] Video corto (opcional) de 2 minutos demostrando el sistema

---

## 6. Decisiones técnicas clave (documentar en ADRs)

### ADR-001: Datos sintéticos vs públicos
**Decisión:** datos sintéticos generados por nosotros.
**Razón:** los datasets públicos de ITSM son escasos y no permiten controlar la distribución para demostrar capacidades específicas (anomalías sembradas, patrones de escalación realistas).

### ADR-002: Autoencoder para anomaly detection
**Decisión:** Isolation Forest como baseline obligatorio, Autoencoder PyTorch como modelo principal.
**Razón:** Isolation Forest es rápido y explicable, pero Autoencoder captura patrones temporales y de co-ocurrencia que IF no ve. Sin baseline, no podemos justificar la complejidad del DL.

### ADR-003: LightGBM sobre XGBoost para escalación
**Decisión:** LightGBM.
**Razón:** entrenamiento más rápido, manejo nativo de categorías, mejor performance con datasets desbalanceados de tamaño medio. SHAP funciona igual en ambos.

### ADR-004: Provider de LLM
**Decisión:** Claude API (anthropic) como primario, Hugging Face como fallback.
**Razón:** Claude tiene mejor calidad para narrativas técnicas y soporta contextos largos. HF como fallback para evitar dependencia única y para CI sin claves de API.

### ADR-005: SQLite vs PostgreSQL
**Decisión:** SQLite para el proyecto base.
**Razón:** cero infraestructura, suficiente para el volumen del demo. Una nota en el README explica cómo escalar a PostgreSQL en producción.

---

## 7. Narrativa de entrevista

Una vez completado el proyecto, tu relato de entrevista cambia completamente:

> "Construí una plataforma de inteligencia operacional para datos ITSM que combina tres modelos: un autoencoder en PyTorch para detección de anomalías, un clasificador LightGBM para predecir escalación de tickets, y una capa LLM que genera explicaciones en lenguaje natural sobre las predicciones para equipos no técnicos.
>
> Está construido end-to-end: dataset sintético realista, pipeline de features, modelos con explicabilidad por SHAP, API REST con FastAPI, dashboard en Power BI por audiencia, y MLOps completo con MLflow, tests automatizados y CI/CD.
>
> Es un sistema, no un notebook. Resuelve un problema real de operaciones de IT: predecir qué tickets van a escalar antes de que lo hagan, explicar por qué, y dar acciones recomendadas al equipo. El código es público en GitHub."

**Preguntas de seguimiento que ya puedes responder con autoridad:**
- "¿Cómo detectas anomalías?" → te paseas por IF vs Autoencoder con criterio
- "¿Cómo manejas clases desbalanceadas?" → costo asimétrico, threshold tuning
- "¿Cómo integras LLM en producción?" → caché, fallback, prompts versionados
- "¿Cómo explicas un modelo a un equipo de operaciones?" → SHAP + LLM narrativo
- "¿Cómo evalúas el modelo?" → CV temporal, métricas correctas, costo de negocio

---

## 8. Quick Start (para Claude Code u otro agente)

Si vas a delegar la implementación a un agente o a Claude Code, este prompt funciona como punto de partida:

```
Estoy construyendo el proyecto "IT Operations Intelligence Platform" siguiendo
el plan en docs/PLAN.md. Quiero empezar por la Fase 1: dataset sintético + EDA.

Por favor:

1. Crea la estructura del repositorio según la sección 4 del plan.
2. Implementa src/itops/data/synthesizer.py con un generador parametrizable
   que produzca el dataset descrito en la Fase 1, incluyendo los patrones
   realistas y las anomalías sembradas.
3. Genera data/raw/tickets_synthetic.csv con 50,000 tickets.
4. Crea notebooks/01_eda.ipynb con análisis exploratorio que valide:
   - Ratio de escalación cercano al 5%
   - Estacionalidad horaria y semanal visible
   - Diferencias por tier de cliente
   - Detección visual de anomalías sembradas
5. Escribe los tests unitarios listados en la Fase 1.
6. Documenta cada decisión técnica con un comentario breve en el código.

Stack: Python 3.11, pandas, numpy, faker, pytest, jupyter.
Sigue las convenciones de calidad: ruff, mypy, tests con coverage.

Empieza por confirmar el plan antes de escribir código.
```

---

## 9. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Datos sintéticos poco realistas | Inyectar patrones reales documentados de literatura ITSM |
| Autoencoder no converge | Empezar simple (3 capas), aumentar complejidad sólo si baseline lo justifica |
| Costos de LLM en testing | Caché agresivo + fallback a HF en CI |
| Scope creep | Cada fase es independiente. Si se atasca una fase, las anteriores siguen siendo defendibles |
| Tiempo total > 12 días | Priorizar fases 1, 3 y 5 (las que cubren más requisitos del JD) |

---

## 10. Métricas de éxito del proyecto

Al final del proyecto, debe ser cierto que:

- [ ] El repositorio público se puede clonar y correr localmente con `docker-compose up`
- [ ] Los tres modelos están entrenados y serializados
- [ ] El API responde a las cuatro rutas definidas
- [ ] El dashboard Power BI muestra datos reales del pipeline
- [ ] Tests pasan en GitHub Actions con coverage ≥85%
- [ ] README documenta el problema, la solución y las decisiones técnicas
- [ ] Puedo explicar cualquier decisión del proyecto en una entrevista sin titubear

---

**Última actualización del plan:** seguir esta versión hasta completar Fase 6. Cualquier cambio significativo debe documentarse como ADR en `docs/decisions/`.
