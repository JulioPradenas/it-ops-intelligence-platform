# Fase 4 — LLM Narratives: Design Spec

**Fecha:** 2026-06-15
**Fase del plan:** 4 de 6
**Estado:** aprobado

---

## Contexto

Con el predictor de escalación (Fase 3) produciendo scores y SHAP values por ticket, el siguiente paso es traducir esas señales en explicaciones en lenguaje natural en español, accesibles para equipos de operaciones no técnicos. La capa LLM recibe un ticket con su score de riesgo y top-3 features SHAP, y devuelve un `Narrative` estructurado con resumen, recomendación y nivel de confianza.

---

## Data model

**`Narrative`** — tipo de retorno de todo el sistema (Pydantic `BaseModel`):

| Campo | Tipo | Descripción |
|---|---|---|
| `summary` | `str` | Resumen ejecutivo en 2 frases en español |
| `recommendation` | `str` | Acción concreta recomendada al equipo de operaciones |
| `confidence` | `float` | 0.0–1.0, derivado del risk_score del modelo de escalación |
| `provider` | `str` | `"claude"` \| `"hf"` — para observabilidad y debugging |

---

## Prompts (`src/itops/llm/prompts.py`)

**Función pública:** `build_escalation_prompt(ticket_context: dict, top_features: list[dict]) -> str`

- `ticket_context` keys: `ticket_id`, `category`, `priority`, `customer_tier`, `risk_score`, `description_snippet` (primeros 200 caracteres de la descripción original)
- `top_features`: lista de dicts `{"feature": str, "shap": float}`, longitud 1–5
- Devuelve un prompt en español que instruye al LLM a responder con JSON `{"summary": ..., "recommendation": ..., "confidence": ...}`
- Template es una f-string — sin lógica dinámica más allá de la interpolación de datos

---

## `NarrativeGenerator` (`src/itops/llm/narrative.py`)

```
NarrativeGenerator
├── __init__(api_key=None, cache_path=PROCESSED_DIR/"narrative_cache.db", hf_model="google/flan-t5-small")
│     Inicializa cliente Anthropic (si api_key disponible), abre/crea SQLite cache.
│     HF pipeline se carga lazy en el primer fallback (no en __init__).
└── generate(ticket_context: dict, top_features: list[dict]) -> Narrative
      1. cache_key = SHA256(repr((ticket_context, top_features)))
      2. Consultar SQLite → si hit, deserializar y devolver
      3. Llamar Claude API (claude-haiku-4-5-20251001)
      4. Si falla (APIError, AuthError, sin api_key) → llamar HF pipeline
      5. Guardar en SQLite → devolver Narrative
```

**Métodos internos:**
- `_cache_key(ticket_context, top_features) -> str` — SHA256 hexdigest
- `_call_claude(prompt: str) -> Narrative` — `anthropic.Anthropic().messages.create(...)`, parsea JSON de la respuesta
- `_call_hf(prompt: str) -> Narrative` — `transformers.pipeline("text2text-generation", model=self._hf_model)`, parsea salida
- `_parse_llm_response(text: str) -> Narrative` — extrae JSON del texto, valida con Pydantic; si falla el parse, `confidence=0.0` y texto genérico

---

## Caché SQLite

Tabla única en `data/processed/narrative_cache.db`:

```sql
CREATE TABLE IF NOT EXISTS narrative_cache (
    key        TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
)
```

- `key`: SHA256 del contexto del ticket
- `data`: JSON serializado del `Narrative`
- `created_at`: ISO timestamp

El path se configura vía `cache_path` en `__init__` — testeable con `:memory:` o `tmp_path`.

---

## Tests (`tests/unit/test_narrative.py`)

| Test | Qué verifica |
|---|---|
| `test_build_prompt_contains_context` | El prompt incluye ticket_id, risk_score y nombres de features |
| `test_narrative_generator_calls_claude` | Con Claude mockeado, `generate()` devuelve `Narrative` con `provider="claude"` |
| `test_fallback_to_hf_on_api_error` | Si Claude lanza `anthropic.APIError`, se activa HF (mockeado) con `provider="hf"` |
| `test_cache_hit_skips_llm` | Segunda llamada con mismos datos no llama Claude ni HF |
| `test_narrative_schema` | `confidence` ∈ [0,1], `summary` y `recommendation` son strings no vacíos |

Todos los tests usan `unittest.mock.patch` — cero llamadas reales a APIs.

---

## Notebook `notebooks/04_llm_explanations.ipynb`

1. Cargar `tickets_synthetic.csv`, entrenar `EscalationModel`, crear `ShapExplainer`
2. Identificar top-10 tickets de mayor riesgo (`predict_proba` más alto)
3. Obtener top-3 SHAP features por ticket via `ShapExplainer.top_features()`
4. Llamar `NarrativeGenerator.generate()` por ticket (requiere `ANTHROPIC_API_KEY` en env)
5. Mostrar tabla: ticket_id | risk_score | summary | recommendation | confidence | provider
6. Mostrar una narrativa completa formateada para ilustrar la salida

---

## ADR `docs/decisions/004_llm_provider.md`

Documenta:
- Claude API (claude-haiku-4-5-20251001) como primario: calidad para narrativas técnicas en español, respuestas estructuradas confiables, costo bajo con haiku
- `google/flan-t5-small` como fallback: offline/CI, ~300MB, sin API key
- SQLite como caché: cero infraestructura extra, persistente entre sesiones, suficiente para el volumen del demo
- Por qué JSON estructurado sobre texto libre: consumible directamente por la API de Fase 5

---

## Archivos a crear/modificar

| Archivo | Estado actual | Acción |
|---|---|---|
| `src/itops/llm/prompts.py` | stub | implementar `build_escalation_prompt` |
| `src/itops/llm/narrative.py` | stub | implementar `Narrative` + `NarrativeGenerator` |
| `tests/unit/test_narrative.py` | no existe | crear |
| `notebooks/04_llm_explanations.ipynb` | no existe | crear y ejecutar |
| `docs/decisions/004_llm_provider.md` | no existe | crear |

**Dependencias nuevas a agregar:**
- `anthropic>=0.30` — SDK oficial de Claude
- `transformers>=4.40` — HF fallback
- `pydantic>=2.5` — ya incluida en FastAPI (Fase 5); añadir como dependencia explícita

**No se tocan:** `synthesizer.py`, `features.py`, `anomaly.py`, `escalation.py`, `explainer.py`, tests existentes (40 pasando).
