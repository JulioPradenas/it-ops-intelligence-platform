# LLM Narratives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar una capa LLM que traduce scores de riesgo y SHAP values en narrativas en español para equipos de operaciones, con caché SQLite y fallback a HF flan-t5-small.

**Architecture:** `build_escalation_prompt()` en `prompts.py` construye el prompt a partir del contexto del ticket y las top features SHAP. `NarrativeGenerator` en `narrative.py` intenta Claude primero, cae al pipeline HF si falla, y cachea resultados en SQLite. El tipo de retorno `Narrative` (Pydantic) tiene campos `summary`, `recommendation`, `confidence`, `provider`.

**Tech Stack:** anthropic SDK, transformers (HF), pydantic v2, sqlite3 (stdlib), pytest + unittest.mock

---

## Archivos a crear/modificar

| Archivo | Acción |
|---|---|
| `pyproject.toml` | Agregar anthropic>=0.30, transformers>=4.40, pydantic>=2.5 |
| `src/itops/llm/prompts.py` | Implementar `build_escalation_prompt` |
| `src/itops/llm/narrative.py` | Implementar `Narrative` + `NarrativeGenerator` |
| `tests/unit/test_narrative.py` | Crear con 5 tests TDD |
| `notebooks/04_llm_explanations.ipynb` | Crear y ejecutar |
| `docs/decisions/004_llm_provider.md` | Crear ADR |
| `.gitignore` | Crear |

---

## Task 1: Agregar dependencias

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Agregar dependencias a `pyproject.toml`**

En el bloque `[project]`, reemplazar la lista `dependencies` por:

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
]
```

- [ ] **Step 2: Instalar y verificar**

```bash
uv pip install -e ".[dev,notebook]"
uv run python -c "import anthropic; import transformers; import pydantic; print(anthropic.__version__, transformers.__version__, pydantic.__version__)"
```

Esperado: tres números de versión, sin errores.

- [ ] **Step 3: Verificar que los 40 tests existentes siguen pasando**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -q
```

Esperado: 40 passed.

---

## Task 2: `build_escalation_prompt` (TDD)

**Files:**
- Create: `tests/unit/test_narrative.py`
- Modify: `src/itops/llm/prompts.py`

- [ ] **Step 1: Crear `tests/unit/test_narrative.py` con el primer test**

```python
"""Tests de la capa LLM — prompts y narrativas (Fase 4)."""

from __future__ import annotations

from itops.llm.prompts import build_escalation_prompt

TICKET_CTX = {
    "ticket_id": "T-001",
    "category": "network",
    "priority": "high",
    "customer_tier": "enterprise",
    "risk_score": 0.87,
    "description_snippet": "Server unreachable since 10am",
}
TOP_FEAT = [
    {"feature": "customer_tier", "shap": 0.42},
    {"feature": "num_reassignments", "shap": 0.31},
    {"feature": "has_critical_keyword", "shap": 0.18},
]


def test_build_prompt_contains_context():
    prompt = build_escalation_prompt(TICKET_CTX, TOP_FEAT)
    assert "T-001" in prompt
    assert "0.87" in prompt
    assert "customer_tier" in prompt
    assert "enterprise" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 100
```

- [ ] **Step 2: Verificar que falla**

```bash
uv run pytest tests/unit/test_narrative.py::test_build_prompt_contains_context -v 2>&1 | head -15
```

Esperado: ImportError (`build_escalation_prompt` no existe).

- [ ] **Step 3: Implementar `src/itops/llm/prompts.py`**

```python
"""Templates de prompts parametrizables para el generador de narrativas LLM."""

from __future__ import annotations


def build_escalation_prompt(ticket_context: dict, top_features: list[dict]) -> str:
    """Construye el prompt para generar una narrativa de escalación en español.

    Args:
        ticket_context: dict con ticket_id, category, priority, customer_tier,
                        risk_score (float) y description_snippet (str, max 200 chars).
        top_features: lista de dicts {"feature": str, "shap": float}, top N features SHAP.

    Returns:
        Prompt listo para enviar al LLM. La respuesta esperada es JSON con campos
        summary, recommendation y confidence.
    """
    features_text = "\n".join(
        f"  - {f['feature']}: impacto SHAP = {f['shap']:.3f}"
        for f in top_features
    )
    return f"""Eres un analista de operaciones IT experto. Analiza el siguiente ticket y genera \
una narrativa de riesgo en español.

TICKET: {ticket_context['ticket_id']}
Categoría: {ticket_context['category']}
Prioridad: {ticket_context['priority']}
Tier del cliente: {ticket_context['customer_tier']}
Puntuación de riesgo de escalación: {ticket_context['risk_score']:.2f} (escala 0-1)
Descripción: {ticket_context['description_snippet']}

Factores principales que contribuyen al riesgo:
{features_text}

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{{
  "summary": "<resumen ejecutivo en 2 frases en español>",
  "recommendation": "<acción concreta recomendada al equipo en español>",
  "confidence": <número entre 0.0 y 1.0 que refleje la certeza de la predicción>
}}"""
```

- [ ] **Step 4: Verificar que el test pasa**

```bash
uv run pytest tests/unit/test_narrative.py::test_build_prompt_contains_context -v
```

Esperado: PASSED.

---

## Task 3: `Narrative` y `NarrativeGenerator` (TDD)

**Files:**
- Modify: `tests/unit/test_narrative.py`
- Modify: `src/itops/llm/narrative.py`

- [ ] **Step 1: Agregar los 4 tests restantes a `tests/unit/test_narrative.py`**

Añadir al final del archivo:

```python
from unittest.mock import MagicMock, patch

import pytest

from itops.llm.narrative import Narrative, NarrativeGenerator

_CLAUDE_JSON = (
    '{"summary": "Riesgo alto detectado en ticket de red.", '
    '"recommendation": "Escalar al equipo senior inmediatamente.", '
    '"confidence": 0.87}'
)
_HF_JSON = (
    '{"summary": "Ticket de riesgo elevado.", '
    '"recommendation": "Revisar con el equipo de soporte.", '
    '"confidence": 0.60}'
)


def test_narrative_schema():
    n = Narrative(summary="Resumen.", recommendation="Acción.", confidence=0.75, provider="claude")
    assert 0.0 <= n.confidence <= 1.0
    assert len(n.summary) > 0
    assert len(n.recommendation) > 0
    assert n.provider in {"claude", "hf"}


def test_narrative_generator_calls_claude(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "claude"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.summary) > 0


def test_fallback_to_hf_on_api_error(tmp_path):
    mock_hf_pipe = MagicMock(return_value=[{"generated_text": _HF_JSON}])

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception(
            "API unavailable"
        )
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")
        gen._hf_pipeline = mock_hf_pipe
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "hf"
    assert 0.0 <= result.confidence <= 1.0


def test_cache_hit_skips_llm(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        create_fn = mock_anthropic.Anthropic.return_value.messages.create
        create_fn.return_value = mock_msg
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")

        # Primera llamada — llama a Claude
        gen.generate(TICKET_CTX, TOP_FEAT)
        # Segunda llamada con los mismos datos — debe salir del caché
        gen.generate(TICKET_CTX, TOP_FEAT)

    assert create_fn.call_count == 1  # Claude solo llamado una vez
```

- [ ] **Step 2: Verificar que los 4 nuevos tests fallan**

```bash
uv run pytest tests/unit/test_narrative.py -v 2>&1 | head -25
```

Esperado: el primer test (prompt) pasa, los 4 nuevos fallan con ImportError.

- [ ] **Step 3: Implementar `src/itops/llm/narrative.py`**

```python
"""Generación de narrativas en lenguaje natural con Claude + fallback HF.

NarrativeGenerator intenta Claude primero, cae al pipeline HF si falla,
y cachea resultados en SQLite para evitar llamadas redundantes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from pydantic import BaseModel, field_validator

from itops.config import PROCESSED_DIR
from itops.llm.prompts import build_escalation_prompt


class Narrative(BaseModel):
    """Narrativa de riesgo generada por LLM para un ticket de escalación."""

    summary: str
    recommendation: str
    confidence: float
    provider: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class NarrativeGenerator:
    """Genera narrativas en español usando Claude con fallback a flan-t5-small."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | str = PROCESSED_DIR / "narrative_cache.db",
        hf_model: str = "google/flan-t5-small",
    ) -> None:
        self._api_key = api_key
        self._cache_path = Path(cache_path)
        self._hf_model = hf_model
        self._hf_pipeline = None  # lazy loaded on first fallback
        self._init_cache()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        if str(self._cache_path) != ":memory:":
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._cache_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS narrative_cache "
            "(key TEXT PRIMARY KEY, data TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def _cache_key(self, ticket_context: dict, top_features: list[dict]) -> str:
        content = repr((ticket_context, top_features)).encode()
        return hashlib.sha256(content).hexdigest()

    def _cache_get(self, key: str) -> Narrative | None:
        row = self._conn.execute(
            "SELECT data FROM narrative_cache WHERE key = ?", (key,)
        ).fetchone()
        return Narrative.model_validate_json(row[0]) if row else None

    def _cache_set(self, key: str, narrative: Narrative) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO narrative_cache (key, data, created_at) VALUES (?, ?, ?)",
            (key, narrative.model_dump_json(), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _parse_llm_response(self, text: str, provider: str) -> Narrative:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        clean = match.group(1).strip() if match else text.strip()
        try:
            data = json.loads(clean)
            return Narrative(
                summary=str(data["summary"]),
                recommendation=str(data["recommendation"]),
                confidence=float(data.get("confidence", 0.5)),
                provider=provider,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return Narrative(
                summary="No se pudo generar un resumen automático.",
                recommendation="Revisar el ticket manualmente con el equipo de soporte.",
                confidence=0.0,
                provider=provider,
            )

    def _call_claude(self, prompt: str) -> Narrative:
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_llm_response(response.content[0].text, provider="claude")

    def _call_hf(self, prompt: str) -> Narrative:
        if self._hf_pipeline is None:
            from transformers import pipeline  # lazy import

            self._hf_pipeline = pipeline(
                "text2text-generation", model=self._hf_model
            )
        result = self._hf_pipeline(prompt, max_new_tokens=200)
        text = result[0]["generated_text"]
        return self._parse_llm_response(text, provider="hf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, ticket_context: dict, top_features: list[dict]) -> Narrative:
        """Genera o recupera del caché una narrativa para el ticket dado."""
        key = self._cache_key(ticket_context, top_features)
        cached = self._cache_get(key)
        if cached:
            return cached

        prompt = build_escalation_prompt(ticket_context, top_features)
        try:
            narrative = self._call_claude(prompt)
        except Exception:
            narrative = self._call_hf(prompt)

        self._cache_set(key, narrative)
        return narrative
```

- [ ] **Step 4: Verificar que todos los tests de narrative pasan**

```bash
uv run pytest tests/unit/test_narrative.py -v
```

Esperado: 5 PASSED.

- [ ] **Step 5: Suite completa**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest -q
```

Esperado: 45 passed.

- [ ] **Step 6: Ruff**

```bash
uv run ruff check src/itops/llm/ tests/unit/test_narrative.py
```

Corregir cualquier issue.

---

## Task 4: Notebook `04_llm_explanations.ipynb`

**Files:**
- Create: `scripts/_build_llm_notebook.py`
- Create: `notebooks/04_llm_explanations.ipynb`

- [ ] **Step 1: Crear el script de construcción**

Crear `scripts/_build_llm_notebook.py`:

```python
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from itops.config import PROJECT_ROOT

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md(
    "# 04 — Explicaciones LLM para predicciones de escalación\n\n"
    "Dado un ticket con score de riesgo y factores SHAP, genera narrativas "
    "en español usando Claude (con fallback a flan-t5-small)."
)

code(
    "import os\n"
    "import pandas as pd\n\n"
    "from itops.config import RAW_TICKETS_CSV\n"
    "from itops.data.loader import load_tickets\n"
    "from itops.models.escalation import EscalationModel\n"
    "from itops.models.explainer import ShapExplainer\n"
    "from itops.llm.narrative import NarrativeGenerator\n\n"
    "df = load_tickets(RAW_TICKETS_CSV)\n"
    "print(f'Tickets cargados: {len(df):,}')"
)

md("## 1. Entrenar modelo y extraer top-10 tickets de mayor riesgo")
code(
    "model = EscalationModel(seed=42)\n"
    "model.fit(df)\n"
    "proba = model.predict_proba(df)\n\n"
    "top10_idx = proba.argsort()[-10:][::-1]\n"
    "df_top10 = df.iloc[top10_idx].copy()\n"
    "df_top10['risk_score'] = proba[top10_idx]\n"
    "print(f'AUC-ROC: {model.eval_metrics_[\"auc_roc\"]:.3f}')\n"
    "df_top10[['ticket_id', 'category', 'customer_tier', 'risk_score']].head(10)"
)

md("## 2. SHAP features por ticket")
code(
    "explainer = ShapExplainer(model)\n"
    "top_shap = explainer.top_features(df_top10, n=3)\n"
    "top_shap.head()"
)

md("## 3. Generar narrativas LLM")
code(
    "api_key = os.getenv('ANTHROPIC_API_KEY')  # None → usa env var automáticamente\n"
    "gen = NarrativeGenerator(api_key=api_key)\n\n"
    "narratives = []\n"
    "for i, (_, ticket) in enumerate(df_top10.iterrows()):\n"
    "    shap_row = top_shap.iloc[i]\n"
    "    top_features = [\n"
    "        {'feature': shap_row[f'feature_{j}'], 'shap': shap_row[f'shap_{j}']}\n"
    "        for j in range(1, 4)\n"
    "    ]\n"
    "    ticket_ctx = {\n"
    "        'ticket_id': ticket['ticket_id'],\n"
    "        'category': ticket['category'],\n"
    "        'priority': ticket['priority_initial'],\n"
    "        'customer_tier': ticket['customer_tier'],\n"
    "        'risk_score': float(ticket['risk_score']),\n"
    "        'description_snippet': str(ticket['description'])[:200],\n"
    "    }\n"
    "    narrative = gen.generate(ticket_ctx, top_features)\n"
    "    narratives.append({\n"
    "        'ticket_id': ticket['ticket_id'],\n"
    "        'risk_score': round(float(ticket['risk_score']), 3),\n"
    "        'provider': narrative.provider,\n"
    "        'confidence': round(narrative.confidence, 2),\n"
    "        'summary': narrative.summary,\n"
    "        'recommendation': narrative.recommendation,\n"
    "    })\n\n"
    "df_narratives = pd.DataFrame(narratives)\n"
    "df_narratives[['ticket_id', 'risk_score', 'provider', 'confidence']]"
)

md("## 4. Tabla completa")
code("df_narratives[['ticket_id', 'summary', 'recommendation']].to_string(index=False)")

md("## 5. Narrativa completa del ticket de mayor riesgo")
code(
    "best = df_narratives.iloc[0]\n"
    "print(f'Ticket: {best.ticket_id}')\n"
    "print(f'Risk score: {best.risk_score}')\n"
    "print(f'Provider: {best.provider} (confidence: {best.confidence})')\n"
    "print()\n"
    "print('RESUMEN:')\n"
    "print(best.summary)\n"
    "print()\n"
    "print('RECOMENDACIÓN:')\n"
    "print(best.recommendation)"
)

md(
    "## Conclusiones\n\n"
    "- `NarrativeGenerator` produce explicaciones en español a partir del score de riesgo "
    "y los factores SHAP del ticket.\n"
    "- El campo `provider` indica si la narrativa vino de Claude o del fallback HF.\n"
    "- El caché SQLite evita llamadas redundantes a la API para tickets similares.\n"
    "- Las narrativas son consumibles directamente por la API de Fase 5 "
    "en el endpoint `POST /explain`."
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

out = PROJECT_ROOT / "notebooks" / "04_llm_explanations.ipynb"
nbformat.write(nb, out)
print("escrito", out)
```

- [ ] **Step 2: Ejecutar el script**

```bash
uv run python scripts/_build_llm_notebook.py
```

Esperado: `escrito .../notebooks/04_llm_explanations.ipynb`

- [ ] **Step 3: Ejecutar el notebook**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/04_llm_explanations.ipynb 2>&1 | tail -5
```

Esperado: sin errores. Si `ANTHROPIC_API_KEY` está en el entorno, las narrativas vendrán de Claude. Si no, el fallback HF generará texto (puede tardar ~2 min en descargar el modelo la primera vez).

- [ ] **Step 4: Borrar el script auxiliar**

```bash
rm scripts/_build_llm_notebook.py
```

- [ ] **Step 5: Verificar que el notebook tiene outputs**

```bash
uv run python -c "
import json, pathlib
nb = json.loads(pathlib.Path('notebooks/04_llm_explanations.ipynb').read_text())
n = sum(1 for c in nb['cells'] if c.get('outputs'))
print(f'Celdas con output: {n} / {len(nb[\"cells\"])}')
"
```

Esperado: ≥ 4 celdas con output.

---

## Task 5: ADR + verificación final

**Files:**
- Create: `docs/decisions/004_llm_provider.md`

- [ ] **Step 1: Crear el ADR**

Crear `docs/decisions/004_llm_provider.md`:

```markdown
# ADR-004: Proveedor LLM para narrativas de escalación

**Fecha:** 2026-06-15
**Estado:** aceptado

## Contexto

Necesitamos generar narrativas en español para tickets de alto riesgo de escalación.
Los consumidores son equipos de operaciones no técnicos — la calidad y naturalidad
del texto es importante. Las narrativas deben tener estructura fija (resumen,
recomendación, confianza) para ser consumibles por la API de Fase 5.

## Decisión

**Primario:** Claude API (`claude-haiku-4-5-20251001`)
**Fallback:** `google/flan-t5-small` via `transformers.pipeline`
**Caché:** SQLite en `data/processed/narrative_cache.db`

## Justificaciones

**Claude haiku como primario:** Genera texto en español de alta calidad, respeta
instrucciones de formato JSON de forma confiable, y con haiku el costo por narrativa
es <$0.001. El contexto largo de Claude permite incluir el prompt completo del ticket
sin truncar.

**flan-t5-small como fallback:** Modelo seq2seq de 300MB que funciona completamente
offline. Permite que el notebook y CI ejecuten sin API key. La calidad es inferior
a Claude pero suficiente para demostrar el flujo completo.

**SQLite como caché:** Cero infraestructura extra, persiste entre sesiones del notebook
y la API (Fase 5), y la clave SHA256 garantiza determinismo. Para el volumen del demo
(~50k tickets con 5% escalados = ~2.500 tickets a narrar) SQLite es más que suficiente.

**JSON estructurado sobre texto libre:** El tipo `Narrative` con `summary`,
`recommendation` y `confidence` es consumible directamente por el endpoint
`POST /explain` de Fase 5 sin parsing adicional.

## Trade-offs

- `flan-t5-small` puede generar JSON inválido o texto parcialmente incoherente en
  español. El método `_parse_llm_response` tiene fallback a texto genérico.
- La primera ejecución del fallback HF descarga ~300MB. Las siguientes usan caché
  del modelo en `~/.cache/huggingface/`.
- Si se quiere escalar: reemplazar flan-t5-small por un modelo más grande (Phi-3-mini,
  Mistral-7B) o usar la API de OpenAI/Anthropic Claude Opus como alternativa de mayor
  calidad.

## Consecuencias

- `ANTHROPIC_API_KEY` en entorno → narrativas de Claude. Sin la key → fallback HF.
- La API de Fase 5 (`POST /explain`) recibirá un `Narrative` serializado.
- Los tests nunca llaman APIs reales — usan `unittest.mock.patch`.
```

- [ ] **Step 2: Verificar suite completa + calidad**

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uv run pytest --cov=itops --cov-report=term-missing -q
```

```bash
uv run ruff check .
```

```bash
uv run mypy 2>&1 | tail -5
```

Esperado: 45 passed, ruff clean, mypy clean. Reportar coverage de `narrative.py` y `prompts.py`.

---

## Task 6: Git init y commit inicial

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Crear `.gitignore`**

Crear `.gitignore` en la raíz del proyecto:

```gitignore
# Entorno virtual
.venv/
venv/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Artefactos de build
dist/
*.egg-info/
build/

# Herramientas
.mypy_cache/
.pytest_cache/
.ruff_cache/

# Datos procesados y caché (generados, no versionados)
data/processed/
*.db

# Credenciales y configuración local
.env
.env.local

# macOS
.DS_Store

# Jupyter checkpoints
.ipynb_checkpoints/

# Scripts temporales de build (se crean y borran durante implementación)
scripts/_build_*.py
```

- [ ] **Step 2: Inicializar el repositorio git**

```bash
git init
git config user.email "pradnas@gmail.com"
git config user.name "Julio"
```

Esperado: `Initialized empty Git repository in .../IT Operations Intelligence Platform/.git/`

- [ ] **Step 3: Commit inicial — Fase 1 (dataset sintético + EDA)**

```bash
git add pyproject.toml src/itops/__init__.py src/itops/config.py
git add src/itops/data/synthesizer.py src/itops/data/loader.py
git add tests/conftest.py tests/unit/test_synthesizer.py
git add data/raw/tickets_synthetic.csv data/raw/seeded_anomalies.json
git add notebooks/01_eda.ipynb docs/decisions/001_synthetic_data.md
git add docs/superpowers/ .gitignore
git commit -m "feat: fase 1 — dataset sintético ITSM 50k tickets + EDA

Generador parametrizable (SynthConfig), patrones realistas (estacionalidad,
tier bias, reassignment effect), 6 anomalías sembradas detectables.
Notebook EDA con análisis exploratorio completo."
```

- [ ] **Step 4: Commit Fase 2 (anomaly detection)**

```bash
git add src/itops/data/features.py src/itops/models/anomaly.py
git add tests/unit/test_anomaly.py
git add notebooks/02_anomaly_detection.ipynb docs/decisions/002_autoencoder_for_anomaly.md
git commit -m "feat: fase 2 — anomaly detection (IsolationForest + Autoencoder PyTorch)

build_hourly_features agrega tickets por (fecha, hora, categoría).
IsolationForestDetector baseline + AutoencoderDetector MLP 5-capas.
Anomalías sembradas detectadas al percentil 97."
```

- [ ] **Step 5: Commit Fase 3 (escalation predictor)**

```bash
git add src/itops/models/escalation.py src/itops/models/explainer.py
git add tests/unit/test_escalation.py
git add notebooks/03_escalation_model.ipynb docs/decisions/003_lightgbm_over_xgboost.md
git commit -m "feat: fase 3 — predictor de escalación LightGBM + SHAP

EscalationModel con split temporal 80/20, threshold óptimo por costo
asimétrico FN:FP=5:1. ShapExplainer con TreeExplainer para explicaciones
por ticket. AUC-ROC > 0.90 en holdout."
```

- [ ] **Step 6: Commit Fase 4 (LLM narratives)**

```bash
git add src/itops/llm/prompts.py src/itops/llm/narrative.py src/itops/llm/__init__.py
git add tests/unit/test_narrative.py
git add notebooks/04_llm_explanations.ipynb docs/decisions/004_llm_provider.md
git commit -m "feat: fase 4 — capa LLM para narrativas en español

NarrativeGenerator: Claude haiku primario + flan-t5-small fallback + caché
SQLite. Salida estructurada Pydantic (summary, recommendation, confidence).
5 tests unitarios con mocks — cero llamadas reales a APIs."
```

- [ ] **Step 7: Verificar estado del repo**

```bash
git log --oneline
git status
```

Esperado: 4 commits, working tree clean.

---

## Self-Review

**Cobertura del spec:**
- ✅ `build_escalation_prompt` con ticket_context + top_features → Task 2
- ✅ `Narrative` Pydantic (summary, recommendation, confidence, provider) → Task 3
- ✅ `NarrativeGenerator` con Claude + HF fallback + SQLite cache → Task 3
- ✅ Lazy load del HF pipeline → Task 3 (`_hf_pipeline = None`)
- ✅ `_parse_llm_response` con fallback genérico → Task 3
- ✅ 5 tests con mocks, cero API reales → Tasks 2-3
- ✅ Notebook con top-10 tickets, SHAP, narrativas, tabla → Task 4
- ✅ ADR-004 con Claude/HF/SQLite/JSON → Task 5
- ✅ Git init + 4 commits por fase → Task 6

**Consistencia de tipos:**
- `Narrative` definida en Task 3, importada en test desde `itops.llm.narrative` ✓
- `build_escalation_prompt(ticket_context: dict, top_features: list[dict]) -> str` consistente en Tasks 2-3-4 ✓
- `NarrativeGenerator.generate()` siempre devuelve `Narrative` ✓
- `cache_path` acepta `Path | str` en todos los usos ✓

**Sin placeholders:** todo el código está completo en cada step.
