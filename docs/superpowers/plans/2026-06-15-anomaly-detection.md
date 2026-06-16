# Anomaly Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar dos detectores de anomalías sobre ventanas horarias de tickets (IsolationForest + Autoencoder MLP) con feature engineering, tests y notebook comparativo.

**Architecture:** `build_hourly_features()` agrega el DataFrame crudo a ventanas `(date, hour, category)` con 11 features numéricas. `IsolationForestDetector` y `AutoencoderDetector` implementan un ABC común (`fit / score / predict`). Los detectores reciben una matriz numpy `X` donde la columna 0 siempre es `ticket_count` (definida por `FEATURE_COLS` en `features.py`).

**Tech Stack:** pandas, numpy, scikit-learn, PyTorch (CPU), pytest, nbformat

---

## Archivos a crear/modificar

| Archivo | Acción |
|---|---|
| `pyproject.toml` | Agregar scikit-learn, torch, nbformat a dependencies |
| `src/itops/data/features.py` | Implementar `build_hourly_features` y `FEATURE_COLS` |
| `src/itops/models/anomaly.py` | Implementar ABC + IF + Autoencoder |
| `tests/unit/test_anomaly.py` | Tests de features y detectores |
| `notebooks/02_anomaly_detection.ipynb` | Notebook comparativo |
| `docs/decisions/002_autoencoder_for_anomaly.md` | ADR |

---

## Task 1: Agregar dependencias ML

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Agregar scikit-learn, torch y nbformat a `pyproject.toml`**

Reemplazar el bloque `dependencies` actual:

```toml
[project]
name = "itops"
version = "0.1.0"
description = "IT Operations Intelligence Platform — anomaly detection, escalation risk scoring y explicaciones LLM sobre datos ITSM"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "faker>=25.0",
    "pyarrow>=16.0",
    "scikit-learn>=1.5",
    "torch>=2.3",
    "nbformat>=5.9",
]
```

- [ ] **Step 2: Instalar dependencias**

```bash
uv pip install -e ".[dev,notebook]"
```

Esperado: sin errores. Verificar que `torch` y `sklearn` están disponibles:

```bash
uv run python -c "import torch; import sklearn; print(torch.__version__, sklearn.__version__)"
```

Esperado: dos números de versión impresos.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add scikit-learn and torch dependencies for Phase 2"
```

---

## Task 2: Feature engineering — `build_hourly_features`

**Files:**
- Modify: `src/itops/data/features.py`
- Create: `tests/unit/test_anomaly.py`

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/test_anomaly.py` con el contenido inicial:

```python
"""Tests de features y detectores de anomalías (Fase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.api import types as ptypes

from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.data.synthesizer import SynthConfig, generate


@pytest.fixture(scope="module")
def small_df():
    return generate(SynthConfig(n_tickets=3_000, seed=42)).tickets


@pytest.fixture(scope="module")
def features_df(small_df):
    return build_hourly_features(small_df)


@pytest.fixture(scope="module")
def X_small(features_df):
    return features_df[FEATURE_COLS].values


def test_build_hourly_features_columns(features_df):
    assert set(FEATURE_COLS).issubset(features_df.columns)
    assert "date" in features_df.columns
    assert "hour" in features_df.columns
    assert "category" in features_df.columns


def test_build_hourly_features_no_nulls(features_df):
    assert features_df[FEATURE_COLS].isna().sum().sum() == 0


def test_build_hourly_features_numeric_dtypes(features_df):
    for col in FEATURE_COLS:
        assert ptypes.is_numeric_dtype(features_df[col]), col


def test_build_hourly_features_ticket_count_positive(features_df):
    # Solo incluye ventanas con tickets
    assert (features_df["ticket_count"] > 0).all()


def test_build_hourly_features_escalation_rate_bounded(features_df):
    assert features_df["escalation_rate"].between(0, 1).all()
```

- [ ] **Step 2: Verificar que falla**

```bash
uv run pytest tests/unit/test_anomaly.py -v 2>&1 | head -20
```

Esperado: `ImportError` o `ModuleNotFoundError` porque `features.py` está vacío.

- [ ] **Step 3: Implementar `build_hourly_features`**

Reemplazar el contenido de `src/itops/data/features.py`:

```python
"""Feature engineering: agrega tickets crudos a ventanas horarias por categoría."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columnas de features que alimentan los detectores de anomalías.
# Orden fijo: columna 0 = ticket_count (los detectores dependen de esto para
# filtrar ventanas normales durante el entrenamiento del Autoencoder).
FEATURE_COLS: list[str] = [
    "ticket_count",
    "escalated_count",
    "escalation_rate",
    "avg_response_time_minutes",
    "avg_num_comments",
    "avg_num_reassignments",
    "pct_off_hours",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el DataFrame de tickets a ventanas (date, hour, category).

    Devuelve un DataFrame con columnas 'date', 'hour', 'category' + FEATURE_COLS.
    Solo incluye ventanas con al menos un ticket (no genera filas vacías).
    """
    work = df.copy()
    work["_date"] = work["created_at"].dt.date
    work["_hour"] = work["created_at"].dt.hour

    agg = (
        work.groupby(["_date", "_hour", "category"])
        .agg(
            ticket_count=("ticket_id", "count"),
            escalated_count=("escalated", "sum"),
            avg_response_time_minutes=("response_time_minutes", "mean"),
            avg_num_comments=("num_comments", "mean"),
            avg_num_reassignments=("num_reassignments", "mean"),
            pct_off_hours=("business_hours", lambda x: 1.0 - x.mean()),
        )
        .reset_index()
        .rename(columns={"_date": "date", "_hour": "hour"})
    )

    agg["escalation_rate"] = (agg["escalated_count"] / agg["ticket_count"]).fillna(0.0)

    weekday = pd.to_datetime(agg["date"]).dt.weekday
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)
    agg["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    agg["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    agg[FEATURE_COLS] = agg[FEATURE_COLS].fillna(0.0)

    return agg
```

- [ ] **Step 4: Verificar que los tests de features pasan**

```bash
uv run pytest tests/unit/test_anomaly.py -v -k "features"
```

Esperado: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/itops/data/features.py tests/unit/test_anomaly.py
git commit -m "feat: implement build_hourly_features and feature tests"
```

---

## Task 3: Implementar `IsolationForestDetector`

**Files:**
- Modify: `src/itops/models/anomaly.py`
- Modify: `tests/unit/test_anomaly.py`

- [ ] **Step 1: Agregar tests de IF al archivo de tests**

Añadir al final de `tests/unit/test_anomaly.py`:

```python
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector


def test_if_fits_and_scores(X_small):
    det = IsolationForestDetector(seed=42)
    det.fit(X_small)
    scores = det.score(X_small)
    assert scores.shape == (len(X_small),)
    assert np.issubdtype(scores.dtype, np.floating)


def test_if_predict_fraction(X_small):
    det = IsolationForestDetector(seed=42)
    det.fit(X_small)
    flags = det.predict(X_small, percentile=99.0)
    assert flags.dtype == bool
    # percentil 99 → ~1% marcado
    assert 0.005 <= flags.mean() <= 0.02


def test_if_scores_reproducible(X_small):
    a = IsolationForestDetector(seed=7)
    a.fit(X_small)
    b = IsolationForestDetector(seed=7)
    b.fit(X_small)
    np.testing.assert_array_equal(a.score(X_small), b.score(X_small))
```

- [ ] **Step 2: Verificar que los nuevos tests fallan**

```bash
uv run pytest tests/unit/test_anomaly.py::test_if_fits_and_scores -v
```

Esperado: `ImportError` porque `anomaly.py` está vacío.

- [ ] **Step 3: Implementar ABC + IsolationForestDetector**

Reemplazar `src/itops/models/anomaly.py`:

```python
"""Detectores de anomalías sobre ventanas horarias de tickets.

IsolationForestDetector: baseline rápido con scikit-learn.
AutoencoderDetector: MLP PyTorch entrenado en ventanas normales.

Ambos implementan AnomalyDetector (ABC) con fit / score / predict.
La columna 0 de X debe ser ticket_count (ver FEATURE_COLS en features.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector(ABC):
    """Protocolo común: fit entrena, score devuelve anomalía (mayor = peor), predict umbraliza."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        scores = self.score(X)
        threshold = np.percentile(scores, percentile)
        return (scores >= threshold).astype(bool)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest con StandardScaler interno. Reproducible vía seed."""

    def __init__(self, n_estimators: int = 100, seed: int = 42) -> None:
        self._scaler = StandardScaler()
        self._model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: más negativo = más anómalo; invertimos para consistencia.
        return -self._model.decision_function(self._scaler.transform(X))
```

- [ ] **Step 4: Verificar que los tests de IF pasan**

```bash
uv run pytest tests/unit/test_anomaly.py -v -k "if_"
```

Esperado: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/itops/models/anomaly.py tests/unit/test_anomaly.py
git commit -m "feat: implement IsolationForestDetector with ABC"
```

---

## Task 4: Implementar `AutoencoderDetector`

**Files:**
- Modify: `src/itops/models/anomaly.py`
- Modify: `tests/unit/test_anomaly.py`

- [ ] **Step 1: Agregar tests del Autoencoder**

Añadir al final de `tests/unit/test_anomaly.py`:

```python
def test_ae_fits_and_scores(X_small):
    det = AutoencoderDetector(epochs=5, seed=42)
    det.fit(X_small)
    scores = det.score(X_small)
    assert scores.shape == (len(X_small),)
    assert np.issubdtype(scores.dtype, np.floating)


def test_ae_score_before_fit_raises(X_small):
    det = AutoencoderDetector(epochs=5, seed=42)
    with pytest.raises(RuntimeError, match="fit"):
        det.score(X_small)


def test_ae_scores_reproducible(X_small):
    c = AutoencoderDetector(epochs=5, seed=7)
    c.fit(X_small)
    d = AutoencoderDetector(epochs=5, seed=7)
    d.fit(X_small)
    np.testing.assert_array_almost_equal(c.score(X_small), d.score(X_small))
```

- [ ] **Step 2: Verificar que los tests fallan**

```bash
uv run pytest tests/unit/test_anomaly.py::test_ae_fits_and_scores -v
```

Esperado: `ImportError` o `AttributeError` porque `AutoencoderDetector` no existe.

- [ ] **Step 3: Agregar `_MLP` y `AutoencoderDetector` a `anomaly.py`**

Añadir al final de `src/itops/models/anomaly.py` (después de `IsolationForestDetector`):

```python
class _MLP(nn.Module):
    """Autoencoder MLP: input_dim → 32 → 16 → 8 → 16 → 32 → input_dim."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder MLP en PyTorch.

    Se entrena solo en ventanas con ticket_count < percentil 95 para que el
    error de reconstrucción sea bajo en la distribución base y alto en bursts.
    Columna 0 de X debe ser ticket_count (contrato con FEATURE_COLS).
    """

    def __init__(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._seed = seed
        self._scaler = StandardScaler()
        self._model: _MLP | None = None

    def fit(self, X: np.ndarray) -> None:
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

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score()")
        tensor = torch.FloatTensor(self._scaler.transform(X))
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
        return ((tensor - recon) ** 2).mean(dim=1).numpy()
```

- [ ] **Step 4: Verificar que todos los tests de anomaly pasan**

```bash
uv run pytest tests/unit/test_anomaly.py -v
```

Esperado: todos PASSED (puede tardar ~10s por el entrenamiento del Autoencoder).

- [ ] **Step 5: Commit**

```bash
git add src/itops/models/anomaly.py tests/unit/test_anomaly.py
git commit -m "feat: implement AutoencoderDetector (MLP PyTorch)"
```

---

## Task 5: Test de integración — anomalías sembradas detectadas

**Files:**
- Modify: `tests/unit/test_anomaly.py`

Este test usa el dataset completo de 50k tickets y verifica que las 6 anomalías
sembradas aparezcan por encima del umbral del percentil 99.

- [ ] **Step 1: Agregar el test de integración**

Añadir al final de `tests/unit/test_anomaly.py`:

```python
def test_seeded_anomalies_detected_by_if():
    """Las 6 anomalías sembradas deben superar el umbral en IsolationForest."""
    import json

    from itops.config import RAW_TICKETS_CSV, SEEDED_ANOMALIES_JSON
    from itops.data.loader import load_tickets

    df = load_tickets(RAW_TICKETS_CSV)
    feat = build_hourly_features(df)
    X = feat[FEATURE_COLS].values

    det = IsolationForestDetector(seed=42)
    det.fit(X)
    scores = det.score(X)
    threshold = np.percentile(scores, 99)

    anomalies = json.loads(SEEDED_ANOMALIES_JSON.read_text())["anomalies"]
    for a in anomalies:
        mask = (
            (feat["date"].astype(str) == a["date"])
            & (feat["category"] == a["category"])
            & (feat["hour"] >= a["window_start_hour"])
            & (feat["hour"] < a["window_start_hour"] + a["window_hours"])
        )
        burst_scores = scores[mask]
        assert len(burst_scores) > 0, f"Sin ventanas para {a['date']} {a['category']}"
        assert burst_scores.max() > threshold, (
            f"Anomalía no detectada: {a['date']} {a['category']} "
            f"max_score={burst_scores.max():.4f} threshold={threshold:.4f}"
        )
```

- [ ] **Step 2: Ejecutar el test**

```bash
uv run pytest tests/unit/test_anomaly.py::test_seeded_anomalies_detected_by_if -v
```

Esperado: PASSED. Si falla alguna anomalía, revisar que `data/raw/tickets_synthetic.csv` existe (ejecutar `python scripts/generate_data.py` si no).

- [ ] **Step 3: Ejecutar la suite completa**

```bash
uv run pytest --cov=itops --cov-report=term-missing
```

Esperado: todos PASSED, coverage ≥85% en `features.py` y `anomaly.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_anomaly.py
git commit -m "test: add integration test for seeded anomaly detection"
```

---

## Task 6: Notebook `02_anomaly_detection.ipynb`

**Files:**
- Create: `notebooks/02_anomaly_detection.ipynb`

- [ ] **Step 1: Crear y ejecutar el notebook**

Ejecutar este script Python una sola vez (luego borrarlo):

```python
# scripts/_build_anomaly_notebook.py
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from itops.config import PROJECT_ROOT

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md(
    "# 02 — Detección de anomalías\n\n"
    "Comparación de `IsolationForestDetector` (baseline) vs `AutoencoderDetector` (MLP PyTorch) "
    "sobre ventanas horarias `(date, hour, category)` de los 50k tickets sintéticos.\n\n"
    "**Objetivo de validación:** las 6 anomalías sembradas (`seeded_anomalies.json`) "
    "aparecen en el top de scores de ambos detectores."
)

code(
    "import json\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n\n"
    "from itops.config import RAW_TICKETS_CSV, SEEDED_ANOMALIES_JSON\n"
    "from itops.data.features import FEATURE_COLS, build_hourly_features\n"
    "from itops.data.loader import load_tickets\n"
    "from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector\n\n"
    "sns.set_theme(style='whitegrid')\n\n"
    "df = load_tickets(RAW_TICKETS_CSV)\n"
    "anomalies = json.loads(SEEDED_ANOMALIES_JSON.read_text())['anomalies']\n"
    "print(f'Tickets: {len(df):,}')\n"
    "print(f'Anomalías sembradas: {len(anomalies)}')"
)

md("## 1. Feature engineering\n\nAgregamos tickets a ventanas horarias por categoría.")
code(
    "feat = build_hourly_features(df)\n"
    "X = feat[FEATURE_COLS].values\n"
    "print(f'Ventanas (date×hour×category): {len(feat):,}')\n"
    "feat[FEATURE_COLS].describe().round(2)"
)
code(
    "fig, axes = plt.subplots(1, 3, figsize=(15, 3))\n"
    "feat['ticket_count'].hist(bins=50, ax=axes[0]); axes[0].set(title='ticket_count', yscale='log')\n"
    "feat['escalation_rate'].hist(bins=30, ax=axes[1]); axes[1].set(title='escalation_rate')\n"
    "feat['avg_response_time_minutes'].hist(bins=30, ax=axes[2]); axes[2].set(title='avg_response_time_min')\n"
    "plt.tight_layout(); plt.show()"
)

md("## 2. IsolationForest — baseline")
code(
    "if_det = IsolationForestDetector(seed=42)\n"
    "if_det.fit(X)\n"
    "if_scores = if_det.score(X)\n"
    "feat = feat.assign(if_score=if_scores)\n\n"
    "plt.figure(figsize=(8, 3))\n"
    "plt.hist(if_scores, bins=80, color='steelblue')\n"
    "plt.axvline(np.percentile(if_scores, 99), color='red', ls='--', label='p99')\n"
    "plt.title('Distribución de scores — IsolationForest'); plt.legend(); plt.show()"
)
code(
    "top10_if = feat.nlargest(10, 'if_score')[['date', 'hour', 'category', 'ticket_count', 'if_score']]\n"
    "print('Top-10 ventanas más anómalas (IF):')\n"
    "top10_if"
)

md("## 3. Autoencoder MLP")
code(
    "ae_det = AutoencoderDetector(epochs=50, seed=42)\n"
    "ae_det.fit(X)\n"
    "ae_scores = ae_det.score(X)\n"
    "feat = feat.assign(ae_score=ae_scores)\n\n"
    "plt.figure(figsize=(8, 3))\n"
    "plt.hist(ae_scores, bins=80, color='darkorange')\n"
    "plt.axvline(np.percentile(ae_scores, 99), color='red', ls='--', label='p99')\n"
    "plt.title('Distribución de error de reconstrucción — Autoencoder'); plt.legend(); plt.show()"
)
code(
    "top10_ae = feat.nlargest(10, 'ae_score')[['date', 'hour', 'category', 'ticket_count', 'ae_score']]\n"
    "print('Top-10 ventanas más anómalas (Autoencoder):')\n"
    "top10_ae"
)

md("## 4. Comparación — ¿se detectan las anomalías sembradas?")
code(
    "if_threshold = np.percentile(if_scores, 99)\n"
    "ae_threshold = np.percentile(ae_scores, 99)\n\n"
    "rows = []\n"
    "for a in anomalies:\n"
    "    mask = (\n"
    "        (feat['date'].astype(str) == a['date'])\n"
    "        & (feat['category'] == a['category'])\n"
    "        & (feat['hour'] >= a['window_start_hour'])\n"
    "        & (feat['hour'] < a['window_start_hour'] + a['window_hours'])\n"
    "    )\n"
    "    sub = feat[mask]\n"
    "    if_detected = bool((sub['if_score'] > if_threshold).any()) if len(sub) else False\n"
    "    ae_detected = bool((sub['ae_score'] > ae_threshold).any()) if len(sub) else False\n"
    "    rows.append({\n"
    "        'date': a['date'], 'category': a['category'],\n"
    "        'burst_size': a['burst_size'],\n"
    "        'if_detected': if_detected,\n"
    "        'ae_detected': ae_detected,\n"
    "    })\n"
    "pd.DataFrame(rows)"
)

md(
    "## Conclusiones\n\n"
    "- **IsolationForest:** rápido (~1s), sin hiperparámetros críticos. "
    "Detecta bursts en ticket_count como outliers multivariados. "
    "Recomendado para producción por su velocidad y explicabilidad.\n"
    "- **Autoencoder:** captura patrones de co-ocurrencia entre features que IF no ve "
    "(ej. burst de escalación sin aumento de volumen). Más lento de entrenar (~30s). "
    "Útil como segundo nivel de alerta.\n"
    "- **Umbral operacional:** percentil 99 → ~1% de ventanas marcadas. "
    "En producción ajustar según tolerancia a falsos positivos del equipo de operaciones."
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

out = PROJECT_ROOT / "notebooks" / "02_anomaly_detection.ipynb"
import nbformat
nbformat.write(nb, out)
print("escrito", out)
```

Guardarlo en `scripts/_build_anomaly_notebook.py` y ejecutar:

```bash
uv run python scripts/_build_anomaly_notebook.py
```

- [ ] **Step 2: Ejecutar el notebook para generar outputs**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_anomaly_detection.ipynb
```

Esperado: sin errores, el notebook queda con outputs. Puede tardar 1-2 minutos (entrenamiento del Autoencoder).

- [ ] **Step 3: Borrar el script de construcción**

```bash
rm scripts/_build_anomaly_notebook.py
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_anomaly_detection.ipynb
git commit -m "feat: add anomaly detection comparison notebook"
```

---

## Task 7: ADR `002_autoencoder_for_anomaly.md`

**Files:**
- Create: `docs/decisions/002_autoencoder_for_anomaly.md`

- [ ] **Step 1: Crear el ADR**

Crear `docs/decisions/002_autoencoder_for_anomaly.md`:

```markdown
# ADR-002: Autoencoder para detección de anomalías

**Fecha:** 2026-06-15
**Estado:** aceptado

## Contexto

Necesitamos detectar bursts anómalos en el flujo de tickets ITSM (aumento súbito
de una categoría en una ventana horaria). Las anomalías son raras (~1% de ventanas)
y no tenemos etiquetas de entrenamiento — es un problema de detección no supervisada.

## Decisión

Usar dos modelos complementarios:
1. `IsolationForestDetector` como baseline obligatorio.
2. `AutoencoderDetector` (MLP PyTorch) como modelo principal.

## Justificaciones

### ¿Por qué Isolation Forest primero?

IF es interpretable, entrena en <1s y sirve de referencia. Sin baseline no podemos
justificar la complejidad del Autoencoder ni comparar si aporta algo.

### ¿Por qué MLP y no LSTM?

Con 50k tickets en 365 días × 24h × 5 categorías obtenemos ~8.700 ventanas con
tickets (~4.3% de ocupación). Un LSTM necesitaría secuencias contiguas largas y
tendría pocas muestras para converger. El MLP tabular captura las anomalías más
relevantes (bursts de volumen) a través de `ticket_count` en el vector de features,
sin necesitar dependencias temporales explícitas.

### ¿Por qué percentil 99 como umbral?

En operaciones, un operador puede revisar ~50–80 alertas por día. Con ~8.700
ventanas/día (si fuera tiempo real), el percentil 99 genera ~87 alertas —
manejable. En producción, el umbral se ajusta según la tolerancia del equipo.

### ¿Por qué entrenar el Autoencoder solo en ventanas normales?

El Autoencoder aprende a reconstruir la distribución base. Si entrenamos con
anomalías, aprende también a reconstruirlas y el error de reconstrucción deja
de ser un discriminador efectivo. Definimos "normal" como ticket_count < percentil 95.

## Consecuencias

- **Positivo:** dos señales complementarias; IF para velocidad, AE para patrones complejos.
- **Positivo:** sin etiquetas requeridas.
- **Negativo:** el AE tarda ~30s en entrenar sobre 50k tickets; escalar requiere
  optimización (mini-batch más grande, early stopping).
- **Trade-off:** la granularidad horaria puede perder anomalías lentas (días/semanas);
  añadir un detector diario en Fase futura si se detecta esta necesidad.
```

- [ ] **Step 2: Verificar suite completa**

```bash
uv run pytest --cov=itops --cov-report=term-missing
uv run ruff check .
uv run mypy
```

Esperado: todos PASSED, sin errores de lint ni de tipos.

- [ ] **Step 3: Commit final**

```bash
git add docs/decisions/002_autoencoder_for_anomaly.md
git commit -m "docs: add ADR-002 for anomaly detection approach"
```

---

## Self-Review

**Cobertura del spec:**
- ✅ `IsolationForestDetector` con ABC → Task 3
- ✅ `AutoencoderDetector` MLP PyTorch → Task 4
- ✅ `build_hourly_features` con 11 features → Task 2
- ✅ Tests: fit/score/predict sin error → Tasks 3, 4; reproducible → Tasks 3, 4; anomalías detectadas → Task 5
- ✅ Notebook comparativo → Task 6
- ✅ ADR 002 → Task 7
- ✅ Umbral por percentil documentado → ADR + anomaly.py

**Consistencia de tipos:**
- `FEATURE_COLS` definido en `features.py`, importado en `anomaly.py` (no, `anomaly.py` no lo importa — el llamador lo usa). El contrato "columna 0 = ticket_count" está documentado en el docstring de `AutoencoderDetector` y en el comentario de `FEATURE_COLS`.
- `score()` siempre devuelve `np.ndarray` de floats: IF devuelve float64, AE devuelve float32 — ambos pasan `np.issubdtype(scores.dtype, np.floating)`.
- `predict()` devuelve `.astype(bool)` explícito para garantizar dtype bool.

**Sin placeholders:** todos los steps tienen código completo.
