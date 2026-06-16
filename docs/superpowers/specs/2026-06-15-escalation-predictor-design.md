# Fase 3 — Escalation Predictor: Design Spec

**Fecha:** 2026-06-15
**Fase del plan:** 3 de 6
**Estado:** aprobado

---

## Contexto

Con el dataset sintético de 50k tickets (Fase 1) y los detectores de anomalías (Fase 2) implementados, el siguiente paso es un clasificador supervisado que prediga qué tickets escalarán a crítico antes de que lo hagan. El target es `escalated` (bool, ~5% positivos). El modelo debe ser explicable por ticket vía SHAP para que el equipo de operaciones entienda por qué un ticket fue marcado como riesgo.

---

## Feature engineering (`features.py`)

**Nueva función:** `build_ticket_features(df: pd.DataFrame) -> pd.DataFrame`

Produce una fila por ticket. Coexiste con `build_hourly_features` en el mismo módulo.

| Feature | Tipo | Origen |
|---|---|---|
| `response_time_minutes` | int | directo del dataset |
| `num_comments` | int | directo |
| `num_reassignments` | int | directo |
| `business_hours` | int (0/1) | cast de bool |
| `description_length` | int | `len(description)` |
| `has_critical_keyword` | int (0/1) | presencia de cualquiera de: "down", "outage", "production", "critical", "data loss", "unreachable" |
| `hour_of_day` | int | `created_at.dt.hour` |
| `day_of_week` | int | `created_at.dt.weekday` (0=lunes) |
| `team_load_4h` | int | tickets del mismo `assigned_team` en las 4h anteriores al `created_at` de este ticket (merge ordenado, ventana rodante) |
| `category` | category | dtype `pd.Categorical` — LightGBM nativo |
| `subcategory` | category | dtype `pd.Categorical` |
| `priority_initial` | category | dtype `pd.Categorical` |
| `customer_tier` | category | dtype `pd.Categorical` |
| `assigned_team` | category | dtype `pd.Categorical` |

**Target:** `escalated` (bool) — NO se incluye en el DataFrame de features, se extrae por separado.

**`TICKET_FEATURE_COLS`:** lista exportada con los nombres de todas las features numéricas + categóricas (sin `ticket_id`, `created_at`, `escalated` ni columnas de derivación interna). Usada por el modelo y los tests.

### `team_load_4h` — implementación

Para cada ticket, contar cuántos otros tickets del mismo equipo tienen `created_at` en `(created_at - 4h, created_at)`. Algoritmo:

1. Ordenar el DataFrame por `created_at`.
2. Para cada equipo, usar `pd.merge_asof` o un loop vectorizado con `searchsorted` sobre timestamps ordenados.
3. El resultado no incluye el ticket actual: se cuentan tickets con `created_at ∈ [t-4h, t)` — abierto por la izquierda, estrictamente menor que `t`.

---

## Modelo (`src/itops/models/escalation.py`)

**Clase:** `EscalationModel`

```
EscalationModel
├── fit(df: pd.DataFrame) -> None
│     Llama a build_ticket_features, hace split temporal 80/20,
│     entrena LGBMClassifier, optimiza threshold sobre val set.
├── predict_proba(df: pd.DataFrame) -> np.ndarray
│     Retorna array 1D de probabilidades de escalación.
├── predict(df: pd.DataFrame) -> np.ndarray
│     Umbraliza predict_proba con self.threshold_ → bool array.
├── threshold_: float
│     Umbral óptimo: minimiza cost = 5·FN + 1·FP sobre validation set.
│     Grid search: np.arange(0.01, 1.0, 0.01).
└── eval_metrics_: dict
      AUC-ROC, PR-AUC, F1, Precision, Recall, cost_optimal, cost_at_05
      Calculados sobre el holdout temporal (último 20% por fecha).
```

**Split temporal:** `df.sort_values('created_at')` → primeros 80% para train, últimos 20% para test. Sin shuffle. Sin data leakage.

**LGBMClassifier params base:**
- `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`
- `class_weight='balanced'` — maneja el desbalance ~5% positivos
- `random_state=42` — reproducibilidad
- `categorical_feature='auto'` — LightGBM detecta columnas con dtype `category`

**Optimización del threshold:**

```python
costs = [5 * FN(t) + 1 * FP(t) for t in np.arange(0.01, 1.0, 0.01)]
threshold_ = np.arange(0.01, 1.0, 0.01)[np.argmin(costs)]
```

`eval_metrics_['cost_optimal'] < eval_metrics_['cost_at_05']` debe ser True.

---

## Explainer (`src/itops/models/explainer.py`)

**Clase:** `ShapExplainer`

```
ShapExplainer
├── __init__(model: EscalationModel)
│     Instancia shap.TreeExplainer con model._lgbm_model (LGBMClassifier fitted).
├── explain(df: pd.DataFrame) -> pd.DataFrame
│     Devuelve DataFrame de SHAP values de la clase positiva (escalation=True).
│     TreeExplainer.shap_values() devuelve [class0, class1]; se usa class1.
│     Columnas = TICKET_FEATURE_COLS, índice = df.index.
└── top_features(df: pd.DataFrame, n: int = 3) -> pd.DataFrame
      Por cada ticket: top-N features por |SHAP value| con columnas
      [feature_1, shap_1, feature_2, shap_2, ..., feature_N, shap_N].
```

`EscalationModel` expone `_lgbm_model: LGBMClassifier` (atributo interno, accesible para el explainer).

Se usa `shap.TreeExplainer` (no `KernelExplainer`) — exacto, rápido (~1s/10k tickets), compatible nativo con LightGBM.

---

## Tests (`tests/unit/test_escalation.py`)

| Test | Qué verifica |
|---|---|
| `test_build_ticket_features_schema` | TICKET_FEATURE_COLS presente, tipos correctos (numéricos + category), sin nulos |
| `test_team_load_4h_is_non_negative` | `team_load_4h >= 0` para todos los tickets |
| `test_escalation_model_fits` | `fit()` sin error, `threshold_` float en (0,1), `eval_metrics_` tiene todas las claves |
| `test_predictions_reproducible` | misma semilla → `predict_proba` idéntico |
| `test_threshold_beats_05_cost` | `cost_optimal < cost_at_05` en el holdout |
| `test_shap_shape` | `explain(df)` devuelve DataFrame con shape (n, len(TICKET_FEATURE_COLS)) |
| `test_top_features_columns` | `top_features(df, n=3)` devuelve 6 columnas por ticket |

Todos los tests usan el dataset de 8k tickets del conftest (rápido, sin cargar 50k).

---

## Notebook `notebooks/03_escalation_model.ipynb`

1. **Feature engineering** — distribución de features numéricas, top categorías.
2. **Entrenamiento** — split temporal, curva de aprendizaje.
3. **Métricas en holdout** — AUC-ROC curve, PR curve, tabla comparativa threshold 0.5 vs óptimo.
4. **Optimización del threshold** — curva de costo vs threshold, threshold óptimo marcado.
5. **SHAP** — summary plot (importancia global), waterfall plot de un ticket de alto riesgo.

---

## ADR `docs/decisions/003_lightgbm_over_xgboost.md`

Documenta: velocidad de entrenamiento, manejo nativo de categorías, mejor performance con datasets medianos desbalanceados, y que SHAP funciona igual en ambos.

---

## Archivos a crear/modificar

| Archivo | Estado actual | Acción |
|---|---|---|
| `src/itops/data/features.py` | implementado | agregar `build_ticket_features` y `TICKET_FEATURE_COLS` |
| `src/itops/models/escalation.py` | stub | implementar `EscalationModel` |
| `src/itops/models/explainer.py` | stub | implementar `ShapExplainer` |
| `tests/unit/test_escalation.py` | no existe | crear |
| `notebooks/03_escalation_model.ipynb` | no existe | crear |
| `docs/decisions/003_lightgbm_over_xgboost.md` | no existe | crear |

**No se tocan:** `anomaly.py`, `synthesizer.py`, tests existentes (25 pasando).

## Dependencias nuevas a agregar

- `lightgbm>=4.3`
- `shap>=0.45`
