# Fase 2 — Anomaly Detection: Design Spec

**Fecha:** 2026-06-15
**Fase del plan:** 2 de 6
**Estado:** aprobado

---

## Contexto

El dataset sintético (Fase 1) contiene 50.000 tickets ITSM con 6 anomalías sembradas: bursts de una categoría específica concentrados en una ventana horaria de 4h (verdad-base en `data/raw/seeded_anomalies.json`). El objetivo de Fase 2 es construir dos detectores que recuperen esas anomalías y queden listos para integrarse en la API de Fase 5.

---

## Unidad de detección

**Ventana temporal: `(date, hour, category)`.**

Cada fila del input a los modelos representa una hora calendario para una categoría concreta. Con 50k tickets en 365 días × 24h × 5 categorías el dataset de features tiene ~43.800 ventanas densas. Los bursts sembrados (250–325 tickets extra en 4h) son claramente detectables a esta granularidad sin diluirse como ocurriría con ventanas diarias.

El API de Fase 5 (`POST /anomaly`) recibirá un batch de tickets crudos, los agregará internamente con `build_hourly_features` y devolverá las ventanas marcadas como anómalas.

---

## Pipeline de features

**Módulo:** `src/itops/data/features.py`
**Función principal:** `build_hourly_features(df: pd.DataFrame) -> pd.DataFrame`

Agrega el DataFrame crudo de tickets a ventanas horarias por categoría. Cada ventana produce:

| Columna | Descripción |
|---|---|
| `ticket_count` | señal principal de burst |
| `escalated_count` | conteo de escalaciones en la ventana |
| `escalation_rate` | proporción escalada (0 si ticket_count=0) |
| `avg_response_time_minutes` | velocidad media de primera respuesta |
| `avg_num_comments` | actividad media en tickets |
| `avg_num_reassignments` | routing caótico |
| `pct_off_hours` | fracción de tickets fuera de horario laboral |
| `hour_sin`, `hour_cos` | codificación cíclica de la hora del día |
| `weekday_sin`, `weekday_cos` | codificación cíclica del día de la semana |

Ventanas sin tickets → `ticket_count=0`, métricas de tickets = 0 (no NaN). Los modelos reciben la matriz ya limpia; la imputación ocurre en `build_hourly_features`.

`features.py` no contiene lógica de ML: es puro pandas, testeable sin modelos.

---

## Detectores

**Módulo:** `src/itops/models/anomaly.py`

### Protocolo común `AnomalyDetector` (ABC)

```python
class AnomalyDetector(ABC):
    def fit(self, X: np.ndarray) -> None: ...
    def score(self, X: np.ndarray) -> np.ndarray: ...       # float, mayor = más anómalo
    def predict(self, X: np.ndarray, percentile: float = 99.0) -> np.ndarray: ...  # bool
```

Ambos detectores normalizan internamente (StandardScaler) para que el llamador no gestione el escalado.

### `IsolationForestDetector`

- Wrapper de `sklearn.ensemble.IsolationForest`.
- Scaler + IF encadenados internamente.
- `score()` = negative anomaly score de IF, invertido para que valores altos sean más anómalos.
- Threshold de `predict()` = percentil `p` de los scores de entrenamiento.
- Reproducible con `random_state=seed`.
- Serializable con pickle.

### `AutoencoderDetector`

- MLP PyTorch: `input_dim → 32 → 16 → 8 → 16 → 32 → input_dim`.
- Activaciones ReLU entre capas ocultas; sin activación en la capa de salida (valores normalizados continuos).
- Entrenado **solo con ventanas normales** (percentil <95 de `ticket_count`) para que el error de reconstrucción sea bajo en la distribución base.
- `score()` = MSE de reconstrucción por ventana.
- Threshold de `predict()` = percentil `p` del error de entrenamiento.
- Reproducible con `torch.manual_seed(seed)`.
- Hiperparámetros configurables: `hidden_dims`, `epochs`, `lr`, `batch_size`.

---

## Tests (`tests/unit/test_anomaly.py`)

| Test | Qué verifica |
|---|---|
| `test_build_hourly_features_shape` | columnas correctas, sin NaN, tipos numéricos |
| `test_if_detector_fits_and_scores` | IsolationForest entrena sin error, shape del output |
| `test_autoencoder_fits_and_scores` | Autoencoder entrena en datos pequeños, shape del output |
| `test_scores_reproducible_with_seed` | misma semilla → mismo array (IF y Autoencoder) |
| `test_seeded_anomalies_detected` | las ventanas del burst tienen score > threshold (percentil 99) |

Todos los tests usan datos sintéticos pequeños (generados en conftest) para correr en <5s.
`test_seeded_anomalies_detected` usa el dataset completo desde `seeded_anomalies.json`.

---

## Notebook `notebooks/02_anomaly_detection.ipynb`

1. **Feature engineering** — stats descriptivas de las 43k+ ventanas, distribución de `ticket_count`.
2. **IsolationForest** — entrena, distribución de scores, top-10 ventanas más anómalas con fecha/hora/categoría.
3. **Autoencoder** — curva de pérdida, distribución del error de reconstrucción, top-10 ventanas.
4. **Comparación** — ¿las 6 anomalías sembradas aparecen en el top de ambos detectores? Tabla comparativa.
5. **Conclusión** — cuándo usar cada modelo: IF para velocidad y explicabilidad, Autoencoder para patrones de co-ocurrencia que IF no captura.

---

## ADR `docs/decisions/002_autoencoder_for_anomaly.md`

Documenta:
- Por qué IF primero (baseline obligatorio antes de añadir complejidad).
- Por qué MLP y no LSTM (volumen de datos insuficiente para secuencias; bursts detectables en features tabulares).
- Por qué percentil 99 como umbral operacional.
- Trade-offs: IF es más explicable, Autoencoder es más sensible a patrones multivariados.

---

## Archivos a crear/modificar

| Archivo | Estado actual | Acción |
|---|---|---|
| `src/itops/data/features.py` | stub | implementar |
| `src/itops/models/anomaly.py` | stub | implementar |
| `tests/unit/test_anomaly.py` | no existe | crear |
| `notebooks/02_anomaly_detection.ipynb` | no existe | crear |
| `docs/decisions/002_autoencoder_for_anomaly.md` | no existe | crear |

**No se tocan:** `synthesizer.py`, `loader.py`, `config.py`, tests existentes.
