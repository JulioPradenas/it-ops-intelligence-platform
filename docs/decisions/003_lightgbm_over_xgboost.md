# ADR-003: LightGBM sobre XGBoost para predicción de escalación

**Fecha:** 2026-06-15
**Estado:** aceptado

## Contexto

Necesitamos un clasificador de escalación de tickets con estas características:
- Dataset de tamaño medio (~50k tickets, ~5% positivos)
- Features mixtas: numéricas, categóricas nativas, texto derivado
- Explicabilidad por SHAP requerida
- Tiempo de entrenamiento razonable para re-entrenar en CI

## Decisión

Usar LightGBM (`lightgbm.LGBMClassifier`) como clasificador principal.

## Justificaciones

**Velocidad:** LightGBM es 2-10× más rápido que XGBoost en datasets de este
tamaño gracias a su algoritmo histogram-based y leaf-wise tree growth.

**Categorías nativas:** LightGBM maneja columnas con dtype `pd.Categorical`
sin necesidad de one-hot encoding. XGBoost requiere encoding explícito, lo que
añade complejidad al pipeline y pierde información de ordinalidad.

**Desbalance:** `class_weight='balanced'` funciona bien con el ~5% de positivos.
Alternativas (scale_pos_weight, sample_weight) producen resultados similares
con más configuración.

**SHAP:** `shap.TreeExplainer` funciona igualmente bien con LightGBM y XGBoost.
La elección del clasificador no afecta la calidad de las explicaciones.

**Trade-offs:**

- XGBoost tiene mejor documentación en inglés y más ejemplos en literatura.
- LightGBM puede ser más sensible a overfitting con datasets pequeños
  (mitigado con `num_leaves=31` conservador y `n_estimators=500` + early stopping
  si se añade en el futuro).

## Consecuencias

- El modelo serializado (Fase 6) requiere `lightgbm` instalado.
- El API de Fase 5 carga el modelo con `lgb.Booster` o `pickle` del estimator sklearn.
- Si se necesita escalar a GPU, LightGBM soporta CUDA nativo; migración es directa.
