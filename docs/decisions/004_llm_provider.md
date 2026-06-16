# ADR-004: Proveedor LLM para narrativas de escalación

**Fecha:** 2026-06-15 (actualizado 2026-06-16)
**Estado:** actualizado

## Contexto

Necesitamos generar narrativas en español para tickets de alto riesgo de escalación.
Los consumidores son equipos de operaciones no técnicos — la calidad y naturalidad
del texto es importante. Las narrativas deben tener estructura fija (resumen,
recomendación, confianza) para ser consumibles por la API de Fase 5.

## Decisión

**Primario:** Claude API (`claude-haiku-4-5-20251001`)
**Fallback 1:** Groq API (`llama-3.1-8b-instant`)
**Fallback 2:** Template determinista (sin LLM)
**Caché:** SQLite en `data/processed/narrative_cache.db`

## Justificaciones

**Claude Haiku como primario:** Genera texto en español de alta calidad, respeta
instrucciones de formato JSON de forma confiable, y con Haiku el costo por narrativa
es <$0.001. El contexto largo de Claude permite incluir el prompt completo del ticket
sin truncar.

**Groq como fallback primario:** Groq ofrece inferencia de `llama-3.1-8b-instant`
con latencia muy baja (<1s) y tier gratuito generoso. Produce JSON estructurado
de forma más confiable que flan-t5-small. API compatible con el estándar OpenAI.

**Template determinista como fallback final:** Construye la narrativa directamente
desde `risk_score` y los top SHAP features sin llamada a ningún LLM. Garantiza
que siempre haya una narrativa disponible en CI, tests o entornos offline.

**Reemplaza flan-t5-small (decisión original):** flan-t5-small requería ~300MB de
descarga, generaba JSON inválido con frecuencia en español, y su calidad de texto era
muy inferior. Groq resuelve todos esos problemas con una API simple y gratuita.

**SQLite como caché:** Cero infraestructura extra, persiste entre sesiones del notebook
y la API (Fase 5), y la clave SHA256 garantiza determinismo. Solo se cachean narrativas
con `confidence > 0.0` para evitar envenenar el caché con respuestas fallidas.

**JSON estructurado sobre texto libre:** El tipo `Narrative` con `summary`,
`recommendation` y `confidence` es consumible directamente por el endpoint
`POST /explain` de Fase 5 sin parsing adicional.

## Trade-offs

- Si Claude y Groq no están disponibles, la narrativa template es funcional pero
  menos fluida. Adecuado para demos y CI.
- La primera llamada a Groq puede tener cold-start de ~200ms.
- Si se quiere escalar: añadir OpenAI/Mistral como tercer proveedor siguiendo
  el mismo patrón de fallback en `NarrativeGenerator`.

## Consecuencias

- `GROQ_API_KEY` en `.env` (gitignored) → narrativas de Groq como fallback de Claude.
- Sin ninguna API key → narrativa template determinista.
- La API de Fase 5 (`POST /explain`) recibe un `Narrative` serializado.
- Los tests nunca llaman APIs reales — usan `unittest.mock.patch`.
