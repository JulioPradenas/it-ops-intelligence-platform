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

**flan-t5-small como fallback:** Modelo seq2seq de ~300MB que funciona completamente
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
