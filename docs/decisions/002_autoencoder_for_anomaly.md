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

### ¿Por qué percentil 97 como umbral operacional?

Las anomalías sembradas en el dataset están distribuidas entre p97 y p99.9 del
score de IF. Usar p97 garantiza que todos los bursts son detectados con una tasa
de falsos positivos del ~3% — manejable para un equipo de operaciones. En producción
el umbral se ajusta según la tolerancia del equipo.

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
