# Dataset — tickets ITSM sintéticos

Generado por `src/itops/data/synthesizer.py` (reproducible vía `seed`). Regenerar con:

```bash
python scripts/generate_data.py            # 50.000 tickets, seed 42
```

- `raw/tickets_synthetic.csv` — un ticket por fila.
- `raw/seeded_anomalies.json` — verdad-base de las anomalías sembradas (día,
  categoría, tamaño del burst y ventana horaria). Lo usa el EDA y los tests para
  validar la detección.
- `processed/` — features derivadas (Fase 3).

## Schema

| Columna | Tipo | Descripción |
|---|---|---|
| `ticket_id` | string | ID único, formato `INC0000001` |
| `created_at` | datetime | Apertura del ticket |
| `closed_at` | datetime (nullable) | Cierre; nulo si sigue abierto (~4%) |
| `category` | string | `network`, `hardware`, `software`, `access`, `other` |
| `subcategory` | string | Granularidad dentro de la categoría |
| `priority_initial` | string | `low`, `medium`, `high` |
| `priority_final` | string | `low`, `medium`, `high`, `critical` (`critical` ⇔ escaló) |
| `assigned_team` | string | `team_a` … `team_e` (equipo natural por categoría + 20% ruido) |
| `assignee_id` | string | `agent_NNN` |
| `customer_tier` | string | `basic`, `standard`, `premium`, `enterprise` |
| `description` | string | Texto libre; keywords críticas más frecuentes en riesgo |
| `response_time_minutes` | int | Tiempo a primera respuesta (menor en tiers altos / alta prioridad) |
| `num_comments` | int | Comentarios en la primera hora |
| `num_reassignments` | int | Veces que cambió de equipo |
| `business_hours` | bool | Creado en horario laboral (L-V, 9-17h) |
| `escalated` | bool | **Target principal** |
| `hours_to_escalation` | float (nullable) | Horas hasta escalar; nulo si no escaló |

## Patrones inyectados (verdad-base para fases posteriores)

- **Estacionalidad horaria:** picos 9-11h y 14-16h.
- **Estacionalidad semanal:** peak los lunes, valle el fin de semana.
- **Tier:** `enterprise` escala ~3× más que `basic`.
- **Reasignaciones:** `num_reassignments > 2` ⇒ ~50% de probabilidad de escalar.
- **Categoría:** `network` tiene la tasa de escalación más alta.
- **Anomalías sembradas:** días concretos con burst de tickets de una sola
  categoría (incidente sistémico), concentrados en una ventana horaria.

La tasa global de escalación se calibra a ~5% (intercepto del logit resuelto por
bisección), independientemente de los efectos individuales.
