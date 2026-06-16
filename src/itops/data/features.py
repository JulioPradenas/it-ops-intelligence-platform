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


# ---------------------------------------------------------------------------
# Features por ticket — predictor de escalación (Fase 3)
# ---------------------------------------------------------------------------

CRITICAL_KEYWORDS: list[str] = [
    "down", "outage", "production", "critical", "data loss", "unreachable",
]

TICKET_FEATURE_COLS: list[str] = [
    "response_time_minutes",
    "num_comments",
    "num_reassignments",
    "business_hours",
    "description_length",
    "has_critical_keyword",
    "hour_of_day",
    "day_of_week",
    "team_load_4h",
    "category",
    "subcategory",
    "priority_initial",
    "customer_tier",
    "assigned_team",
]

_CATEGORICAL_COLS: list[str] = [
    "category", "subcategory", "priority_initial", "customer_tier", "assigned_team",
]


def _compute_team_load_4h(df: pd.DataFrame) -> np.ndarray:
    """Tickets del mismo equipo creados en [t-4h, t) para cada ticket t."""
    teams = df["assigned_team"].to_numpy()
    timestamps = df["created_at"].to_numpy()
    result = np.zeros(len(df), dtype=np.int32)
    four_hours = np.timedelta64(4, "h")

    for team in np.unique(teams):
        pos = np.where(teams == team)[0]
        t_arr = timestamps[pos]
        order = np.argsort(t_arr, kind="stable")
        t_sorted = t_arr[order]
        pos_sorted = pos[order]

        for i in range(len(t_sorted)):
            lo = np.searchsorted(t_sorted, t_sorted[i] - four_hours, side="left")
            result[pos_sorted[i]] = i - lo

    return result


def build_ticket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce una fila por ticket con las features del predictor de escalación.

    No incluye el target `escalated` — extraerlo de df directamente.
    """
    work = df.copy()
    pattern = "|".join(CRITICAL_KEYWORDS)

    work["description_length"] = work["description"].str.len().astype(np.int32)
    work["has_critical_keyword"] = (
        work["description"].str.contains(pattern, case=False, regex=True).astype(np.int8)
    )
    work["hour_of_day"] = work["created_at"].dt.hour.astype(np.int8)
    work["day_of_week"] = work["created_at"].dt.weekday.astype(np.int8)
    work["business_hours"] = work["business_hours"].astype(np.int8)
    work["team_load_4h"] = _compute_team_load_4h(work)

    for col in _CATEGORICAL_COLS:
        work[col] = work[col].astype("category")

    return work[TICKET_FEATURE_COLS].copy()
