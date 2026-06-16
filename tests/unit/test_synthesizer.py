"""Tests del generador sintético (Fase 1).

Cubren los tres requisitos del plan: ratio de escalación esperado (±2%),
schema con los tipos definidos, y anomalías sembradas detectables.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.api import types as ptypes

from itops.data.synthesizer import SynthConfig, generate

# Tipo lógico esperado por columna (robusto entre pandas 2.x y 3.x, donde los
# strings pasan de "object" a "str" y los datetimes pueden ser ns/us).
EXPECTED_KIND: dict[str, str] = {
    "ticket_id": "string",
    "created_at": "datetime",
    "closed_at": "datetime",
    "category": "string",
    "subcategory": "string",
    "priority_initial": "string",
    "priority_final": "string",
    "assigned_team": "string",
    "assignee_id": "string",
    "customer_tier": "string",
    "description": "string",
    "response_time_minutes": "int",
    "num_comments": "int",
    "num_reassignments": "int",
    "business_hours": "bool",
    "escalated": "bool",
    "hours_to_escalation": "float",
}

_KIND_CHECKS = {
    "string": lambda s: ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s),
    "datetime": ptypes.is_datetime64_any_dtype,
    "int": ptypes.is_integer_dtype,
    "float": ptypes.is_float_dtype,
    "bool": ptypes.is_bool_dtype,
}


def test_escalation_ratio_within_tolerance(tickets: pd.DataFrame, test_config: SynthConfig) -> None:
    rate = tickets["escalated"].mean()
    assert abs(rate - test_config.target_escalation_rate) <= 0.02


def test_schema_columns_and_dtypes(tickets: pd.DataFrame) -> None:
    assert list(tickets.columns) == list(EXPECTED_KIND)
    for column, kind in EXPECTED_KIND.items():
        assert _KIND_CHECKS[kind](tickets[column]), f"{column} no es {kind}"


def test_row_count_matches_config(tickets: pd.DataFrame, test_config: SynthConfig) -> None:
    assert len(tickets) == test_config.n_tickets


def test_ticket_id_is_unique(tickets: pd.DataFrame) -> None:
    assert tickets["ticket_id"].is_unique


def test_categorical_values_are_valid(tickets: pd.DataFrame) -> None:
    assert set(tickets["category"]) <= {"network", "hardware", "software", "access", "other"}
    assert set(tickets["customer_tier"]) <= {"basic", "standard", "premium", "enterprise"}
    assert set(tickets["priority_final"]) <= {"low", "medium", "high", "critical"}


def test_hours_to_escalation_present_iff_escalated(tickets: pd.DataFrame) -> None:
    assert tickets.loc[tickets["escalated"], "hours_to_escalation"].notna().all()
    assert tickets.loc[~tickets["escalated"], "hours_to_escalation"].isna().all()


def test_critical_priority_only_when_escalated(tickets: pd.DataFrame) -> None:
    assert (tickets["priority_final"] == "critical").equals(tickets["escalated"])


def test_closed_after_created(tickets: pd.DataFrame) -> None:
    closed = tickets.dropna(subset=["closed_at"])
    assert (closed["closed_at"] >= closed["created_at"]).all()


def test_enterprise_escalates_more_than_basic(tickets: pd.DataFrame) -> None:
    by_tier = tickets.groupby("customer_tier")["escalated"].mean()
    assert by_tier["enterprise"] > by_tier["basic"]


def test_reassignments_increase_escalation(tickets: pd.DataFrame) -> None:
    high_reassign = tickets.loc[tickets["num_reassignments"] > 2, "escalated"].mean()
    low_reassign = tickets.loc[tickets["num_reassignments"] <= 2, "escalated"].mean()
    assert high_reassign > low_reassign


def test_reproducible_with_same_seed() -> None:
    a = generate(SynthConfig(n_tickets=2_000, seed=99)).tickets
    b = generate(SynthConfig(n_tickets=2_000, seed=99)).tickets
    pd.testing.assert_frame_equal(a, b)


def test_seeded_anomalies_are_detectable() -> None:
    # Cada burst sembrado debe sobresalir del volumen diario típico de su categoría.
    result = generate(SynthConfig(n_tickets=20_000, seed=7))
    df = result.tickets
    df = df.assign(date=df["created_at"].dt.date)
    daily_by_cat = df.groupby([df["date"], "category"]).size()
    for anomaly in result.seeded_anomalies:
        category = anomaly["category"]
        per_day = daily_by_cat.xs(category, level="category")
        threshold = per_day.mean() + 3 * per_day.std()
        anomaly_date = pd.Timestamp(anomaly["date"]).date()
        assert per_day.loc[anomaly_date] > threshold, anomaly


def test_too_many_anomalies_raises() -> None:
    with pytest.raises(ValueError):
        generate(SynthConfig(n_tickets=100, n_anomaly_days=6, anomaly_burst_size=250))
