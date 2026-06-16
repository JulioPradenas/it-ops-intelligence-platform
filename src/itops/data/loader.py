"""Carga del dataset de tickets desde CSV (y, en fases futuras, SQLite)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from itops.config import RAW_TICKETS_CSV

# Columnas de fecha que deben parsearse a datetime al cargar desde CSV.
_DATETIME_COLUMNS: list[str] = ["created_at", "closed_at"]


def load_tickets(path: str | Path = RAW_TICKETS_CSV) -> pd.DataFrame:
    """Carga el CSV de tickets parseando timestamps y tipos básicos."""
    return pd.read_csv(path, parse_dates=_DATETIME_COLUMNS)
