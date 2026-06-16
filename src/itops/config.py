"""Configuración centralizada de rutas y constantes del proyecto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROCESSED_DIR / "models"

RAW_TICKETS_CSV: Path = RAW_DIR / "tickets_synthetic.csv"
SEEDED_ANOMALIES_JSON: Path = RAW_DIR / "seeded_anomalies.json"

RANDOM_SEED: int = 42
