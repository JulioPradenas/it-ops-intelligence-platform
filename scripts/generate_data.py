"""Genera el dataset sintético de tickets y lo guarda en data/raw/.

Uso:
    python scripts/generate_data.py                 # 50k tickets, seed 42
    python scripts/generate_data.py --n 10000 --seed 7
"""

from __future__ import annotations

import argparse

from itops.config import RANDOM_SEED, RAW_TICKETS_CSV, SEEDED_ANOMALIES_JSON
from itops.data.synthesizer import SynthConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera tickets ITSM sintéticos")
    parser.add_argument("--n", type=int, default=50_000, help="número de tickets")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="semilla de reproducibilidad")
    args = parser.parse_args()

    result = generate(SynthConfig(n_tickets=args.n, seed=args.seed))
    csv_path = result.to_csv(RAW_TICKETS_CSV)
    anomalies_path = result.save_anomalies(SEEDED_ANOMALIES_JSON)

    rate = result.tickets["escalated"].mean()
    print(f"Generados {len(result.tickets):,} tickets -> {csv_path}")
    print(f"Tasa de escalación: {rate:.2%}")
    print(f"Anomalías sembradas: {len(result.seeded_anomalies)} -> {anomalies_path}")


if __name__ == "__main__":
    main()
