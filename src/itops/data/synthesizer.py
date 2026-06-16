"""Generador de dataset sintético de tickets ITSM.

Produce tickets con patrones operacionales conocidos (estacionalidad horaria y
semanal, sesgos de escalación por tier/categoría/reasignaciones) y anomalías
sembradas (bursts de una categoría en días concretos). El objetivo es que las
fases posteriores tengan señal real que recuperar. Todo es reproducible vía
`seed`: misma config -> mismo DataFrame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# --- Vocabulario del dominio -------------------------------------------------
# Proporciones basadas en distribuciones típicas de mesas de ayuda ITSM.
CATEGORIES: list[str] = ["network", "hardware", "software", "access", "other"]
CATEGORY_PROBS: list[float] = [0.22, 0.18, 0.30, 0.20, 0.10]

SUBCATEGORIES: dict[str, list[str]] = {
    "network": ["vpn", "connectivity", "dns", "firewall", "latency"],
    "hardware": ["laptop", "server", "printer", "peripheral", "storage"],
    "software": ["app_crash", "license", "update", "performance", "bug"],
    "access": ["password_reset", "permissions", "account_lockout", "mfa", "provisioning"],
    "other": ["request", "inquiry", "documentation", "training", "misc"],
}

TIERS: list[str] = ["basic", "standard", "premium", "enterprise"]
TIER_PROBS: list[float] = [0.35, 0.35, 0.20, 0.10]

PRIORITIES: list[str] = ["low", "medium", "high"]
PRIORITY_PROBS: list[float] = [0.45, 0.40, 0.15]

TEAMS: list[str] = ["team_a", "team_b", "team_c", "team_d", "team_e"]
# Cada categoría tiene un equipo "natural"; el resto entra como ruido de routing.
CATEGORY_TEAM: dict[str, str] = {
    "network": "team_a",
    "hardware": "team_b",
    "software": "team_c",
    "access": "team_d",
    "other": "team_e",
}

# Pesos relativos de volumen por hora del día (picos 9-11h y 14-16h).
HOUR_WEIGHTS: np.ndarray = np.array(
    [0.20, 0.18, 0.15, 0.15, 0.18, 0.30, 0.55, 0.90, 1.60, 2.50, 2.70, 2.40,
     1.60, 1.50, 2.50, 2.60, 2.20, 1.40, 0.90, 0.60, 0.45, 0.35, 0.28, 0.22],
    dtype=float,
)
# Pesos relativos por día de la semana (0=lunes); peak los lunes, valle fin de semana.
WEEKDAY_WEIGHTS: np.ndarray = np.array([1.40, 1.15, 1.05, 1.00, 0.90, 0.40, 0.30], dtype=float)

# Efectos aditivos sobre el logit de escalación. El intercepto se resuelve por
# bisección para fijar la tasa global; estos efectos modelan los patrones del plan.
TIER_LOGIT: dict[str, float] = {"basic": 0.0, "standard": 0.40, "premium": 0.80, "enterprise": 1.20}
CATEGORY_LOGIT: dict[str, float] = {
    "network": 0.90,   # categoría con mayor tasa de escalación
    "hardware": 0.10,
    "software": 0.30,
    "access": -0.20,
    "other": -0.40,
}
PRIORITY_LOGIT: dict[str, float] = {"low": -0.50, "medium": 0.0, "high": 0.90}
REASSIGN_GT2_LOGIT: float = 3.5  # salto fuerte: >2 reasignaciones ~= 50% de escalar
OFFHOURS_LOGIT: float = 0.35     # incidentes fuera de horario escalan algo más

CRITICAL_KEYWORDS: list[str] = ["down", "outage", "production", "critical", "data loss", "unreachable"]


@dataclass
class SynthConfig:
    """Parámetros del generador. Los defaults producen el dataset de la Fase 1."""

    n_tickets: int = 50_000
    seed: int = 42
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    target_escalation_rate: float = 0.05
    n_anomaly_days: int = 6        # días con incidente sistémico sembrado
    anomaly_burst_size: int = 250  # tickets extra de una categoría ese día
    anomaly_window_hours: int = 4  # ventana horaria en que se concentra el burst
    open_ticket_fraction: float = 0.04  # tickets aún sin cerrar (closed_at nulo)


@dataclass
class GenerationResult:
    """DataFrame generado más la verdad-base de las anomalías sembradas."""

    tickets: pd.DataFrame
    seeded_anomalies: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tickets.to_csv(path, index=False)
        return path

    def save_anomalies(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": self.config, "anomalies": self.seeded_anomalies}, indent=2))
        return path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Encuentra el intercepto que hace que la prob. media de escalación = target.

    Bisección sobre el intercepto: garantiza la tasa global sin importar cómo
    se calibren los efectos individuales.
    """
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _sigmoid(mid + logits).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _plan_anomalies(rng: np.random.Generator, days: pd.DatetimeIndex, cfg: SynthConfig) -> list[dict]:
    """Selecciona días laborables y categorías para los bursts sembrados."""
    weekday_mask = days.weekday < 5  # incidentes visibles en días con volumen base alto
    candidate_idx = np.flatnonzero(weekday_mask)
    chosen = rng.choice(candidate_idx, size=cfg.n_anomaly_days, replace=False)
    anomalies: list[dict] = []
    for i, day_idx in enumerate(sorted(chosen)):
        category = CATEGORIES[i % len(CATEGORIES)]
        burst = int(cfg.anomaly_burst_size * rng.uniform(0.8, 1.3))
        start_hour = int(rng.integers(0, 24 - cfg.anomaly_window_hours))
        anomalies.append(
            {
                "date": days[day_idx].strftime("%Y-%m-%d"),
                "day_index": int(day_idx),
                "category": category,
                "burst_size": burst,
                "window_start_hour": start_hour,
                "window_hours": cfg.anomaly_window_hours,
            }
        )
    return anomalies


def _make_timestamps(
    rng: np.random.Generator, days: pd.DatetimeIndex, day_idx: np.ndarray, hours: np.ndarray
) -> pd.DatetimeIndex:
    minutes = rng.integers(0, 60, size=len(day_idx))
    seconds = rng.integers(0, 60, size=len(day_idx))
    base = days[day_idx]
    return (
        pd.to_datetime(base)
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )


def _build_descriptions(rng: np.random.Generator, faker: Faker, df: pd.DataFrame) -> list[str]:
    """Texto libre por ticket; inyecta keywords críticas según riesgo del ticket."""
    descriptions: list[str] = []
    for cat, sub, esc in zip(df["category"], df["subcategory"], df["escalated"], strict=True):
        # network y tickets escalados llevan keyword crítica con mayor frecuencia.
        keyword_prob = 0.55 if (esc or cat == "network") else 0.08
        opener = f"{sub.replace('_', ' ').capitalize()} issue reported"
        if rng.random() < keyword_prob:
            kw = CRITICAL_KEYWORDS[int(rng.integers(0, len(CRITICAL_KEYWORDS)))]
            descriptions.append(f"{opener}: service is {kw}. {faker.sentence(nb_words=8)}")
        else:
            descriptions.append(f"{opener}. {faker.sentence(nb_words=8)}")
    return descriptions


def generate(config: SynthConfig | None = None) -> GenerationResult:
    """Genera el dataset completo de tickets ITSM."""
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    faker = Faker()
    Faker.seed(cfg.seed)

    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D")

    # --- 1. Anomalías sembradas: reservan parte del presupuesto de tickets ---
    seeded = _plan_anomalies(rng, days, cfg)
    n_anomaly = sum(a["burst_size"] for a in seeded)
    n_base = cfg.n_tickets - n_anomaly
    if n_base <= 0:
        raise ValueError("n_tickets demasiado pequeño para los bursts configurados")

    # --- 2. Timestamps base con estacionalidad horaria y semanal ---
    weekday_w = WEEKDAY_WEIGHTS[days.weekday.to_numpy()]
    day_p = weekday_w / weekday_w.sum()
    base_day_idx = rng.choice(len(days), size=n_base, p=day_p)
    hour_p = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
    base_hours = rng.choice(24, size=n_base, p=hour_p)
    base_category = rng.choice(CATEGORIES, size=n_base, p=CATEGORY_PROBS)

    # --- 3. Tickets de los bursts: misma categoría, ventana horaria estrecha ---
    anom_day_idx: list[int] = []
    anom_hours: list[int] = []
    anom_category: list[str] = []
    for a in seeded:
        size = a["burst_size"]
        anom_day_idx.extend([a["day_index"]] * size)
        hi = a["window_start_hour"] + a["window_hours"]
        anom_hours.extend(rng.integers(a["window_start_hour"], hi, size=size).tolist())
        anom_category.extend([a["category"]] * size)

    day_idx = np.concatenate([base_day_idx, np.array(anom_day_idx, dtype=int)])
    hours = np.concatenate([base_hours, np.array(anom_hours, dtype=int)])
    category = np.concatenate([base_category, np.array(anom_category, dtype=object)])

    # Normaliza a resolución ns para consistencia con closed_at y el roundtrip a CSV.
    created_at = _make_timestamps(rng, days, day_idx, hours).astype("datetime64[ns]")
    df = pd.DataFrame({"created_at": created_at, "category": category})
    df = df.sort_values("created_at", ignore_index=True)
    n = len(df)

    # --- 4. Atributos categóricos por ticket ---
    df["subcategory"] = [
        SUBCATEGORIES[c][int(rng.integers(0, len(SUBCATEGORIES[c])))] for c in df["category"]
    ]
    df["customer_tier"] = rng.choice(TIERS, size=n, p=TIER_PROBS)
    df["priority_initial"] = rng.choice(PRIORITIES, size=n, p=PRIORITY_PROBS)
    # Routing: equipo natural de la categoría con 20% de ruido hacia otro equipo.
    natural_team = df["category"].map(CATEGORY_TEAM).to_numpy()
    noise_team = rng.choice(TEAMS, size=n)
    use_noise = rng.random(n) < 0.20
    df["assigned_team"] = np.where(use_noise, noise_team, natural_team)
    df["assignee_id"] = [f"agent_{int(rng.integers(1, 80)):03d}" for _ in range(n)]

    df["num_reassignments"] = np.minimum(rng.poisson(0.5, size=n), 6).astype(int)
    hour_of_day = df["created_at"].dt.hour
    df["business_hours"] = (df["created_at"].dt.weekday < 5) & hour_of_day.between(9, 17)

    # --- 5. Escalación: logit aditivo + intercepto calibrado a la tasa objetivo ---
    logits = (
        df["customer_tier"].map(TIER_LOGIT).to_numpy()
        + df["category"].map(CATEGORY_LOGIT).to_numpy()
        + df["priority_initial"].map(PRIORITY_LOGIT).to_numpy()
        + np.where(df["num_reassignments"] > 2, REASSIGN_GT2_LOGIT, 0.30 * df["num_reassignments"])
        + np.where(df["business_hours"], 0.0, OFFHOURS_LOGIT)
    )
    intercept = _solve_intercept(logits, cfg.target_escalation_rate)
    escalation_prob = _sigmoid(intercept + logits)
    df["escalated"] = rng.random(n) < escalation_prob

    # --- 6. Campos derivados de la escalación ---
    # priority_final = critical si escaló; si no, conserva la inicial.
    df["priority_final"] = np.where(df["escalated"], "critical", df["priority_initial"])
    # hours_to_escalation: lognormal (mediana ~4.5h), nulo si no escaló.
    hte = np.exp(rng.normal(1.5, 0.7, size=n))
    df["hours_to_escalation"] = np.where(df["escalated"], np.round(hte, 2), np.nan)

    # response_time_minutes: enterprise/high responden más rápido (mejor SLA).
    tier_speed = df["customer_tier"].map({"basic": 1.0, "standard": 0.85, "premium": 0.6, "enterprise": 0.4})
    prio_speed = df["priority_initial"].map({"low": 1.2, "medium": 1.0, "high": 0.6})
    response = np.exp(rng.normal(3.0, 0.6, size=n)) * tier_speed.to_numpy() * prio_speed.to_numpy()
    df["response_time_minutes"] = np.clip(np.round(response), 1, None).astype(int)

    # num_comments en la primera hora: más en alta prioridad y tickets escalados.
    comments_mean = (
        1.0
        + df["priority_initial"].map({"low": 0.0, "medium": 0.6, "high": 1.5}).to_numpy()
        + df["escalated"].to_numpy() * 1.5
    )
    df["num_comments"] = rng.poisson(comments_mean).astype(int)

    # closed_at: created + resolución (critical tarda más); fracción aún abierta -> NaT.
    base_res_hours = df["priority_final"].map({"low": 6.0, "medium": 12.0, "high": 24.0, "critical": 48.0})
    resolution = np.exp(rng.normal(0.0, 0.5, size=n)) * base_res_hours.to_numpy()
    closed = df["created_at"] + pd.to_timedelta(resolution, unit="h")
    is_open = rng.random(n) < cfg.open_ticket_fraction
    df["closed_at"] = closed.mask(is_open, pd.NaT)

    # --- 7. Descripción de texto libre (después de conocer escalated) ---
    df["description"] = _build_descriptions(rng, faker, df)

    df.insert(0, "ticket_id", [f"INC{i:07d}" for i in range(1, n + 1)])

    # Orden de columnas según el schema documentado en data/README.md.
    df = df[
        [
            "ticket_id", "created_at", "closed_at", "category", "subcategory",
            "priority_initial", "priority_final", "assigned_team", "assignee_id",
            "customer_tier", "description", "response_time_minutes", "num_comments",
            "num_reassignments", "business_hours", "escalated", "hours_to_escalation",
        ]
    ]
    return GenerationResult(tickets=df, seeded_anomalies=seeded, config=asdict(cfg))


def generate_dataframe(**kwargs: object) -> pd.DataFrame:
    """Atajo: devuelve solo el DataFrame con overrides de SynthConfig."""
    return generate(SynthConfig(**kwargs)).tickets  # type: ignore[arg-type]
