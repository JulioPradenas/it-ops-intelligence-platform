"""Generación de narrativas en lenguaje natural con Claude + fallback HF.

NarrativeGenerator intenta Claude primero, cae al pipeline HF si falla,
y cachea resultados en SQLite para evitar llamadas redundantes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, field_validator

from itops.config import PROCESSED_DIR
from itops.llm.prompts import build_escalation_prompt


class Narrative(BaseModel):
    """Narrativa de riesgo generada por LLM para un ticket de escalación."""

    summary: str
    recommendation: str
    confidence: float
    provider: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class NarrativeGenerator:
    """Genera narrativas en español usando Claude con fallback a flan-t5-small."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | str = PROCESSED_DIR / "narrative_cache.db",
        hf_model: str = "google/flan-t5-small",
    ) -> None:
        self._api_key = api_key
        self._cache_path = Path(cache_path)
        self._hf_model = hf_model
        self._hf_pipeline: Any = None
        self._init_cache()

    def _init_cache(self) -> None:
        if str(self._cache_path) != ":memory:":
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._cache_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS narrative_cache "
            "(key TEXT PRIMARY KEY, data TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def _cache_key(self, ticket_context: dict, top_features: list[dict]) -> str:
        content = repr((ticket_context, top_features)).encode()
        return hashlib.sha256(content).hexdigest()

    def _cache_get(self, key: str) -> Narrative | None:
        row = self._conn.execute(
            "SELECT data FROM narrative_cache WHERE key = ?", (key,)
        ).fetchone()
        return Narrative.model_validate_json(row[0]) if row else None

    def _cache_set(self, key: str, narrative: Narrative) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO narrative_cache (key, data, created_at) VALUES (?, ?, ?)",
            (key, narrative.model_dump_json(), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def _parse_llm_response(self, text: str, provider: str) -> Narrative:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        clean = match.group(1).strip() if match else text.strip()
        try:
            data = json.loads(clean)
            return Narrative(
                summary=str(data["summary"]),
                recommendation=str(data["recommendation"]),
                confidence=float(data.get("confidence", 0.5)),
                provider=provider,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return Narrative(
                summary="No se pudo generar un resumen automático.",
                recommendation="Revisar el ticket manualmente con el equipo de soporte.",
                confidence=0.0,
                provider=provider,
            )

    def _call_claude(self, prompt: str) -> Narrative:
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        text = getattr(block, "text", "")
        return self._parse_llm_response(str(text), provider="claude")

    def _call_hf(self, prompt: str, ticket_context: dict | None = None) -> Narrative:
        if self._hf_pipeline is None:
            from transformers import pipeline  # noqa: PLC0415

            self._hf_pipeline = pipeline("text-generation", model=self._hf_model)  # type: ignore[call-overload]
        result = self._hf_pipeline(prompt, max_new_tokens=200)
        text = result[0]["generated_text"]
        parsed = self._parse_llm_response(text, provider="hf")
        # flan-t5-small rarely generates valid JSON; build a minimal narrative from context.
        if parsed.confidence == 0.0 and ticket_context:
            risk = ticket_context.get("risk_score", 0.0)
            category = ticket_context.get("category", "desconocida")
            tier = ticket_context.get("customer_tier", "")
            return Narrative(
                summary=(
                    f"Ticket de categoría '{category}' con riesgo de escalación "
                    f"{risk:.0%} (cliente {tier}). Requiere atención prioritaria."
                ),
                recommendation=(
                    "Verificar SLA del cliente, reasignar a técnico senior "
                    "y notificar al responsable del área."
                ),
                confidence=round(float(risk), 2),
                provider="hf",
            )
        return parsed

    def _build_template_narrative(
        self, ticket_context: dict, top_features: list[dict]
    ) -> Narrative:
        """Narrativa determinista desde el contexto — sin llamada a LLM."""
        risk = float(ticket_context.get("risk_score", 0.0))
        category = ticket_context.get("category", "desconocida")
        tier = ticket_context.get("customer_tier", "")
        priority = ticket_context.get("priority", "")
        feat_names = [f["feature"] for f in top_features[:2]] if top_features else []
        feat_str = " y ".join(feat_names) if feat_names else "múltiples indicadores"

        summary = (
            f"Ticket de categoría '{category}' (cliente {tier}, prioridad {priority}) "
            f"con {risk:.0%} de probabilidad de escalación. "
            f"Factores determinantes: {feat_str}."
        )
        if risk >= 0.8:
            recommendation = (
                f"Escalación inmediata. Asignar técnico senior de '{category}' "
                f"y contactar al cliente {tier} en menos de 30 minutos."
            )
        elif risk >= 0.5:
            recommendation = (
                f"Revisión prioritaria en la próxima hora. Verificar SLA del cliente {tier} "
                f"y considerar reasignación si no hay progreso."
            )
        else:
            recommendation = (
                f"Monitorear evolución. Actualizar al cliente {tier} con el estado actual."
            )
        return Narrative(
            summary=summary,
            recommendation=recommendation,
            confidence=round(risk, 2),
            provider="template",
        )

    def generate(self, ticket_context: dict, top_features: list[dict]) -> Narrative:
        """Genera o recupera del caché una narrativa para el ticket dado."""
        key = self._cache_key(ticket_context, top_features)
        cached = self._cache_get(key)
        if cached and cached.confidence > 0.0:
            return cached

        prompt = build_escalation_prompt(ticket_context, top_features)
        try:
            narrative = self._call_claude(prompt)
        except Exception:
            try:
                narrative = self._call_hf(prompt, ticket_context=ticket_context)
            except Exception:
                narrative = self._build_template_narrative(ticket_context, top_features)

        if narrative.confidence > 0.0:
            self._cache_set(key, narrative)
        return narrative
