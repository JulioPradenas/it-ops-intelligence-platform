"""Generación de narrativas en lenguaje natural.

Cadena de proveedores: Claude → Groq (llama-3.1-8b-instant) → Template.
Cachea en SQLite para evitar llamadas redundantes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel, field_validator

from itops.config import PROCESSED_DIR
from itops.llm.prompts import build_escalation_prompt


class Narrative(BaseModel):
    """Narrativa de riesgo generada para un ticket de escalación."""

    summary: str
    recommendation: str
    confidence: float
    provider: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class NarrativeGenerator:
    """Genera narrativas en español. Cadena: Claude → Groq → Template."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        groq_api_key: str | None = None,
        cache_path: Path | str = PROCESSED_DIR / "narrative_cache.db",
    ) -> None:
        self._anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self._cache_path = Path(cache_path)
        self._init_cache()

    def _init_cache(self) -> None:
        if str(self._cache_path) != ":memory:":
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: la conexión se reusa entre reruns/threads
        # de Streamlit y el threadpool de FastAPI.
        self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
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
        # Some models wrap JSON in { } without code fences — try to find it
        if not clean.startswith("{"):
            brace = clean.find("{")
            if brace != -1:
                clean = clean[brace:]
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
                summary="",
                recommendation="",
                confidence=0.0,
                provider=provider,
            )

    def _call_claude(self, prompt: str) -> Narrative:
        if not self._anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=self._anthropic_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        text = getattr(block, "text", "")
        return self._parse_llm_response(str(text), provider="claude")

    def _call_groq(self, prompt: str) -> Narrative:
        if not self._groq_key:
            raise ValueError("GROQ_API_KEY not set")
        from groq import Groq  # noqa: PLC0415
        client = Groq(api_key=self._groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        text = response.choices[0].message.content or ""
        return self._parse_llm_response(text, provider="groq")

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
        narrative: Narrative | None = None
        self.last_errors: dict[str, str] = {}

        for name, caller in (("claude", self._call_claude), ("groq", self._call_groq)):
            try:
                result = caller(prompt)
                if result.confidence > 0.0:
                    narrative = result
                    break
                self.last_errors[name] = "respuesta sin JSON válido (confidence=0)"
            except Exception as exc:
                self.last_errors[name] = f"{type(exc).__name__}: {exc}"

        if narrative is None:
            narrative = self._build_template_narrative(ticket_context, top_features)

        # No cachear el template: si luego se configura una API key, queremos
        # reintentar el LLM en vez de servir el fallback cacheado.
        if narrative.confidence > 0.0 and narrative.provider != "template":
            self._cache_set(key, narrative)
        return narrative
