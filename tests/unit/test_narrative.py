"""Tests de la capa LLM — prompts y narrativas (Fase 4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from itops.llm.narrative import Narrative, NarrativeGenerator
from itops.llm.prompts import build_escalation_prompt

TICKET_CTX = {
    "ticket_id": "T-001",
    "category": "network",
    "priority": "high",
    "customer_tier": "enterprise",
    "risk_score": 0.87,
    "description_snippet": "Server unreachable since 10am",
}
TOP_FEAT = [
    {"feature": "customer_tier", "shap": 0.42},
    {"feature": "num_reassignments", "shap": 0.31},
    {"feature": "has_critical_keyword", "shap": 0.18},
]


def test_build_prompt_contains_context():
    prompt = build_escalation_prompt(TICKET_CTX, TOP_FEAT)
    assert "T-001" in prompt
    assert "0.87" in prompt
    assert "customer_tier" in prompt
    assert "enterprise" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 100


_CLAUDE_JSON = (
    '{"summary": "Riesgo alto detectado en ticket de red.", '
    '"recommendation": "Escalar al equipo senior inmediatamente.", '
    '"confidence": 0.87}'
)
_HF_JSON = (
    '{"summary": "Ticket de riesgo elevado.", '
    '"recommendation": "Revisar con el equipo de soporte.", '
    '"confidence": 0.60}'
)


def test_narrative_schema():
    n = Narrative(summary="Resumen.", recommendation="Acción.", confidence=0.75, provider="claude")
    assert 0.0 <= n.confidence <= 1.0
    assert len(n.summary) > 0
    assert len(n.recommendation) > 0
    assert n.provider in {"claude", "hf"}


def test_narrative_generator_calls_claude(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "claude"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.summary) > 0


def test_fallback_to_hf_on_api_error(tmp_path):
    mock_hf_pipe = MagicMock(return_value=[{"generated_text": _HF_JSON}])

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception(
            "API unavailable"
        )
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")
        gen._hf_pipeline = mock_hf_pipe
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "hf"
    assert 0.0 <= result.confidence <= 1.0


def test_cache_hit_skips_llm(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        create_fn = mock_anthropic.Anthropic.return_value.messages.create
        create_fn.return_value = mock_msg
        gen = NarrativeGenerator(api_key="test-key", cache_path=tmp_path / "cache.db")

        gen.generate(TICKET_CTX, TOP_FEAT)
        gen.generate(TICKET_CTX, TOP_FEAT)

    assert create_fn.call_count == 1
