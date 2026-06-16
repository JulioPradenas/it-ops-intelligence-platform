"""Tests de la capa LLM — prompts y narrativas."""

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

_CLAUDE_JSON = (
    '{"summary": "Riesgo alto detectado en ticket de red.", '
    '"recommendation": "Escalar al equipo senior inmediatamente.", '
    '"confidence": 0.87}'
)
_GROQ_JSON = (
    '{"summary": "Ticket de riesgo elevado.", '
    '"recommendation": "Revisar con el equipo de soporte.", '
    '"confidence": 0.60}'
)


def test_build_prompt_contains_context():
    prompt = build_escalation_prompt(TICKET_CTX, TOP_FEAT)
    assert "T-001" in prompt
    assert "0.87" in prompt
    assert "customer_tier" in prompt
    assert "enterprise" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_narrative_schema():
    n = Narrative(summary="Resumen.", recommendation="Acción.", confidence=0.75, provider="claude")
    assert 0.0 <= n.confidence <= 1.0
    assert len(n.summary) > 0
    assert len(n.recommendation) > 0
    assert n.provider in {"claude", "groq", "template"}


def test_narrative_generator_calls_claude(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_msg
        gen = NarrativeGenerator(
            anthropic_api_key="test-key", cache_path=tmp_path / "cache.db"
        )
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "claude"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.summary) > 0


def test_fallback_to_groq_on_claude_error(tmp_path):
    mock_choice = MagicMock()
    mock_choice.message.content = _GROQ_JSON
    mock_groq_resp = MagicMock()
    mock_groq_resp.choices = [mock_choice]
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = mock_groq_resp

    with (
        patch("itops.llm.narrative.anthropic") as mock_anthropic,
        patch("groq.Groq", return_value=mock_groq_client),
    ):
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception(
            "API unavailable"
        )
        gen = NarrativeGenerator(
            anthropic_api_key="test-key",
            groq_api_key="groq-test-key",
            cache_path=tmp_path / "cache.db",
        )
        result = gen.generate(TICKET_CTX, TOP_FEAT)

    assert result.provider == "groq"
    assert 0.0 <= result.confidence <= 1.0


def test_cache_hit_skips_llm(tmp_path):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=_CLAUDE_JSON)]

    with patch("itops.llm.narrative.anthropic") as mock_anthropic:
        create_fn = mock_anthropic.Anthropic.return_value.messages.create
        create_fn.return_value = mock_msg
        gen = NarrativeGenerator(
            anthropic_api_key="test-key", cache_path=tmp_path / "cache.db"
        )

        gen.generate(TICKET_CTX, TOP_FEAT)
        gen.generate(TICKET_CTX, TOP_FEAT)

    assert create_fn.call_count == 1


def test_template_fallback_when_no_api_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    gen = NarrativeGenerator(cache_path=tmp_path / "cache.db")
    result = gen.generate(TICKET_CTX, TOP_FEAT)
    assert result.provider == "template"
    assert result.confidence > 0.0
    assert len(result.summary) > 0
