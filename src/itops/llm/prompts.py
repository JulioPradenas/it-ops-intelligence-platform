"""Templates de prompts parametrizables para el generador de narrativas LLM."""

from __future__ import annotations


def build_escalation_prompt(ticket_context: dict, top_features: list[dict]) -> str:
    """Construye el prompt para generar una narrativa de escalación en español.

    Args:
        ticket_context: dict con ticket_id, category, priority, customer_tier,
                        risk_score (float) y description_snippet (str, max 200 chars).
        top_features: lista de dicts {"feature": str, "shap": float}, top N features SHAP.

    Returns:
        Prompt listo para enviar al LLM. La respuesta esperada es JSON con campos
        summary, recommendation y confidence.
    """
    features_text = "\n".join(
        f"  - {f['feature']}: impacto SHAP = {f['shap']:.3f}"
        for f in top_features
    )
    return f"""Eres un analista de operaciones IT experto. Analiza el siguiente ticket y genera \
una narrativa de riesgo en español.

TICKET: {ticket_context['ticket_id']}
Categoría: {ticket_context['category']}
Prioridad: {ticket_context['priority']}
Tier del cliente: {ticket_context['customer_tier']}
Puntuación de riesgo de escalación: {ticket_context['risk_score']:.2f} (escala 0-1)
Descripción: {ticket_context['description_snippet']}

Factores principales que contribuyen al riesgo:
{features_text}

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{{
  "summary": "<resumen ejecutivo en 2 frases en español>",
  "recommendation": "<acción concreta recomendada al equipo en español>",
  "confidence": <número entre 0.0 y 1.0 que refleje la certeza de la predicción>
}}"""
