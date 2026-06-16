"""Endpoints de la API REST de IT Ops Intelligence."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request

from itops.api.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    AnomalyWindow,
    EscalationRequest,
    EscalationResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ShapFeature,
)
from itops.data.features import FEATURE_COLS, build_hourly_features

router = APIRouter()


def _ticket_to_df(ticket) -> pd.DataFrame:
    """Convierte un TicketIn a DataFrame de una fila para build_ticket_features."""
    return pd.DataFrame([{
        "ticket_id": ticket.ticket_id,
        "created_at": pd.Timestamp(ticket.created_at),
        "category": ticket.category,
        "subcategory": ticket.subcategory,
        "priority_initial": ticket.priority_initial,
        "customer_tier": ticket.customer_tier,
        "description": ticket.description,
        "response_time_minutes": ticket.response_time_minutes,
        "num_comments": ticket.num_comments,
        "num_reassignments": ticket.num_reassignments,
        "business_hours": ticket.business_hours,
        "assigned_team": ticket.assigned_team,
        "escalated": False,
    }])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=getattr(request.app.state, "models_loaded", False),
    )


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly(body: AnomalyRequest, request: Request) -> AnomalyResponse:
    df = pd.concat([_ticket_to_df(t) for t in body.tickets]).reset_index(drop=True)

    hourly_feat = build_hourly_features(df)
    X = hourly_feat[FEATURE_COLS].values
    if_scores = request.app.state.if_detector.score(X)
    is_anomaly = request.app.state.if_detector.predict(X, percentile=97.0)

    windows = [
        AnomalyWindow(
            date=str(row["date"]),
            hour=int(row["hour"]),
            category=str(row["category"]),
            ticket_count=int(row["ticket_count"]),
            if_score=float(if_scores[i]),
            is_anomaly=bool(is_anomaly[i]),
        )
        for i, (_, row) in enumerate(hourly_feat.iterrows())
    ]
    return AnomalyResponse(anomalies=windows, total_windows=len(windows))


@router.post("/predict_escalation", response_model=EscalationResponse)
async def predict_escalation(body: EscalationRequest, request: Request) -> EscalationResponse:
    df = _ticket_to_df(body.ticket)
    risk_score = float(request.app.state.escalation_model.predict_proba(df)[0])
    predicted = bool(request.app.state.escalation_model.predict(df)[0])
    return EscalationResponse(
        ticket_id=body.ticket.ticket_id,
        risk_score=risk_score,
        predicted_escalation=predicted,
        threshold=float(request.app.state.escalation_model.threshold_),
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    df = _ticket_to_df(body.ticket)
    risk_score = float(request.app.state.escalation_model.predict_proba(df)[0])

    shap_row = request.app.state.shap_explainer.top_features(df, n=3).iloc[0]
    top_features_api = [
        ShapFeature(
            feature=str(shap_row[f"feature_{i}"]),
            shap_value=float(shap_row[f"shap_{i}"]),
        )
        for i in range(1, 4)
    ]

    ticket_context = {
        "ticket_id": body.ticket.ticket_id,
        "category": body.ticket.category,
        "priority": body.ticket.priority_initial,
        "customer_tier": body.ticket.customer_tier,
        "risk_score": risk_score,
        "description_snippet": body.ticket.description[:200],
    }
    top_features_llm = [
        {"feature": f.feature, "shap": f.shap_value}
        for f in top_features_api
    ]
    narrative = request.app.state.narrative_gen.generate(ticket_context, top_features_llm)

    return ExplainResponse(
        ticket_id=body.ticket.ticket_id,
        risk_score=risk_score,
        top_features=top_features_api,
        narrative=narrative,
    )
