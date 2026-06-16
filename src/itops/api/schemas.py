"""Modelos Pydantic v2 de request/response para la API REST."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from itops.llm.narrative import Narrative


class TicketIn(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str
    subcategory: str = "unknown"
    priority_initial: str
    customer_tier: str
    description: str
    response_time_minutes: int
    num_comments: int
    num_reassignments: int
    business_hours: bool
    assigned_team: str


class AnomalyRequest(BaseModel):
    tickets: list[TicketIn]


class AnomalyWindow(BaseModel):
    date: str
    hour: int
    category: str
    ticket_count: int
    if_score: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    anomalies: list[AnomalyWindow]
    total_windows: int


class EscalationRequest(BaseModel):
    ticket: TicketIn


class EscalationResponse(BaseModel):
    ticket_id: str
    risk_score: float
    predicted_escalation: bool
    threshold: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class ExplainRequest(BaseModel):
    ticket: TicketIn


class ExplainResponse(BaseModel):
    ticket_id: str
    risk_score: float
    top_features: list[ShapFeature]
    narrative: Narrative


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
