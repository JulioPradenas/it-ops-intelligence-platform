"""FastAPI application con lifespan para IT Ops Intelligence."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from itops.api.routes import router
from itops.config import MODELS_DIR
from itops.llm.narrative import NarrativeGenerator
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel
from itops.models.explainer import ShapExplainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.escalation_model = EscalationModel.load(MODELS_DIR / "escalation_model.pkl")
    app.state.if_detector = IsolationForestDetector.load(MODELS_DIR / "if_detector.pkl")
    app.state.ae_detector = AutoencoderDetector.load(
        MODELS_DIR / "ae_detector.pkl", MODELS_DIR / "ae_weights.pt"
    )
    app.state.shap_explainer = ShapExplainer(app.state.escalation_model)
    app.state.narrative_gen = NarrativeGenerator()
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False


app = FastAPI(title="IT Ops Intelligence API", lifespan=lifespan)
app.include_router(router)
