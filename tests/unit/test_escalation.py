"""Tests del predictor de escalación (Fase 3)."""

from __future__ import annotations

import lightgbm as lgb  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from pandas.api import types as ptypes

from itops.data.features import TICKET_FEATURE_COLS, build_ticket_features
from itops.data.synthesizer import SynthConfig, generate


@pytest.fixture(scope="module")
def small_df():
    return generate(SynthConfig(n_tickets=4_000, seed=42)).tickets


@pytest.fixture(scope="module")
def ticket_feat(small_df):
    return build_ticket_features(small_df)


def test_build_ticket_features_columns(ticket_feat):
    assert list(ticket_feat.columns) == TICKET_FEATURE_COLS


def test_build_ticket_features_no_nulls(ticket_feat):
    assert ticket_feat.isna().sum().sum() == 0


def test_build_ticket_features_numeric_cols(ticket_feat):
    numeric = [
        "response_time_minutes", "num_comments", "num_reassignments",
        "business_hours", "description_length", "has_critical_keyword",
        "hour_of_day", "day_of_week", "team_load_4h",
    ]
    for col in numeric:
        assert ptypes.is_numeric_dtype(ticket_feat[col]), col


def test_build_ticket_features_categorical_cols(ticket_feat):
    cats = ["category", "subcategory", "priority_initial", "customer_tier", "assigned_team"]
    for col in cats:
        assert isinstance(ticket_feat[col].dtype, pd.CategoricalDtype), col


def test_build_ticket_features_bounds(ticket_feat):
    assert ticket_feat["has_critical_keyword"].isin([0, 1]).all()
    assert ticket_feat["business_hours"].isin([0, 1]).all()
    assert (ticket_feat["team_load_4h"] >= 0).all()
    assert ticket_feat["hour_of_day"].between(0, 23).all()
    assert ticket_feat["day_of_week"].between(0, 6).all()


def test_team_load_first_ticket_is_zero(small_df):
    """El primer ticket de cada equipo (por tiempo) debe tener team_load_4h = 0."""
    df_sorted = small_df.sort_values("created_at").reset_index(drop=True)
    feat = build_ticket_features(df_sorted)
    first_per_team = df_sorted.groupby("assigned_team")["created_at"].idxmin()
    for idx in first_per_team:
        assert feat.loc[idx, "team_load_4h"] == 0, f"team first ticket at {idx}"


from itops.models.escalation import EscalationModel  # noqa: E402


@pytest.fixture(scope="module")
def fitted_model(small_df):
    model = EscalationModel(seed=42, n_estimators=50)  # 50 for fast tests
    model.fit(small_df)
    return model


def test_escalation_model_fits(fitted_model):
    assert fitted_model._lgbm_model is not None
    assert 0.0 < fitted_model.threshold_ < 1.0
    required_keys = {"auc_roc", "pr_auc", "f1", "precision", "recall", "cost_optimal", "cost_at_05"}
    assert required_keys.issubset(fitted_model.eval_metrics_)


def test_escalation_model_auc_reasonable(fitted_model):
    # Fixture uses n_estimators=50 for speed; full model (500 trees, 50k tickets) targets 0.80+
    assert fitted_model.eval_metrics_["auc_roc"] >= 0.70


def test_predictions_reproducible(small_df):
    a = EscalationModel(seed=7, n_estimators=30)
    a.fit(small_df)
    b = EscalationModel(seed=7, n_estimators=30)
    b.fit(small_df)
    np.testing.assert_array_almost_equal(a.predict_proba(small_df), b.predict_proba(small_df))


def test_threshold_beats_05_cost(fitted_model):
    assert fitted_model.eval_metrics_["cost_optimal"] <= fitted_model.eval_metrics_["cost_at_05"]


def test_predict_returns_bool_array(fitted_model, small_df):
    preds = fitted_model.predict(small_df)
    assert preds.dtype == bool
    assert preds.shape == (len(small_df),)


def test_predict_proba_before_fit_raises(small_df):
    model = EscalationModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(small_df)


from itops.models.explainer import ShapExplainer  # noqa: E402


def test_shap_explain_shape(fitted_model, small_df):
    exp = ShapExplainer(fitted_model)
    shap_df = exp.explain(small_df.head(50))
    assert shap_df.shape == (50, len(TICKET_FEATURE_COLS))
    assert list(shap_df.columns) == TICKET_FEATURE_COLS


def test_shap_top_features_columns(fitted_model, small_df):
    exp = ShapExplainer(fitted_model)
    top = exp.top_features(small_df.head(20), n=3)
    assert top.shape == (20, 6)  # 3 features × (name + value) = 6 cols
    expected_cols = ["feature_1", "shap_1", "feature_2", "shap_2", "feature_3", "shap_3"]
    assert list(top.columns) == expected_cols


def test_shap_before_fit_raises(small_df):
    unfitted = EscalationModel()
    with pytest.raises(RuntimeError, match="fitted"):
        ShapExplainer(unfitted)
