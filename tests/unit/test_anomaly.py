"""Tests de features y detectores de anomalías (Fase 2)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pandas.api import types as ptypes

from itops.config import RAW_TICKETS_CSV, SEEDED_ANOMALIES_JSON
from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.data.loader import load_tickets
from itops.data.synthesizer import SynthConfig, generate
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector


@pytest.fixture(scope="module")
def small_df():
    return generate(SynthConfig(n_tickets=3_000, seed=42)).tickets


@pytest.fixture(scope="module")
def features_df(small_df):
    return build_hourly_features(small_df)


@pytest.fixture(scope="module")
def X_small(features_df):
    return features_df[FEATURE_COLS].values


def test_build_hourly_features_columns(features_df):
    assert set(FEATURE_COLS).issubset(features_df.columns)
    assert "date" in features_df.columns
    assert "hour" in features_df.columns
    assert "category" in features_df.columns


def test_build_hourly_features_no_nulls(features_df):
    assert features_df[FEATURE_COLS].isna().sum().sum() == 0


def test_build_hourly_features_numeric_dtypes(features_df):
    for col in FEATURE_COLS:
        assert ptypes.is_numeric_dtype(features_df[col]), col


def test_build_hourly_features_ticket_count_positive(features_df):
    # Solo incluye ventanas con tickets
    assert (features_df["ticket_count"] > 0).all()


def test_build_hourly_features_escalation_rate_bounded(features_df):
    assert features_df["escalation_rate"].between(0, 1).all()


def test_if_fits_and_scores(X_small):
    det = IsolationForestDetector(seed=42)
    det.fit(X_small)
    scores = det.score(X_small)
    assert scores.shape == (len(X_small),)
    assert np.issubdtype(scores.dtype, np.floating)


def test_if_predict_fraction(X_small):
    det = IsolationForestDetector(seed=42)
    det.fit(X_small)
    flags = det.predict(X_small, percentile=99.0)
    assert flags.dtype == bool
    # percentil 99 → ~1% marcado
    assert 0.005 <= flags.mean() <= 0.02


def test_if_scores_reproducible(X_small):
    a = IsolationForestDetector(seed=7)
    a.fit(X_small)
    b = IsolationForestDetector(seed=7)
    b.fit(X_small)
    np.testing.assert_array_equal(a.score(X_small), b.score(X_small))


def test_ae_fits_and_scores(X_small):
    det = AutoencoderDetector(epochs=5, seed=42)
    det.fit(X_small)
    scores = det.score(X_small)
    assert scores.shape == (len(X_small),)
    assert np.issubdtype(scores.dtype, np.floating)


def test_ae_score_before_fit_raises(X_small):
    det = AutoencoderDetector(epochs=5, seed=42)
    with pytest.raises(RuntimeError, match="fit"):
        det.score(X_small)


def test_ae_scores_reproducible(X_small):
    c = AutoencoderDetector(epochs=5, seed=7)
    c.fit(X_small)
    d = AutoencoderDetector(epochs=5, seed=7)
    d.fit(X_small)
    np.testing.assert_array_almost_equal(c.score(X_small), d.score(X_small))


def test_seeded_anomalies_detected_by_if():
    """Las 6 anomalías sembradas deben estar en el top 3% de scores (IsolationForest)."""
    df = load_tickets(RAW_TICKETS_CSV)
    feat = build_hourly_features(df)
    X = feat[FEATURE_COLS].values

    det = IsolationForestDetector(seed=42)
    det.fit(X)
    scores = det.score(X)
    threshold = np.percentile(scores, 97)

    anomalies = json.loads(SEEDED_ANOMALIES_JSON.read_text())["anomalies"]
    for a in anomalies:
        mask = (
            (feat["date"].astype(str) == a["date"])
            & (feat["category"] == a["category"])
            & (feat["hour"] >= a["window_start_hour"])
            & (feat["hour"] < a["window_start_hour"] + a["window_hours"])
        )
        burst_scores = scores[mask]
        assert len(burst_scores) > 0, f"Sin ventanas para {a['date']} {a['category']}"
        assert burst_scores.max() > threshold, (
            f"Anomalía no detectada: {a['date']} {a['category']} "
            f"max_score={burst_scores.max():.4f} threshold={threshold:.4f}"
        )
