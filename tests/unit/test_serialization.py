"""Tests TDD para serialización de modelos a disco."""

from __future__ import annotations

import numpy as np
import pytest

from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.data.synthesizer import SynthConfig, generate
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel


@pytest.fixture(scope="module")
def small_df():
    return generate(SynthConfig(n_tickets=3_000, seed=42)).tickets


@pytest.fixture(scope="module")
def X_small(small_df):
    return build_hourly_features(small_df)[FEATURE_COLS].values


def test_escalation_model_save_load(tmp_path, small_df):
    model = EscalationModel(seed=42, n_estimators=50)
    model.fit(small_df)
    path = tmp_path / "escalation.pkl"
    model.save(path)
    loaded = EscalationModel.load(path)

    np.testing.assert_array_equal(model.predict_proba(small_df), loaded.predict_proba(small_df))
    assert loaded.threshold_ == model.threshold_
    assert loaded.eval_metrics_["auc_roc"] == model.eval_metrics_["auc_roc"]


def test_isolation_forest_save_load(tmp_path, X_small):
    detector = IsolationForestDetector(seed=42)
    detector.fit(X_small)
    path = tmp_path / "if_detector.pkl"
    detector.save(path)
    loaded = IsolationForestDetector.load(path)

    np.testing.assert_array_equal(detector.score(X_small), loaded.score(X_small))


def test_autoencoder_input_dim_stored(X_small):
    detector = AutoencoderDetector(epochs=2, seed=42)
    detector.fit(X_small)
    assert hasattr(detector, "_input_dim")
    assert detector._input_dim == X_small.shape[1]


def test_autoencoder_save_load(tmp_path, X_small):
    detector = AutoencoderDetector(epochs=2, seed=42)
    detector.fit(X_small)
    path = tmp_path / "ae_detector.pkl"
    weights_path = tmp_path / "ae_weights.pt"
    detector.save(path, weights_path)
    loaded = AutoencoderDetector.load(path, weights_path)

    np.testing.assert_allclose(detector.score(X_small), loaded.score(X_small), rtol=1e-5)
