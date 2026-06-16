"""Fixtures compartidas. Generan datasets pequeños para tests rápidos."""

from __future__ import annotations

import os

import pytest

# Prevents segfault from OpenMP conflict between PyTorch and LightGBM on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from itops.data.synthesizer import GenerationResult, SynthConfig, generate

# Dataset reducido: mantiene patrones pero corre en <1s.
_TEST_CONFIG = SynthConfig(n_tickets=8_000, seed=123)


@pytest.fixture(scope="session")
def generation() -> GenerationResult:
    return generate(_TEST_CONFIG)


@pytest.fixture(scope="session")
def tickets(generation: GenerationResult):
    return generation.tickets


@pytest.fixture(scope="session")
def test_config() -> SynthConfig:
    return _TEST_CONFIG
