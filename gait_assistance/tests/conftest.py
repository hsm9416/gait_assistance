"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# make the package importable when pytest is run from inside the package
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gait_assistance.config import Config  # noqa: E402
from gait_assistance.gait.phase_detector import create_phase_detector  # noqa: E402
from gait_assistance.gait.stride_segmenter import Stride, StrideSegmenter  # noqa: E402
from gait_assistance.sensors.encoder import MockMotor  # noqa: E402
from gait_assistance.sensors.sensor_manager import (  # noqa: E402
    MockSensorSource,
    SensorManager,
)


@pytest.fixture()
def config() -> Config:
    """Default configuration."""
    return Config()


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator."""
    return np.random.default_rng(1234)


def make_spd(n: int = 5, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Build a random SPD matrix of size ``n``."""
    generator = np.random.default_rng(seed)
    data = generator.normal(size=(200, n)) * scale
    cov = np.cov(data, rowvar=False)
    return 0.5 * (cov + cov.T) + 1e-6 * np.eye(n)


def simulate_strides(
    config: Config, n_seconds: float = 60.0, drift_after_s: float = 0.0
) -> List[Stride]:
    """Run the mock sensor through the segmenter and return the strides."""
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    source = MockSensorSource(config.sensor, motor=motor, drift_after_s=drift_after_s)
    manager = SensorManager(source, config.sensor)
    manager.open()
    detector = create_phase_detector(config.phase)
    segmenter = StrideSegmenter(config.stride)
    strides: List[Stride] = []
    for _ in range(int(n_seconds * config.sensor.sample_rate_hz)):
        polled = manager.poll()
        if polled is None:
            break
        sample, _health = polled
        completed = segmenter.update(sample, detector.update(sample))
        if completed is not None:
            strides.append(completed)
    return strides


@pytest.fixture()
def strides(config: Config) -> List[Stride]:
    """Fifty seconds of synthetic strides."""
    return simulate_strides(config, 60.0)
