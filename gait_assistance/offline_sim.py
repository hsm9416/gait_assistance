"""Offline simulation: CSV in, assist gains out (spec 33).

A single pass over the recording drives exactly the same code as the live
system: the low-level loop consumes every sample, the segmenter emits strides,
the first ``baseline_strides`` form the patient-specific baseline the model is
built from, and every following stride is classified, scored and turned into an
assist gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import Config, RunMode
from .control.assistance_policy import DeviationScorer
from .gait.feature_extractor import (
    FeatureExtractor,
    aggregate_metrics,
    collect_metric_samples,
    compute_stride_metrics,
)
from .gait.phase_detector import create_phase_detector
from .gait.stride_segmenter import Stride, StrideSegmenter
from .loops import AssistCommand, HighLevelOutput, LowLevelLoop, TwoLoopRuntime
from .manifold.covariance import covariance_matrix
from .manifold.log_euclidean import matrix_log
from .manifold.reference import (
    HealthyReference,
    MetricRange,
    ReferenceBank,
    build_healthy_reference,
    load_reference,
)
from .patient.baseline import BaselineCollector, build_baseline_from_strides
from .patient.patient_model import PatientModel
from .sensors.encoder import MockMotor
from .sensors.sensor_manager import CsvSensorSource, SensorManager, SensorSample
from .utils.logger import StrideLogger


@dataclass
class SimulationResult:
    """Everything one offline run produced."""

    config: Config
    model: Optional[PatientModel]
    outputs: List[HighLevelOutput]
    baseline_stride_ids: List[int]
    table: pd.DataFrame
    n_samples: int
    n_strides: int
    n_cycles: int
    rejected_strides: List[Tuple[int, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    #: reference intervals the biomechanical errors were measured against,
    #: carried so the plots can draw the same bounds the policy used
    metric_ranges: Dict[str, "MetricRange"] = field(default_factory=dict)
    #: where those intervals came from (healthy data set or patient baseline)
    reference_source: str = ""

    @property
    def log_vectors(self) -> np.ndarray:
        """``(n_analysed, d)`` matrix of vectorised log-SPD strides (for PCA)."""
        vectors = [
            o.analysis.log_vector for o in self.outputs if o.analysis.log_vector is not None
        ]
        if not vectors:
            return np.zeros((0, 0), dtype=float)
        return np.stack(vectors, axis=0)

    def summary(self) -> Dict[str, object]:
        """Human-readable summary of the run."""
        info: Dict[str, object] = {
            "samples": self.n_samples,
            "control_cycles": self.n_cycles,
            "strides_detected": self.n_strides,
            "strides_rejected": len(self.rejected_strides),
            "baseline_strides": len(self.baseline_stride_ids),
            "strides_analyzed": len(self.outputs),
        }
        if self.model is not None:
            info.update(self.model.summary())
        if not self.table.empty:
            info["mean_assist_gain"] = float(self.table["assist_gain"].mean())
            info["max_assist_gain"] = float(self.table["assist_gain"].max())
            info["ood_strides"] = int(self.table["ood_flag"].sum())
            info["mean_manifold_deviation"] = float(
                self.table["manifold_deviation"].mean()
            )
            info["mean_biomechanical_deviation"] = float(
                self.table["biomechanical_deviation"].mean()
            )
            info["decision_states"] = {
                str(k): int(v)
                for k, v in self.table["decision_state"].value_counts().items()
            }
            sources = [s for s in self.table["reference_source"].unique() if s]
            if sources:
                info["reference_source"] = str(sources[0])
        if self.errors:
            info["errors"] = self.errors
        return info

    def save_table(self, path: Union[str, Path]) -> Path:
        """Write the stride table to ``path`` as CSV and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out, index=False)
        return out


def segment_csv(
    csv_path: Union[str, Path], config: Config
) -> Tuple[List[Stride], List[SensorSample], List[Tuple[int, str]]]:
    """Segment a CSV recording into strides without running any control.

    Args:
        csv_path: recording to read.
        config: system configuration.

    Returns:
        ``(strides, samples, rejected)``.
    """
    source = CsvSensorSource(str(csv_path), config.sensor)
    manager = SensorManager(source, config.sensor)
    manager.open()
    detector = create_phase_detector(config.phase)
    segmenter = StrideSegmenter(config.stride)
    strides: List[Stride] = []
    samples: List[SensorSample] = []
    for sample, _health in manager.stream():
        samples.append(sample)
        completed = segmenter.update(sample, detector.update(sample))
        if completed is not None:
            strides.append(completed)
    return strides, samples, list(segmenter.rejected)


def build_healthy_reference_from_csv(
    csv_path: Union[str, Path],
    config: Config,
    *,
    max_strides: Optional[int] = None,
) -> HealthyReference:
    """Build a healthy reference from an able-bodied recording (spec 15).

    The reference is normalised with its *own* statistics, which keeps it
    independent of any patient session.

    Both halves of the region are built here: the manifold boundary from the
    distances of the healthy strides to their own centroid, and one reference
    interval per biomechanical metric from the healthy distribution of that
    metric.  A metric no healthy stride carried gets no interval, and the
    policy then reports the corresponding error as undefined.

    Args:
        csv_path: able-bodied recording.
        config: system configuration.
        max_strides: cap on the number of strides used.

    Returns:
        The :class:`~.manifold.reference.HealthyReference`.

    Raises:
        RuntimeError: if the recording yields fewer than two strides.
    """
    strides, _samples, _rejected = segment_csv(csv_path, config)
    if max_strides is not None:
        strides = strides[:max_strides]
    if len(strides) < 2:
        raise RuntimeError(
            f"{csv_path} produced {len(strides)} strides; at least 2 are required"
        )
    extractor = FeatureExtractor(config.feature, config.stride)
    healthy_config = Config.from_dict(config.to_dict())
    healthy_config.patient.baseline_strides = len(strides)
    baseline = build_baseline_from_strides(strides, healthy_config, extractor)
    return build_healthy_reference(
        baseline.log_matrices,
        config=config.reference,
        feature_names=extractor.channels,
        metrics=baseline.mean_metrics,
        metric_samples=collect_metric_samples(baseline.metrics),
        epsilon=config.manifold.epsilon,
    )


class OfflineSimulator:
    """Replay a CSV recording through the complete control stack.

    Args:
        config: system configuration.
        csv_path: patient recording to replay.
        healthy: optional healthy reference; loaded from
            ``config.reference.path`` when omitted and that path is set.
        stride_csv: optional stride-log destination; overrides the config.
    """

    def __init__(
        self,
        config: Config,
        csv_path: Union[str, Path],
        healthy: Optional[Union[HealthyReference, ReferenceBank]] = None,
        *,
        stride_csv: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config
        self.csv_path = Path(csv_path)
        if healthy is None and config.reference.path:
            healthy = load_reference(config.reference.path)
        self.healthy = healthy
        self.stride_csv = stride_csv if stride_csv is not None else config.logging.stride_csv

    def run(self, *, realtime: bool = False) -> SimulationResult:
        """Execute the replay.

        Args:
            realtime: pace the low-level loop at the configured rate instead of
                running as fast as possible.

        Returns:
            The :class:`SimulationResult`.
        """
        config = self.config
        source = CsvSensorSource(str(self.csv_path), config.sensor)
        sensors = SensorManager(source, config.sensor)
        sensors.open()

        motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
        assist = AssistCommand()
        low = LowLevelLoop(config, sensors, motor, assist)
        logger = StrideLogger(self.stride_csv or None)
        runtime = TwoLoopRuntime(
            config,
            low,
            baseline_collector=BaselineCollector(config),
            healthy=self.healthy,
            logger=logger,
        )
        # deterministic offline replay: no worker thread
        runtime.config = Config.from_dict(config.to_dict())
        runtime.config.loop.high_level_in_thread = False

        cycles = runtime.run(realtime=realtime)
        logger.close()

        outputs = runtime.high.outputs if runtime.high is not None else []
        evaluator = runtime.high.policy.evaluator if runtime.high is not None else None
        table = build_table(
            outputs, runtime.baseline_stride_ids, config.assist.max_retraction_mm
        )
        return SimulationResult(
            config=config,
            model=runtime.model,
            outputs=list(outputs),
            baseline_stride_ids=list(runtime.baseline_stride_ids),
            table=table,
            n_samples=len(source),
            n_strides=len(runtime.baseline_stride_ids) + len(outputs),
            n_cycles=cycles,
            rejected_strides=list(low.segmenter.rejected),
            errors=list(runtime.errors),
            metric_ranges=dict(evaluator.metric_ranges) if evaluator else {},
            reference_source=evaluator.reference_source if evaluator else "",
        )


def build_table(
    outputs: Sequence[HighLevelOutput],
    baseline_stride_ids: Sequence[int],
    max_retraction_mm: float = 30.0,
) -> pd.DataFrame:
    """Assemble the per-stride result table.

    Args:
        outputs: high-level outputs of the analysed strides.
        baseline_stride_ids: identifiers of the strides used for the baseline.
        max_retraction_mm: peak retraction, used to report the belt target at
            the peak of the swing profile.

    Returns:
        A dataframe with one row per analysed stride.
    """
    rows: List[Dict[str, object]] = []
    baseline_ids = set(int(i) for i in baseline_stride_ids)
    for output in outputs:
        analysis = output.analysis
        metrics = output.metrics
        assessment = output.assessment
        rows.append(
            {
                "stride_id": output.stride_id,
                "timestamp": analysis.timestamp,
                "is_baseline": int(output.stride_id in baseline_ids),
                "gait_state": analysis.state_label,
                "gait_state_id": analysis.gait_state,
                "confidence": analysis.confidence,
                "nearest_distance": analysis.nearest_distance,
                # d_patient is the baseline deviation (consistency) and
                # d_healthy the pathological one; they are separate columns on
                # purpose and d_healthy stays empty without a healthy reference
                "d_patient": analysis.d_patient,
                "d_healthy": analysis.d_healthy,
                "d_target": analysis.d_target,
                "healthy_region_threshold": analysis.healthy_region_threshold,
                "inside_healthy_region": (
                    None
                    if analysis.inside_healthy_region is None
                    else int(analysis.inside_healthy_region)
                ),
                "manifold_deviation": analysis.manifold_deviation,
                "delta_healthy": (
                    float("nan") if assessment is None else assessment.delta_healthy
                ),
                "stride_time": metrics.stride_time,
                "swing_time": metrics.swing_time,
                "stance_time": metrics.stance_time,
                "swing_ratio": metrics.swing_ratio,
                "stance_ratio": metrics.stance_ratio,
                "swing_stance_ratio": metrics.swing_stance_ratio,
                "phase_source": metrics.phase_source,
                "swing_symmetry_ratio": metrics.swing_symmetry_ratio,
                "stance_symmetry_ratio": metrics.stance_symmetry_ratio,
                "swing_stance_symmetry_ratio": metrics.swing_stance_symmetry_ratio,
                "belt_excursion": metrics.belt_excursion,
                "peak_belt_velocity": metrics.peak_belt_velocity,
                "trunk_rms": metrics.trunk_acc_rms,
                "trunk_acc_rms": metrics.trunk_acc_rms,
                "trunk_gyro_rms": metrics.trunk_gyro_rms,
                "swing_ratio_error": (
                    None if assessment is None else assessment.swing_ratio_error
                ),
                "belt_excursion_error": (
                    None if assessment is None else assessment.belt_excursion_error
                ),
                "temporal_symmetry_error": (
                    None if assessment is None else assessment.temporal_symmetry_error
                ),
                "trunk_error": None if assessment is None else assessment.trunk_error,
                "biomechanical_deviation": (
                    float("nan")
                    if assessment is None
                    else assessment.biomechanical_deviation
                ),
                "reference_source": (
                    "" if assessment is None else assessment.reference_source
                ),
                "decision_state": (
                    "" if assessment is None else assessment.decision_state
                ),
                "assist_score": (
                    float("nan") if assessment is None else assessment.assist_score
                ),
                "raw_assist_gain": (
                    float("nan") if assessment is None else assessment.raw_assist_gain
                ),
                # legacy weighted-sum components, kept for comparison only
                "e_manifold": output.components.e_manifold,
                "e_baseline": output.components.e_baseline,
                "e_healthy": output.components.e_healthy,
                "assist_mode": output.components.mode.value,
                "e_swing": output.components.e_swing,
                "e_excursion": output.components.e_excursion,
                "deviation_score": output.components.score,
                "assist_gain": output.assist_gain,
                "gain_delta": output.gain_delta,
                "target_belt_length": output.baseline_belt_length
                - output.assist_gain * max_retraction_mm,
                "ood_flag": int(analysis.is_ood),
                "ood_engaged": (
                    0 if assessment is None else int(assessment.ood_engaged)
                ),
                "valid": int(analysis.valid),
            }
        )
    return pd.DataFrame(rows)


def run_offline(
    csv_path: Union[str, Path],
    config: Optional[Config] = None,
    *,
    healthy_csv: Optional[Union[str, Path]] = None,
    stride_csv: Optional[Union[str, Path]] = None,
) -> SimulationResult:
    """Convenience entry point used by the CLI and the tests.

    Args:
        csv_path: patient recording.
        config: system configuration; defaults are used when omitted.
        healthy_csv: able-bodied recording used to build the healthy reference.
        stride_csv: stride-log destination.

    Returns:
        The :class:`SimulationResult`.
    """
    cfg = config or Config()
    cfg.mode = RunMode.OFFLINE
    healthy = (
        build_healthy_reference_from_csv(healthy_csv, cfg) if healthy_csv else None
    )
    return OfflineSimulator(cfg, csv_path, healthy, stride_csv=stride_csv).run()


__all__ = [
    "OfflineSimulator",
    "SimulationResult",
    "build_healthy_reference_from_csv",
    "build_table",
    "run_offline",
    "segment_csv",
]
