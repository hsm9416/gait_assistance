"""Two-loop runtime architecture (spec 24).

``LowLevelLoop`` runs at 100-1000 Hz and only ever touches cheap arithmetic:
acquisition, safety, gait phase, belt-target interpolation and the impedance
law.  ``HighLevelLoop`` runs once per stride (0.5-2 Hz) and owns the expensive
manifold chain.  The only data crossing from the slow loop to the fast one is
``(assist_gain, baseline_belt_length)``, published under a lock, so a slow
stride analysis can never stall the current command.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .control.assistance_policy import (
    AssistanceAssessment,
    AssistanceGainPolicy,
    AssistancePolicy,
    DecisionState,
    DeviationComponents,
    DeviationScorer,
)
from .control.impedance_controller import ImpedanceController, MotorCommand
from .control.safety import SafetyMonitor, SafetyStatus
from .control.target_generator import BeltTarget, BeltTargetGenerator
from .gait.feature_extractor import GaitMetrics, StrideMetrics, compute_stride_metrics
from .gait.phase_detector import GaitPhase, PhaseDetector, PhaseResult, create_phase_detector
from .gait.stride_segmenter import Stride, StrideSegmenter
from .manifold.reference import HealthyReference
from .patient.baseline import BaselineCollector
from .patient.patient_model import PatientModel, StrideAnalysis
from .sensors.encoder import MotorInterface
from .sensors.sensor_manager import SensorManager, SensorSample
from .state_machine import StateMachine, SystemState
from .utils.logger import StrideLogger, StrideLogRecord
from .utils.timing import LoopRate, RateMonitor, now


# --------------------------------------------------------------------------- #
# Shared set-point
# --------------------------------------------------------------------------- #


class AssistCommand:
    """Thread-safe hand-over between the high-level and low-level loops."""

    def __init__(self, baseline_belt_length: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._gain = 0.0
        self._baseline_belt_length = baseline_belt_length
        self._stride_id = -1

    def publish(self, gain: float, baseline_belt_length: float, stride_id: int) -> None:
        """Store a new set-point (called by the high-level loop)."""
        with self._lock:
            self._gain = float(gain)
            self._baseline_belt_length = float(baseline_belt_length)
            self._stride_id = int(stride_id)

    def read(self) -> Tuple[float, float, int]:
        """Return ``(gain, baseline_belt_length, stride_id)``."""
        with self._lock:
            return self._gain, self._baseline_belt_length, self._stride_id

    def zero(self) -> None:
        """Force the gain to zero (safety path)."""
        with self._lock:
            self._gain = 0.0


# --------------------------------------------------------------------------- #
# High level
# --------------------------------------------------------------------------- #


@dataclass
class HighLevelOutput:
    """Everything one stride analysis produced."""

    stride_id: int
    analysis: StrideAnalysis
    metrics: GaitMetrics
    components: DeviationComponents
    assist_gain: float
    previous_gain: float
    baseline_belt_length: float
    #: full decision record; the gain published to the fast loop comes from
    #: :attr:`AssistanceAssessment.limited_assist_gain`
    assessment: Optional[AssistanceAssessment] = None

    @property
    def gain_delta(self) -> float:
        """Absolute gain change with respect to the previous stride."""
        return abs(self.assist_gain - self.previous_gain)

    @property
    def decision_state(self) -> str:
        """Decision state of the stride, or an empty string when unassessed."""
        return "" if self.assessment is None else self.assessment.decision_state


class HighLevelLoop:
    """Stride-rate loop: features, manifold, deviation and assist gain.

    The gain comes from :class:`AssistancePolicy`, which requires a
    device-actionable biomechanical deficit before any assistance is produced
    and uses the manifold deviation only as a severity factor on top of it.
    The legacy :class:`DeviationScorer` is still constructed so its weighted-sum
    components stay in the record for comparison, but it no longer decides the
    gain.

    Args:
        config: full configuration.
        model: fitted patient model.
        scorer: legacy deviation scorer; derived from ``model`` when omitted.
        gain_policy: assist-gain policy; created from config when omitted.
        logger: stride logger; an in-memory logger is used when omitted.
        policy: assistance policy; derived from ``model`` when omitted.
    """

    def __init__(
        self,
        config: Config,
        model: PatientModel,
        scorer: Optional[DeviationScorer] = None,
        gain_policy: Optional[AssistanceGainPolicy] = None,
        logger: Optional[StrideLogger] = None,
        policy: Optional[AssistancePolicy] = None,
        walking_speed: Optional[float] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.scorer = scorer or DeviationScorer.from_model(model, config.deviation)
        self.policy = policy or AssistancePolicy.from_model(config, model)
        if gain_policy is not None:
            self.policy.gain_policy = gain_policy
        self.gain_policy = self.policy.gain_policy
        self.logger = logger or StrideLogger(config.logging.stride_csv or None)
        self.baseline_belt_length = model.baseline.baseline_belt_length
        self.walking_speed = walking_speed
        self.outputs: List[HighLevelOutput] = []
        self.rate = RateMonitor()

    def process_stride(
        self, stride: Stride, *, assist_allowed: bool = True
    ) -> HighLevelOutput:
        """Run the whole high-level chain on one stride.

        Args:
            stride: completed stride from the segmenter.
            assist_allowed: safety veto forwarded from the low-level loop.

        Returns:
            The :class:`HighLevelOutput`, also appended to :attr:`outputs`.
        """
        self.rate.tick()
        analysis = self.model.analyze_stride(stride)
        metrics = analysis.metrics or compute_stride_metrics(stride)
        components = self.scorer.score(analysis, metrics)
        previous = self.policy.gain
        assessment = self.policy.assess(
            analysis,
            metrics,
            assist_allowed=assist_allowed and analysis.valid,
            walking_speed=self.walking_speed,
        )
        gain = assessment.limited_assist_gain
        output = HighLevelOutput(
            stride_id=stride.stride_id,
            analysis=analysis,
            metrics=metrics,
            components=components,
            assist_gain=gain,
            previous_gain=previous,
            baseline_belt_length=self.baseline_belt_length,
            assessment=assessment,
        )
        self.outputs.append(output)
        self.logger.write(self.make_record(output, stride))
        return output

    def make_record(self, output: HighLevelOutput, stride: Stride) -> StrideLogRecord:
        """Build the CSV record of one stride (spec 19/27).

        Nothing here substitutes a value: a metric the run could not measure
        reaches the record as ``None`` and the writer blanks the cell.
        """
        last = stride.samples[-1] if stride.samples else None
        target_belt = (
            output.baseline_belt_length
            - output.assist_gain * self.config.assist.max_retraction_mm
        )
        analysis = output.analysis
        metrics = output.metrics
        assessment = output.assessment
        return StrideLogRecord(
            stride_id=output.stride_id,
            timestamp=analysis.timestamp,

            gait_cluster=analysis.state_label,
            cluster_confidence=round(analysis.confidence, 6),
            ood=int(analysis.is_ood),

            d_patient=analysis.d_patient,
            d_healthy=analysis.d_healthy,
            d_target=analysis.d_target,
            healthy_region_threshold=analysis.healthy_region_threshold,
            manifold_deviation=analysis.manifold_deviation,
            delta_healthy=(
                float("nan") if assessment is None else assessment.delta_healthy
            ),

            estimated_swing_time=metrics.swing_time,
            estimated_stance_time=metrics.stance_time,
            estimated_stride_time=metrics.stride_time,
            estimated_swing_ratio=metrics.swing_ratio,
            estimated_stance_ratio=metrics.stance_ratio,
            estimated_swing_stance_ratio=metrics.swing_stance_ratio,
            phase_source=metrics.phase_source,

            swing_symmetry_ratio=metrics.swing_symmetry_ratio,
            stance_symmetry_ratio=metrics.stance_symmetry_ratio,
            swing_stance_symmetry_ratio=metrics.swing_stance_symmetry_ratio,

            belt_excursion=metrics.belt_excursion,
            peak_belt_velocity=metrics.peak_belt_velocity,
            trunk_acc_rms=metrics.trunk_acc_rms,
            trunk_gyro_rms=metrics.trunk_gyro_rms,

            swing_ratio_error=None if assessment is None else assessment.swing_ratio_error,
            belt_excursion_error=(
                None if assessment is None else assessment.belt_excursion_error
            ),
            temporal_symmetry_error=(
                None if assessment is None else assessment.temporal_symmetry_error
            ),
            trunk_error=None if assessment is None else assessment.trunk_error,

            biomechanical_deviation=(
                float("nan") if assessment is None else assessment.biomechanical_deviation
            ),
            reference_source="" if assessment is None else assessment.reference_source,
            decision_state="" if assessment is None else assessment.decision_state,

            raw_assist_gain=(
                float("nan") if assessment is None else assessment.raw_assist_gain
            ),
            assist_gain=output.assist_gain,
            target_belt_length=target_belt,
            motor_position=last.motor_position if last else float("nan"),
            motor_velocity=last.motor_velocity if last else float("nan"),
            motor_current=last.motor_current if last else float("nan"),
        )


# --------------------------------------------------------------------------- #
# Low level
# --------------------------------------------------------------------------- #


@dataclass
class LowLevelOutput:
    """State of one low-level control cycle."""

    sample: SensorSample
    phase: PhaseResult
    safety: SafetyStatus
    target: BeltTarget
    command: MotorCommand
    completed_stride: Optional[Stride] = None


class LowLevelLoop:
    """Sample-rate loop: acquisition, safety, phase, target and current.

    Args:
        config: full configuration.
        sensors: sensor manager providing frames.
        motor: actuator receiving the current commands.
        assist: shared set-point published by the high-level loop.
        phase_detector: detector instance; created from config when omitted.
        segmenter: stride segmenter; created from config when omitted.
        safety: safety monitor; created from config when omitted.
        impedance: impedance controller; created from config when omitted.
        target_generator: belt-target generator; created when omitted.
    """

    def __init__(
        self,
        config: Config,
        sensors: SensorManager,
        motor: Optional[MotorInterface],
        assist: AssistCommand,
        *,
        phase_detector: Optional[PhaseDetector] = None,
        segmenter: Optional[StrideSegmenter] = None,
        safety: Optional[SafetyMonitor] = None,
        impedance: Optional[ImpedanceController] = None,
        target_generator: Optional[BeltTargetGenerator] = None,
    ) -> None:
        self.config = config
        self.sensors = sensors
        self.motor = motor
        self.assist = assist
        self.phase_detector = phase_detector or create_phase_detector(config.phase)
        self.segmenter = segmenter or StrideSegmenter(config.stride)
        self.safety = safety or SafetyMonitor(config.safety)
        self.impedance = impedance or ImpedanceController(config.impedance)
        self.target_generator = target_generator or BeltTargetGenerator(config.assist)
        self.rate = RateMonitor()
        self.last_output: Optional[LowLevelOutput] = None
        self.cycles = 0

    def step(self) -> Optional[LowLevelOutput]:
        """Execute one control cycle.

        Returns:
            The :class:`LowLevelOutput`, or ``None`` when no new sensor frame
            was available (or the source is exhausted).
        """
        polled = self.sensors.poll()
        if polled is None:
            return None
        sample, _health = polled
        return self.step_with_sample(sample)

    def step_with_sample(self, sample: SensorSample) -> LowLevelOutput:
        """Run one control cycle on an externally supplied sample.

        Args:
            sample: sensor frame to process.

        Returns:
            The :class:`LowLevelOutput` of this cycle.
        """
        self.cycles += 1
        self.rate.tick()

        safety = self.safety.check_sample(sample)
        phase = self.phase_detector.update(sample)
        completed = self.segmenter.update(sample, phase)

        gain, baseline_belt, stride_id = self.assist.read()
        if stride_id < 0:
            # No model yet, so no neutral length is known.  Hold the measured
            # position instead of pulling the belt towards the 0 mm default:
            # during baseline collection that default is tens of millimetres
            # away from where the belt actually sits on the wearer.
            baseline_belt = sample.belt_length
        if not safety.assist_allowed:
            gain = 0.0

        target = self.target_generator.generate(
            baseline_belt_length=baseline_belt,
            assist_gain=gain,
            phase=phase.phase,
            phase_progress=phase.phase_progress,
            timestamp=sample.timestamp,
        )

        if safety.assist_allowed:
            command = self.impedance.compute(
                target.target_belt_length,
                sample.belt_length,
                target_velocity=0.0,
                measured_velocity=sample.belt_velocity,
            )
        else:
            command = self.impedance.zero_command()

        current = self.safety.limit_current(
            command.current_a, self.config.impedance.current_limit_a
        )
        if self.motor is not None:
            self.motor.send_current(current)

        output = LowLevelOutput(sample, phase, safety, target, command, completed)
        self.last_output = output
        return output

    def stop(self) -> None:
        """Command zero current to the motor."""
        if self.motor is not None:
            self.motor.stop()




# --------------------------------------------------------------------------- #
# Runtime binding the two loops
# --------------------------------------------------------------------------- #


class TwoLoopRuntime:
    """Drive both loops and manage the session state machine (spec 24, 26).

    The runtime covers the whole session: it collects the patient baseline,
    builds the model when enough strides are available and only then starts
    analysing strides and publishing assist gains.  The high-level work can run
    on its own thread so that the fast loop is never blocked by it.

    Args:
        config: full configuration.
        low_level: the fast loop.
        high_level: an already built stride-rate loop; when omitted the runtime
            starts in ``BASELINE_COLLECTION`` and builds one itself.
        state_machine: shared state machine; created when omitted.
        baseline_collector: collector to fill during the baseline phase.
        healthy: optional healthy reference used when building the model.
        logger: stride logger handed to the high-level loop it builds.
        on_model_built: callback invoked with the fitted model.
    """

    def __init__(
        self,
        config: Config,
        low_level: LowLevelLoop,
        high_level: Optional[HighLevelLoop] = None,
        state_machine: Optional[StateMachine] = None,
        *,
        baseline_collector: Optional[BaselineCollector] = None,
        healthy: Optional[HealthyReference] = None,
        logger: Optional[StrideLogger] = None,
        on_model_built: Optional[Callable[[PatientModel], None]] = None,
    ) -> None:
        self.config = config
        self.low = low_level
        self.high = high_level
        self.healthy = healthy
        self.logger = logger
        self.on_model_built = on_model_built
        self.baseline = baseline_collector or BaselineCollector(config)
        initial = SystemState.READY if high_level is not None else SystemState.INIT
        self.states = state_machine or StateMachine(initial)
        self._queue: "queue.Queue[Optional[Stride]]" = queue.Queue(
            maxsize=config.loop.high_level_queue_size
        )
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.dropped_strides = 0
        self.baseline_stride_ids: List[int] = []
        self.errors: List[str] = []

    @property
    def model(self) -> Optional[PatientModel]:
        """The fitted patient model, once the baseline has been processed."""
        return self.high.model if self.high is not None else None

    # -- session progression ------------------------------------------------ #

    def _handle_stride(self, stride: Stride) -> Optional[HighLevelOutput]:
        """Route one completed stride to the baseline or the analysis path."""
        if self.high is None:
            self._collect_baseline(stride)
            return None
        allowed = self.low.safety.status.assist_allowed
        output = self.high.process_stride(stride, assist_allowed=allowed)
        if not output.analysis.valid:
            self.low.safety.report_invalid_spd(output.analysis.message)
        self._publish(output)
        return output

    def _collect_baseline(self, stride: Stride) -> None:
        """Add a stride to the baseline and build the model when complete."""
        if self.states.state in (SystemState.INIT, SystemState.CALIBRATION):
            if self.states.state is SystemState.INIT:
                self.states.transition_to(SystemState.CALIBRATION)
            self.states.transition_to(SystemState.BASELINE_COLLECTION)
        self.baseline.add(stride)
        self.baseline_stride_ids.append(stride.stride_id)
        if not self.baseline.is_ready:
            return
        self.states.transition_to(SystemState.MODEL_BUILDING)
        try:
            result = self.baseline.build()
            model = PatientModel.build(
                self.config, result, self.healthy, self.baseline.extractor
            )
        except (RuntimeError, ValueError) as exc:
            self.errors.append(f"model building failed: {exc}")
            self.states.safe_stop()
            return
        self.high = HighLevelLoop(self.config, model, logger=self.logger)
        self.low.assist.publish(0.0, model.baseline.baseline_belt_length, stride.stride_id)
        self.states.transition_to(SystemState.READY)
        if self.on_model_built is not None:
            self.on_model_built(model)

    def _publish(self, output: HighLevelOutput) -> None:
        """Publish the new set-point and update the state machine."""
        self.low.assist.publish(
            output.assist_gain, output.baseline_belt_length, output.stride_id
        )
        if output.analysis.is_ood:
            if self.states.can_transition(SystemState.OOD):
                self.states.transition_to(SystemState.OOD)
        elif self.states.state in (SystemState.OOD, SystemState.READY):
            self.states.transition_to(SystemState.WALKING)

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Start the high-level worker thread when threading is enabled."""
        self._running = True
        if self.config.loop.high_level_in_thread and self._thread is None:
            self._thread = threading.Thread(
                target=self._worker, name="high-level-loop", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Drain the queue, stop the worker thread and de-energise the motor."""
        self._running = False
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None
        self.low.stop()

    def _worker(self) -> None:
        """Consume completed strides from the queue until stopped."""
        while True:
            stride = self._queue.get()
            try:
                if stride is None:
                    return
                self._handle_stride(stride)
            except Exception as exc:  # pragma: no cover - worker must not die
                self.errors.append(f"high-level loop error: {exc}")
            finally:
                self._queue.task_done()

    def dispatch(self, stride: Stride) -> None:
        """Hand a completed stride to the high-level loop.

        The fast loop never blocks: when the queue is full the stride is
        dropped and counted in :attr:`dropped_strides`.
        """
        if self._thread is None:
            self._handle_stride(stride)
            return
        try:
            self._queue.put_nowait(stride)
        except queue.Full:
            self.dropped_strides += 1

    def run(
        self,
        duration_s: Optional[float] = None,
        *,
        max_cycles: Optional[int] = None,
        on_cycle: Optional[Callable[[LowLevelOutput], None]] = None,
        realtime: bool = True,
    ) -> int:
        """Run the low-level loop, dispatching strides as they complete.

        Args:
            duration_s: wall-clock run time; ``None`` runs until the source is
                exhausted or ``max_cycles`` is reached.
            max_cycles: maximum number of control cycles.
            on_cycle: callback invoked with every :class:`LowLevelOutput`.
            realtime: pace the loop at ``config.loop.low_level_hz``.

        Returns:
            The number of executed control cycles.
        """
        self.start()
        pacer = LoopRate(self.config.loop.low_level_hz, sleep=realtime)
        t0 = now()
        cycles = 0
        try:
            while self._running:
                if duration_s is not None and now() - t0 >= duration_s:
                    break
                if max_cycles is not None and cycles >= max_cycles:
                    break
                output = self.low.step()
                if output is None:
                    if self.low.sensors.finished:
                        break
                    if realtime:
                        pacer.sleep()
                    continue
                cycles += 1
                if output.completed_stride is not None:
                    self.dispatch(output.completed_stride)
                if output.safety.faults:
                    self.low.assist.zero()
                    self.states.safe_stop()
                if on_cycle is not None:
                    on_cycle(output)
                if realtime:
                    pacer.sleep()
        finally:
            if self._thread is not None:
                self._queue.join()
            self.stop()
        return cycles


__all__ = [
    "AssistCommand",
    "HighLevelLoop",
    "HighLevelOutput",
    "LowLevelLoop",
    "LowLevelOutput",
    "TwoLoopRuntime",
]
