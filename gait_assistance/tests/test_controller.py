"""Tests for the control layer, safety and the two-loop runtime (spec 30)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gait_assistance.config import (
    AssistConfig,
    AssistMode,
    Config,
    DeviationConfig,
    ImpedanceConfig,
    SafetyConfig,
)
from gait_assistance.control.assistance_policy import (
    AssistanceGainPolicy,
    DeviationScorer,
)
from gait_assistance.control.impedance_controller import (
    ImpedanceController,
    default_force_to_current,
)
from gait_assistance.control.safety import FaultCode, SafetyMonitor
from gait_assistance.control.target_generator import (
    BeltTargetGenerator,
    SwingAssistanceProfile,
    sine_profile,
)
from gait_assistance.gait.feature_extractor import GaitMetrics, StrideMetrics
from gait_assistance.gait.phase_detector import GaitPhase
from gait_assistance.loops import AssistCommand, HighLevelLoop, LowLevelLoop, TwoLoopRuntime
from gait_assistance.patient.baseline import BaselineCollector, build_baseline_from_strides
from gait_assistance.patient.patient_model import PatientModel, StrideAnalysis
from gait_assistance.sensors.encoder import MockMotor
from gait_assistance.sensors.sensor_manager import (
    MockSensorSource,
    SensorManager,
    SensorSample,
)
from gait_assistance.state_machine import (
    InvalidTransitionError,
    StateMachine,
    SystemState,
)
from gait_assistance.utils.filters import RateLimiter
from gait_assistance.utils.logger import STRIDE_LOG_COLUMNS, StrideLogger

from gait_assistance.manifold.log_euclidean import matrix_log

from .conftest import make_spd, simulate_strides


def _analysis(
    d_target: float = 1.0,
    *,
    valid: bool = True,
    ood: bool = False,
    d_patient: float | None = None,
    d_healthy: float | None = None,
) -> StrideAnalysis:
    """Build a synthetic stride analysis for the scorer tests.

    ``d_patient`` and ``d_healthy`` default to ``d_target`` so the existing
    tests are unaffected; pass them to separate the three distances.
    """
    return StrideAnalysis(
        stride_id=0,
        timestamp=0.0,
        valid=valid,
        gait_state=0,
        state_label="S0",
        confidence=0.8,
        nearest_distance=d_target,
        runner_up_distance=2.0 * d_target,
        d_patient=d_target if d_patient is None else d_patient,
        d_healthy=d_target if d_healthy is None else d_healthy,
        d_target=d_target,
        is_ood=ood,
    )


def _metrics(
    swing_ratio: float = 0.38,
    excursion: float = 100.0,
    *,
    symmetry: float | None = None,
    trunk: float | None = 0.2,
) -> GaitMetrics:
    """Build stride metrics with the interesting fields set.

    The derived ratios are kept mutually consistent so the metric identities
    (``swing_ratio + stance_ratio == 1``) hold on the synthetic data too.
    """
    stance_ratio = 1.0 - swing_ratio
    return GaitMetrics(
        stride_time=1.2,
        swing_time=1.2 * swing_ratio,
        stance_time=1.2 * stance_ratio,
        swing_ratio=swing_ratio,
        stance_ratio=stance_ratio,
        swing_stance_ratio=(swing_ratio / stance_ratio) if stance_ratio else 0.0,
        belt_excursion=excursion,
        peak_belt_velocity=400.0,
        trunk_acc_rms=trunk,
        mean_belt_length=-130.0,
        swing_symmetry_ratio=symmetry,
    )


# --------------------------------------------------------------------------- #
# Assist gain
# --------------------------------------------------------------------------- #


def test_gain_never_exceeds_max_gain() -> None:
    """``assist_gain <= max_gain`` for every input, including invalid ones."""
    policy = AssistanceGainPolicy(AssistConfig(max_gain=0.8, max_gain_delta=0.1))
    for score in [0.0, 0.5, 1.0, 5.0, -3.0, float("nan"), float("inf")]:
        for _ in range(30):
            gain = policy.update(score)
            assert 0.0 <= gain <= 0.8 + 1e-12


def test_gain_step_never_exceeds_max_gain_delta() -> None:
    """Successive strides differ by at most ``max_gain_delta`` (spec 20)."""
    policy = AssistanceGainPolicy(AssistConfig(max_gain=0.8, max_gain_delta=0.1))
    previous = 0.0
    rng = np.random.default_rng(0)
    for score in rng.uniform(0.0, 1.0, 200):
        gain = policy.update(float(score))
        assert abs(gain - previous) <= 0.1 + 1e-12
        previous = gain


def test_gain_ramps_up_and_saturates() -> None:
    """A saturated score climbs one delta per stride up to ``max_gain``."""
    policy = AssistanceGainPolicy(AssistConfig(max_gain=0.8, max_gain_delta=0.1))
    gains = [policy.update(1.0) for _ in range(12)]
    assert gains[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert gains[-1] == pytest.approx(0.8)


def test_ood_caps_the_gain() -> None:
    """No aggressive assistance while out of distribution (spec 14, 25)."""
    policy = AssistanceGainPolicy(AssistConfig(max_gain=0.8, max_gain_delta=0.1, ood_max_gain=0.2))
    for _ in range(12):
        policy.update(1.0)
    assert policy.gain == pytest.approx(0.8)
    for _ in range(12):
        gain = policy.update(1.0, is_ood=True)
    assert gain == pytest.approx(0.2)


def test_safety_veto_and_emergency_zero() -> None:
    """A safety veto ramps the gain down; the emergency path zeroes it at once."""
    policy = AssistanceGainPolicy(AssistConfig(max_gain=0.8, max_gain_delta=0.1))
    for _ in range(12):
        policy.update(1.0)
    assert policy.update(1.0, assist_allowed=False) == pytest.approx(0.7)
    assert policy.emergency_zero() == 0.0


def test_rate_limiter_matches_the_gain_policy() -> None:
    """The shared rate limiter enforces the same bound."""
    limiter = RateLimiter(0.1)
    assert limiter.update(1.0) == pytest.approx(0.1)
    assert limiter.update(-1.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Deviation score
# --------------------------------------------------------------------------- #


def test_deviation_score_is_clipped_to_unit_interval() -> None:
    """The combined score never leaves ``[0, 1]`` (spec 19)."""
    scorer = DeviationScorer(manifold_scale=1.0, target_swing_ratio=0.38, reference_excursion_mm=100.0)
    for distance in [0.0, 0.5, 1.0, 100.0]:
        for ratio in [0.0, 0.38, 1.0]:
            for excursion in [0.0, 50.0, 400.0]:
                components = scorer.score(_analysis(distance), _metrics(ratio, excursion))
                assert 0.0 <= components.score <= 1.0
                assert 0.0 <= components.e_manifold <= 1.0
                assert 0.0 <= components.e_swing <= 1.0
                assert 0.0 <= components.e_excursion <= 1.0


def test_score_combines_manifold_and_biomechanics() -> None:
    """The score is the weighted sum of the three error terms."""
    scorer = DeviationScorer(manifold_scale=2.0, target_swing_ratio=0.38, reference_excursion_mm=100.0)
    components = scorer.score(_analysis(1.0), _metrics(0.38, 100.0))
    assert components.e_manifold == pytest.approx(0.5)
    assert components.e_swing == pytest.approx(0.0)
    assert components.e_excursion == pytest.approx(0.0)
    assert components.score == pytest.approx(0.5 * 0.5)


def test_excursion_error_is_a_deficit_only() -> None:
    """A larger-than-normal excursion is not a deviation to be assisted."""
    scorer = DeviationScorer(manifold_scale=1.0, target_swing_ratio=0.38, reference_excursion_mm=100.0)
    assert scorer.score(_analysis(0.0), _metrics(0.38, 200.0)).e_excursion == 0.0
    assert scorer.score(_analysis(0.0), _metrics(0.38, 50.0)).e_excursion > 0.0


def test_invalid_stride_scores_zero() -> None:
    """An unusable stride produces no assistance at all."""
    scorer = DeviationScorer(manifold_scale=1.0)
    assert scorer.score(_analysis(10.0, valid=False)).score == 0.0


# --------------------------------------------------------------------------- #
# Profile and belt target
# --------------------------------------------------------------------------- #


def test_swing_profile_shape() -> None:
    """The default profile is ``sin(pi x)``: zero at the ends, one in the middle."""
    profile = SwingAssistanceProfile("sine")
    assert profile(0.0) == pytest.approx(0.0)
    assert profile(0.5) == pytest.approx(1.0)
    assert profile(1.0) == pytest.approx(0.0, abs=1e-12)
    assert np.all(profile.sample(100) >= 0.0)
    assert np.allclose(profile.sample(50), sine_profile(np.linspace(0, 1, 50)))
    assert profile(-1.0) == pytest.approx(0.0)
    assert profile(2.0) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        SwingAssistanceProfile("nope")


def test_profile_and_magnitude_are_separable() -> None:
    """Doubling the gain doubles the retraction; the shape is unchanged."""
    generator = BeltTargetGenerator(AssistConfig(max_retraction_mm=30.0))
    low = generator.generate(-130.0, 0.2, GaitPhase.SWING, 0.5)
    high = generator.generate(-130.0, 0.4, GaitPhase.SWING, 0.5)
    assert low.profile_value == pytest.approx(high.profile_value)
    assert (-130.0 - low.target_belt_length) * 2.0 == pytest.approx(
        -130.0 - high.target_belt_length
    )


def test_belt_target_formula() -> None:
    """``target = baseline - gain * profile * max_retraction`` (spec 22)."""
    generator = BeltTargetGenerator(AssistConfig(max_retraction_mm=30.0, profile="sine"))
    target = generator.generate(-130.0, 0.5, GaitPhase.SWING, 0.5)
    assert target.target_belt_length == pytest.approx(-130.0 - 0.5 * 1.0 * 30.0)


def test_no_retraction_outside_swing() -> None:
    """Stance and unknown progress leave the belt at its baseline length."""
    generator = BeltTargetGenerator(AssistConfig(max_retraction_mm=30.0))
    assert generator.generate(-130.0, 0.8, GaitPhase.STANCE, 0.5).target_belt_length == pytest.approx(-130.0)
    assert generator.generate(-130.0, 0.8, GaitPhase.SWING, None).target_belt_length == pytest.approx(-130.0)


# --------------------------------------------------------------------------- #
# Impedance controller and motor
# --------------------------------------------------------------------------- #


def test_impedance_law() -> None:
    """The force follows ``K * position_error + D * velocity_error``."""
    config = ImpedanceConfig(
        stiffness_n_per_mm=0.01, damping_n_per_mms=0.002, use_pad_force_map=False,
        current_per_newton=2.0, current_limit_a=100.0,
    )
    controller = ImpedanceController(config)
    command = controller.compute(-100.0, -150.0, measured_velocity=-10.0)
    assert command.position_error_mm == pytest.approx(50.0)
    assert command.velocity_error_mms == pytest.approx(10.0)
    assert command.force_n == pytest.approx(0.01 * 50.0 + 0.002 * 10.0)
    assert command.current_a == pytest.approx(command.force_n * 2.0)


def test_motor_saturation() -> None:
    """The commanded current is clamped to the configured limit."""
    controller = ImpedanceController(
        ImpedanceConfig(stiffness_n_per_mm=10.0, use_pad_force_map=False,
                        current_per_newton=1.0, current_limit_a=1.0)
    )
    command = controller.compute(1000.0, -1000.0)
    assert command.current_a == pytest.approx(1.0)
    assert command.saturated

    negative = controller.compute(-1000.0, 1000.0)
    assert negative.current_a == pytest.approx(-1.0)
    assert negative.saturated


def test_mock_motor_saturates_and_ignores_nan() -> None:
    """The simulated motor clamps its command and never stores a NaN."""
    motor = MockMotor(current_limit_a=1.0)
    motor.send_current(5.0)
    assert motor.last_command_a == pytest.approx(1.0) and motor.saturated
    motor.send_current(-5.0)
    assert motor.last_command_a == pytest.approx(-1.0)
    motor.send_current(float("nan"))
    assert motor.last_command_a == 0.0
    motor.stop()
    assert motor.last_command_a == 0.0


def test_nan_safety_in_the_controller() -> None:
    """Non-finite set-points or measurements produce a zero command (spec 25)."""
    controller = ImpedanceController(ImpedanceConfig())
    for args in [
        (float("nan"), -100.0),
        (-100.0, float("nan")),
        (float("inf"), -100.0),
    ]:
        command = controller.compute(*args)
        assert command.current_a == 0.0
        assert command.is_finite()
    assert controller.force_to_current(float("nan")) == 0.0


def test_default_force_map_matches_the_device() -> None:
    """The fallback map reproduces the PAD conversion."""
    assert default_force_to_current(-1.0) == 0.0
    assert default_force_to_current(1.0) == pytest.approx((1.0 + 0.04) / 0.87)


# --------------------------------------------------------------------------- #
# Safety monitor
# --------------------------------------------------------------------------- #


def test_safety_flags_nan_data() -> None:
    """A NaN anywhere in the frame inhibits assistance."""
    monitor = SafetyMonitor(SafetyConfig())
    status = monitor.check_sample(SensorSample.from_mapping({"accel_x": float("nan")}))
    assert not status.ok and not status.assist_allowed
    assert FaultCode.NAN_DATA in status.faults


@pytest.mark.parametrize(
    "channel,value,fault",
    [
        ("belt_length", 5000.0, FaultCode.ENCODER_ERROR),
        ("belt_velocity", 9000.0, FaultCode.ENCODER_ERROR),
        ("motor_position", 1e9, FaultCode.MOTOR_POSITION_LIMIT),
        ("motor_current", 50.0, FaultCode.MOTOR_CURRENT_LIMIT),
        ("gyro_x", 5000.0, FaultCode.IMU_ERROR),
        ("accel_z", 100.0, FaultCode.IMU_ERROR),
    ],
)
def test_safety_envelope(channel: str, value: float, fault: FaultCode) -> None:
    """Every documented limit is enforced."""
    monitor = SafetyMonitor(SafetyConfig())
    status = monitor.check_sample(SensorSample.from_mapping({channel: value}))
    assert fault in status.faults
    assert not status.assist_allowed


def test_emergency_stop_latches_until_cleared() -> None:
    """An emergency stop survives good samples until it is cleared."""
    monitor = SafetyMonitor(SafetyConfig())
    monitor.trigger_emergency_stop()
    assert not monitor.check_sample(SensorSample.from_mapping({})).assist_allowed
    monitor.clear()
    assert monitor.check_sample(SensorSample.from_mapping({})).assist_allowed


def test_stale_telemetry_is_a_fault() -> None:
    """A gap longer than ``stale_data_s`` inhibits assistance."""
    monitor = SafetyMonitor(SafetyConfig(stale_data_s=0.1, latch_faults=False))
    monitor.check_sample(SensorSample.from_mapping({"timestamp": 0.0}))
    status = monitor.check_sample(SensorSample.from_mapping({"timestamp": 1.0}))
    assert FaultCode.STALE_DATA in status.faults


def test_invalid_spd_is_reported_by_the_high_level_loop() -> None:
    """The manifold layer can raise a fault of its own."""
    monitor = SafetyMonitor(SafetyConfig())
    status = monitor.report_invalid_spd("covariance blew up")
    assert FaultCode.INVALID_SPD in status.faults and not status.assist_allowed


def test_limit_current_handles_nan() -> None:
    """Clamping a NaN command yields zero, never a NaN on the bus."""
    monitor = SafetyMonitor(SafetyConfig())
    assert monitor.limit_current(float("nan"), 1.0) == 0.0
    assert monitor.limit_current(5.0, 1.0) == pytest.approx(1.0)
    assert not monitor.check_command(float("inf"))


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


def test_state_machine_transitions() -> None:
    """The nominal sequence is allowed and illegal jumps are refused."""
    machine = StateMachine()
    for state in (
        SystemState.CALIBRATION,
        SystemState.BASELINE_COLLECTION,
        SystemState.MODEL_BUILDING,
        SystemState.READY,
        SystemState.WALKING,
        SystemState.OOD,
    ):
        machine.transition_to(state)
    assert machine.state is SystemState.OOD
    with pytest.raises(InvalidTransitionError):
        machine.transition_to(SystemState.CALIBRATION)
    machine.safe_stop()
    assert machine.state is SystemState.SAFE_STOP
    assert len(machine.history) == 7


# --------------------------------------------------------------------------- #
# Loops and logging
# --------------------------------------------------------------------------- #


def test_stride_log_columns(tmp_path: Path) -> None:
    """The stride log carries exactly the columns of the specification."""
    from gait_assistance.utils.logger import StrideLogRecord

    path = tmp_path / "log.csv"
    record = StrideLogRecord(
        stride_id=2, timestamp=1.0,
        gait_cluster="S0", cluster_confidence=0.5, ood=0,
        d_patient=0.1, d_healthy=float("nan"), d_target=0.2,
        healthy_region_threshold=float("nan"), manifold_deviation=0.0,
        estimated_swing_time=0.4, estimated_stance_time=0.8,
        estimated_stride_time=1.2, estimated_swing_ratio=0.33,
        estimated_stance_ratio=0.67, estimated_swing_stance_ratio=0.5,
        phase_source="belt_derived",
        belt_excursion=120.0, trunk_acc_rms=0.2,
        biomechanical_deviation=0.4, decision_state="BIOMECH_DEFICIT_ONLY",
        raw_assist_gain=0.32, assist_gain=0.3, target_belt_length=-140.0,
        motor_position=1.0, motor_velocity=2.0, motor_current=0.5,
    )
    with StrideLogger(path) as logger:
        logger.write(record)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header == STRIDE_LOG_COLUMNS

    cells = dict(zip(header, lines[1].split(",")))
    # Anything that was not measured is blank, never 0: a NaN distance, an
    # absent contralateral side and an unevaluated error must not read as
    # measurements.
    assert cells["d_healthy"] == ""
    assert cells["swing_symmetry_ratio"] == ""
    assert cells["trunk_gyro_rms"] == ""
    assert cells["swing_ratio_error"] == ""
    # Measured values survive, including a genuine zero.
    assert cells["d_patient"] == "0.1"
    assert cells["manifold_deviation"] == "0.0"
    assert cells["decision_state"] == "BIOMECH_DEFICIT_ONLY"
    assert cells["phase_source"] == "belt_derived"


def test_high_level_loop_respects_the_gain_bounds() -> None:
    """Running real strides through the high-level loop keeps every bound."""
    config = Config()
    strides = simulate_strides(config, 60.0)
    assert len(strides) > config.patient.baseline_strides + 5
    baseline = build_baseline_from_strides(strides[: config.patient.baseline_strides], config)
    model = PatientModel.build(config, baseline)
    loop = HighLevelLoop(config, model, logger=StrideLogger())

    previous = 0.0
    for stride in strides[config.patient.baseline_strides :]:
        output = loop.process_stride(stride)
        assert 0.0 <= output.assist_gain <= config.assist.max_gain
        assert output.gain_delta <= config.assist.max_gain_delta + 1e-12
        assert abs(output.assist_gain - previous) <= config.assist.max_gain_delta + 1e-12
        previous = output.assist_gain
    assert len(loop.logger.records) == len(strides) - config.patient.baseline_strides


def test_low_level_loop_never_exceeds_the_current_limit() -> None:
    """Every command sent to the motor respects the configured limit."""
    config = Config()
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    sensors = SensorManager(MockSensorSource(config.sensor, motor=motor), config.sensor)
    sensors.open()
    assist = AssistCommand(baseline_belt_length=-130.0)
    assist.publish(config.assist.max_gain, -130.0, 0)
    loop = LowLevelLoop(config, sensors, motor, assist)
    for _ in range(2000):
        output = loop.step()
        assert output is not None
        assert abs(motor.last_command_a) <= config.impedance.current_limit_a + 1e-12
        assert np.isfinite(motor.last_command_a)
        assert output.command.is_finite()


def test_low_level_loop_zeroes_the_command_on_a_fault() -> None:
    """A faulty frame de-energises the motor in the same cycle."""
    config = Config()
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    sensors = SensorManager(MockSensorSource(config.sensor, motor=motor), config.sensor)
    sensors.open()
    assist = AssistCommand(baseline_belt_length=-130.0)
    assist.publish(0.8, -130.0, 0)
    loop = LowLevelLoop(config, sensors, motor, assist)
    output = loop.step_with_sample(SensorSample.from_mapping({"belt_length": float("nan")}))
    assert not output.safety.assist_allowed
    assert output.command.current_a == 0.0
    assert motor.last_command_a == 0.0


def test_two_loop_runtime_builds_the_model_and_assists() -> None:
    """End to end: baseline -> model -> gain, with the state machine following."""
    config = Config()
    config.loop.high_level_in_thread = False
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    sensors = SensorManager(MockSensorSource(config.sensor, motor=motor), config.sensor)
    sensors.open()
    runtime = TwoLoopRuntime(
        config,
        LowLevelLoop(config, sensors, motor, AssistCommand()),
        baseline_collector=BaselineCollector(config),
        logger=StrideLogger(),
    )
    cycles = runtime.run(max_cycles=9000, realtime=False)

    assert cycles == 9000
    assert runtime.model is not None
    assert len(runtime.baseline_stride_ids) == config.patient.baseline_strides
    assert runtime.high is not None and runtime.high.outputs
    assert runtime.states.state in (SystemState.WALKING, SystemState.OOD, SystemState.READY)
    assert not runtime.errors
    gains = [o.assist_gain for o in runtime.high.outputs]
    assert all(0.0 <= g <= config.assist.max_gain for g in gains)
    assert max(abs(b - a) for a, b in zip(gains, gains[1:])) <= config.assist.max_gain_delta + 1e-12


def test_runtime_drift_triggers_ood_and_caps_the_gain() -> None:
    """A drifting gait is detected as OOD and the gain stays under the cap."""
    config = Config()
    config.loop.high_level_in_thread = False
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    source = MockSensorSource(config.sensor, motor=motor, drift_after_s=45.0)
    sensors = SensorManager(source, config.sensor)
    sensors.open()
    runtime = TwoLoopRuntime(
        config,
        LowLevelLoop(config, sensors, motor, AssistCommand()),
        baseline_collector=BaselineCollector(config),
        logger=StrideLogger(),
    )
    runtime.run(max_cycles=12000, realtime=False)
    outputs = runtime.high.outputs if runtime.high is not None else []
    ood = [o for o in outputs if o.analysis.is_ood]
    assert ood, "the drifted gait should leave the patient model"
    assert all(o.assist_gain <= config.assist.ood_max_gain + 1e-12 for o in ood[3:])


# --------------------------------------------------------------------------- #
# Baseline vs pathological deviation, and the assist mode
# --------------------------------------------------------------------------- #


def test_baseline_and_pathological_deviations_are_separate_terms() -> None:
    """The two deviations use their own scales and do not collapse into one."""
    scorer = DeviationScorer(
        manifold_scale=2.0,
        baseline_scale=4.0,
        healthy_scale=8.0,
        target_swing_ratio=0.38,
        reference_excursion_mm=100.0,
        mode=AssistMode.HEALTHY_DIRECTED,
    )
    components = scorer.score(
        _analysis(1.0, d_patient=2.0, d_healthy=4.0), _metrics(0.38, 100.0)
    )
    assert components.e_manifold == pytest.approx(0.5)   # 1.0 / 2.0
    assert components.e_baseline == pytest.approx(0.5)   # 2.0 / 4.0
    assert components.e_healthy == pytest.approx(0.5)    # 4.0 / 8.0
    # Only the target term drives the gain: the other two are reported, never
    # summed in, so the same distance cannot raise the gain twice.
    assert components.score == pytest.approx(0.5 * 0.5)
    assert components.mode is AssistMode.HEALTHY_DIRECTED


def test_pathological_deviation_stays_undefined_without_a_reference() -> None:
    """``e_healthy`` is NaN in BASELINE_STABILIZATION, never a silent zero."""
    scorer = DeviationScorer(
        manifold_scale=2.0,
        baseline_scale=2.0,
        target_swing_ratio=0.38,
        reference_excursion_mm=100.0,
    )
    components = scorer.score(_analysis(1.0, d_healthy=float("nan")))
    assert components.mode is AssistMode.BASELINE_STABILIZATION
    assert np.isnan(components.e_healthy)
    assert not np.isnan(components.e_baseline)
    # An unmeasurable pathology must not read as no pathology.
    assert components.e_healthy != 0.0 or np.isnan(components.e_healthy)


def test_assist_mode_follows_the_healthy_reference() -> None:
    """The mode is derived from the model, not configured by hand."""
    config = Config()
    strides = simulate_strides(config, 60.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )

    stabilising = PatientModel.build(config, baseline)
    assert stabilising.assist_mode is AssistMode.BASELINE_STABILIZATION
    assert np.isnan(stabilising.analyze_stride(strides[-1]).d_healthy)
    assert np.allclose(stabilising.z_target, stabilising.z_patient)
    assert any("BASELINE_STABILIZATION" in w for w in stabilising.warnings)

    from gait_assistance.manifold.reference import build_healthy_reference

    healthy = build_healthy_reference(
        [matrix_log(make_spd(5, seed=s)) for s in range(12)]
    )
    directed = PatientModel.build(config, baseline, healthy)
    assert directed.assist_mode is AssistMode.HEALTHY_DIRECTED
    assert np.isfinite(directed.analyze_stride(strides[-1]).d_healthy)
    assert directed.summary()["assist_mode"] == "healthy_directed"


def test_scorer_from_model_derives_independent_scales() -> None:
    """Baseline and healthy scales come from different distributions."""
    config = Config()
    strides = simulate_strides(config, 60.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    from gait_assistance.manifold.reference import build_healthy_reference

    healthy = build_healthy_reference(
        [matrix_log(make_spd(5, seed=s)) for s in range(12)]
    )
    model = PatientModel.build(config, baseline, healthy)
    scorer = DeviationScorer.from_model(model, DeviationConfig())
    stats = model.baseline_distance_stats
    assert scorer.mode is AssistMode.HEALTHY_DIRECTED
    assert scorer.baseline_scale == pytest.approx(
        stats["d_patient_mean"] + 2.0 * stats["d_patient_std"]
    )
    assert scorer.healthy_scale == pytest.approx(healthy.distance_threshold)
    assert scorer.baseline_scale != scorer.healthy_scale


def test_low_level_loop_holds_position_until_the_model_exists() -> None:
    """Before a baseline is published the belt is not pulled towards 0 mm.

    ``AssistCommand`` starts with ``baseline_belt_length = 0.0``, which is tens
    of millimetres away from where the belt sits on a wearer, so a naive target
    would command a sustained current for the whole baseline collection.
    """
    config = Config()
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    sensors = SensorManager(MockSensorSource(config.sensor, motor=motor), config.sensor)
    sensors.open()
    loop = LowLevelLoop(config, sensors, motor, AssistCommand())
    for _ in range(200):
        output = loop.step()
        assert output is not None
        assert output.target.target_belt_length == pytest.approx(output.sample.belt_length)
        assert output.command.position_error_mm == pytest.approx(0.0)
