"""Central configuration for the gait-assistance system.

Every tunable quantity lives in a nested dataclass so that the whole system can
be reconfigured from a single JSON file without touching code::

    cfg = Config.load("patient_01.json")
    cfg = cfg.override({"assist.max_gain": 0.6, "manifold.epsilon": 1e-5})
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class RunMode(str, Enum):
    """How the system obtains its data."""

    OFFLINE = "offline"        #: replay a CSV file as fast as possible
    SIMULATION = "simulation"  #: synthetic sensors + simulated motor
    HARDWARE = "hardware"      #: real PAD device over the serial link


class AssistMode(str, Enum):
    """What the assistance is steering the patient *towards*.

    The mode is not a user setting: it is decided by whether a healthy
    reference was supplied, and it changes the meaning of the deviation score,
    so it is reported alongside every result.

    ``BASELINE_STABILIZATION``
        No healthy reference.  ``Z_target == Z_patient``, so the only thing the
        system can measure is how far the current stride drifts from *this
        patient's own* habitual pattern.  That is a consistency signal, not a
        measure of pathology: a stride can sit exactly on the baseline and
        still be severely pathological.  Assistance therefore stabilises the
        patient around their own baseline and claims nothing more.

    ``HEALTHY_DIRECTED``
        A healthy reference exists.  ``Z_target`` is interpolated towards
        ``Z_healthy``, and the distance to ``Z_healthy`` is a genuine
        pathological deviation, reported separately from the baseline
        deviation so the two are never read as the same quantity.
    """

    BASELINE_STABILIZATION = "baseline_stabilization"
    HEALTHY_DIRECTED = "healthy_directed"


class TargetMode(str, Enum):
    """What the manifold deviation is measured against.

    ``HEALTHY_REGION`` (default)
        The able-bodied strides define a *region*, not a point: a centroid plus
        a distance threshold.  A stride inside the region has a manifold
        deviation of exactly 0 and the patient is never pushed towards the
        healthy centroid itself.  Only the excess beyond the threshold counts.

    ``INTERPOLATED_TARGET``
        Compatibility mode reproducing the earlier behaviour: the target is the
        point ``(1 - alpha) * Z_patient + alpha * Z_healthy`` and the deviation
        is the plain distance to it, so there is no zero-error region.

    ``PATIENT_BASELINE``
        The target is the patient's own baseline centroid; used when no healthy
        reference exists, where it is the only reference available.
    """

    HEALTHY_REGION = "healthy_region"
    INTERPOLATED_TARGET = "interpolated_target"
    PATIENT_BASELINE = "patient_baseline"


class ThresholdMethod(str, Enum):
    """How the healthy region's distance threshold is derived."""

    PERCENTILE = "percentile"    #: percentile of the healthy distances
    MEAN_STD = "mean_std"        #: mean + sigma * std
    MEDIAN_MAD = "median_mad"    #: median + k * MAD (robust to outliers)


class OodAssistPolicy(str, Enum):
    """What the assist gain does while a stride is out of distribution."""

    HOLD = "hold"                  #: freeze the gain at its previous value
    SAFE_MINIMUM = "safe_minimum"  #: fall back to ``ood_safe_gain``


#: Canonical sensor channel names used everywhere in the package.
CHANNELS: Tuple[str, ...] = (
    "timestamp",
    "belt_length",
    "belt_velocity",
    "motor_position",
    "motor_velocity",
    "motor_current",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)

#: Column-name aliases accepted when reading CSV logs produced by the existing
#: acquisition scripts in this repository.
DEFAULT_CSV_ALIASES: Dict[str, Tuple[str, ...]] = {
    "timestamp": ("host_time_s", "time_s", "device_time_s", "timestamp", "t"),
    "belt_length": ("belt_length", "belt_mm", "belt"),
    "belt_velocity": ("belt_velocity", "belt_vel"),
    "motor_position": ("motor_pos", "motor_position"),
    "motor_velocity": ("motor_vel", "motor_velocity"),
    "motor_current": ("motor_iq_meas", "motor_current", "iq_meas", "motor_iq_set"),
    "accel_x": ("accel_x", "acc_x", "ax"),
    "accel_y": ("accel_y", "acc_y", "ay"),
    "accel_z": ("accel_z", "acc_z", "az"),
    "gyro_x": ("gyro_x", "gx"),
    "gyro_y": ("gyro_y", "gy"),
    "gyro_z": ("gyro_z", "gz"),
}


# --------------------------------------------------------------------------- #
# Configuration sections
# --------------------------------------------------------------------------- #


@dataclass
class SensorConfig:
    """Acquisition settings (spec 1)."""

    sample_rate_hz: float = 100.0
    #: explicit canonical-name -> CSV-column overrides; empty means auto-detect
    csv_column_map: Dict[str, str] = field(default_factory=dict)
    #: replay CSV timestamps in real time instead of as fast as possible
    csv_realtime: bool = False
    serial_port: str = ""
    serial_baudrate: int = 115200
    #: low-pass cut-off applied to belt velocity before phase detection (Hz);
    #: ``0`` disables filtering
    velocity_filter_hz: float = 0.0


@dataclass
class PhaseConfig:
    """Gait phase *estimation* (spec 2).

    The defaults mirror ``heelstrike_detect.py`` of this repository.

    .. warning::
       The default ``scheduled`` detector does not observe the patient at all:
       it emits swing and stance on a fixed clock so the rest of the pipeline
       can run before the real detector is merged.  Every phase-derived number
       it produces describes the schedule, not the gait.

       The two signal-driven detectors produce an **estimated** gait phase
       inferred
       from proximal kinematics (belt travel or shank angular rate).  Neither
       observes foot-ground contact, so neither has been validated against a
       ground-truth source.  The belt-velocity detector in particular marks
       the whole belt-extension interval as SWING, which is longer than the
       biomechanical swing phase, so every phase-derived quantity
       (``swing_time``, ``stance_time``, ``swing_ratio``) is an estimate on an
       uncalibrated scale and must not be compared against published
       normative values.  Validation against a foot-switch/FSR insole or a
       foot-mounted IMU is required before these numbers carry clinical
       meaning; see ``ground_truth_validated`` on the detector classes.
    """

    #: "scheduled" | "belt_velocity" | "gyro", plus anything a future module
    #: registers through ``register_phase_detector``.  The default is the
    #: fixed-cadence placeholder: it stands in for the real swing/stance
    #: detector until that lands, and it is a clock, not a measurement.
    detector: str = "scheduled"

    # -- scheduled (fixed-cadence placeholder) ------------------------------ #
    #: stride period, heel strike to heel strike (s)
    scheduled_period_s: float = 1.2
    #: fraction of the period spent in swing; stance takes the remainder
    scheduled_swing_ratio: float = 0.40
    #: absolute swing duration (s); ``0`` derives it from the ratio.  Set this
    #: when the timing is easier to express as a duration than as a fraction.
    scheduled_swing_duration_s: float = 0.0
    #: shifts the whole cycle in time (s), so the first heel strike can be
    #: aligned with a recording instead of with the first sample
    scheduled_offset_s: float = 0.0

    # -- belt_velocity ------------------------------------------------------ #
    min_belt_mm: float = -130.0          #: negative -> belt <= value, positive -> belt >= value
    refractory_s: float = 0.35
    loading_vel: float = 10.0            #: mm/s, belt extending -> SWING
    strike_vel: float = -5.0             #: mm/s, belt retracting -> heel strike

    # -- gyro --------------------------------------------------------------- #
    gyro_channel: str = "gyro_z"         #: used by the gyro detector
    gyro_swing_threshold: float = 40.0   #: deg/s
    gyro_strike_threshold: float = -20.0 #: deg/s

    def swing_duration_s(self) -> float:
        """Scheduled swing duration, from the explicit value or the ratio.

        Returns:
            The duration in seconds, clamped to leave a positive stance.
        """
        period = max(float(self.scheduled_period_s), 1e-6)
        if self.scheduled_swing_duration_s > 0.0:
            duration = float(self.scheduled_swing_duration_s)
        else:
            duration = period * float(self.scheduled_swing_ratio)
        # A cycle needs both phases: never let swing swallow the whole period.
        return float(min(max(duration, 0.0), period * 0.99))

    def stance_duration_s(self) -> float:
        """Scheduled stance duration, the remainder of the period."""
        return float(max(self.scheduled_period_s, 1e-6)) - self.swing_duration_s()


@dataclass
class StrideConfig:
    """Stride segmentation and normalisation (spec 3, 4)."""

    n_cycle_points: int = 100
    min_samples: int = 15
    min_duration_s: float = 0.5
    max_duration_s: float = 2.5


@dataclass
class FeatureConfig:
    """Feature selection (spec 5, 6)."""

    channels: Tuple[str, ...] = (
        "belt_length",
        "belt_velocity",
        "accel_x",
        "accel_y",
        "gyro_z",
    )
    #: floor applied to the baseline standard deviations
    std_floor: float = 1e-8


@dataclass
class ManifoldConfig:
    """SPD covariance and Log-Euclidean mapping (spec 7-10)."""

    epsilon: float = 1e-6          #: covariance regularisation and eigenvalue floor
    spd_check_tol: float = 1e-10   #: tolerance of the positive-definiteness test


@dataclass
class ClusterConfig:
    """Patient-specific gait-state clustering (spec 12)."""

    k_candidates: Tuple[int, ...] = (1, 2, 3, 4)
    min_silhouette: float = 0.35
    min_stability: float = 0.60
    stability_repeats: int = 10
    stability_subsample: float = 0.8
    n_init: int = 10
    random_state: int = 0
    #: absolute floor on the number of strides per cluster for a K to be
    #: accepted
    min_cluster_size: int = 3
    #: additional *relative* floor: a cluster must also hold at least this
    #: fraction of the strides being clustered.  With the default baseline of
    #: 30 strides this requires 5 strides per cluster, which stops a K from
    #: being accepted on the strength of a handful of outliers.  The effective
    #: requirement is ``max(min_cluster_size, ceil(min_cluster_fraction * n))``.
    min_cluster_fraction: float = 0.15
    #: below this many strides the partition is reported as under-powered; the
    #: K is still evaluated, but the model carries a warning (see
    #: ``PatientConfig.baseline_strides``)
    recommended_strides: int = 30


@dataclass
class PatientConfig:
    """Patient-specific baseline, online classification and OOD (spec 11, 13, 14, 16)."""

    #: Number of opening strides that form the patient-specific baseline.
    #: 30-50 is the recommended range: the baseline has to fix the z-score
    #: statistics, the log-domain centroid *and* a k-means partition, and a
    #: 20-stride sample splits into clusters that are too small for the
    #: silhouette and stability gates to mean much.  Values below
    #: ``ClusterConfig.recommended_strides`` are accepted but flagged.
    baseline_strides: int = 30
    confidence_threshold: float = 0.15
    #: on low confidence: keep the previous state (True) or report UNKNOWN
    hold_state_on_low_confidence: bool = True
    #: OOD threshold = mean + ood_sigma * std of the baseline nearest-centroid
    #: distances; ``ood_threshold_override`` (>0) replaces it outright
    ood_sigma: float = 3.0
    ood_threshold_override: float = 0.0
    alpha: float = 0.3             #: Z_target = (1-alpha)*Z_patient + alpha*Z_healthy


@dataclass
class ReferenceConfig:
    """Healthy reference *region* statistics (spec 15).

    The reference is a region, not a point: a log-domain centroid plus the
    distribution of the healthy strides around it.
    """

    #: how the region's outer boundary is computed from the healthy distances
    threshold_method: ThresholdMethod = ThresholdMethod.PERCENTILE
    #: used by ``PERCENTILE``: the default 95 puts the boundary at the 95th
    #: percentile of the healthy distance distribution
    threshold_percentile: float = 95.0
    #: used by ``MEAN_STD``: threshold = mean + sigma * std
    threshold_sigma: float = 2.0
    #: used by ``MEDIAN_MAD``: threshold = median + k * MAD
    threshold_mad_k: float = 3.0
    #: biomechanical reference intervals are ``[q_lower, q_upper]`` of the
    #: healthy distribution of each metric
    metric_lower_percentile: float = 5.0
    metric_upper_percentile: float = 95.0
    #: reserved for speed-conditioned references; the interface accepts a
    #: walking speed already, the current build ignores it and returns the
    #: pooled population interval
    speed_conditioned: bool = False
    speed_bin_width_mps: float = 0.2
    path: str = ""                 #: optional .npz file with a stored reference


@dataclass
class DeviationConfig:
    """Manifold context score, biomechanical deficit and their combination.

    The manifold term is a *severity/context* factor and the biomechanical
    term is the *actionable* deficit; they are combined multiplicatively
    (see :class:`AssistConfig` and the assistance policy), never summed, so a
    manifold deviation on its own cannot command the motor.
    """

    #: what the manifold deviation is measured against
    target_mode: TargetMode = TargetMode.HEALTHY_REGION

    # -- assistance score E_assist = E_B * (base_factor + manifold_factor*E_R)
    base_factor: float = 0.5
    manifold_factor: float = 0.5

    #: weights of the device-actionable terms inside ``E_B``; only these keys
    #: are recognised, and a weight of 0 removes the term from the gain while
    #: leaving it in the log
    biomech_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "swing_ratio": 0.5,
            "belt_excursion": 0.5,
            "temporal_symmetry": 0.0,
            "trunk_compensation": 0.0,
        }
    )
    #: deviation beyond the healthy interval that saturates each term; ``0``
    #: for ``belt_excursion_scale`` means "use the healthy lower bound", which
    #: makes a total loss of excursion an error of 1
    swing_ratio_scale: float = 0.10
    belt_excursion_scale: float = 0.0
    symmetry_scale: float = 0.25
    trunk_scale: float = 0.0

    # -- legacy weighted-sum score (kept for the compatibility path) -------- #
    w_manifold: float = 0.5
    w_swing: float = 0.3
    w_excursion: float = 0.2
    #: distance to ``Z_target`` that maps to a manifold error of 1.0; ``0`` ->
    #: derived from the baseline distance distribution when the model is built
    manifold_scale: float = 0.0
    #: distance to ``Z_patient`` that maps to a *baseline* deviation of 1.0
    #: (departure from this patient's own habitual pattern); ``0`` -> derived
    #: from the spread of the baseline strides around their own centroid
    baseline_scale: float = 0.0
    #: distance to ``Z_healthy`` that maps to a *pathological* deviation of
    #: 1.0; ``0`` -> the healthy reference threshold.  Undefined, and reported
    #: as NaN, in ``AssistMode.BASELINE_STABILIZATION``
    healthy_scale: float = 0.0
    #: healthy swing ratio target and the deviation that saturates the term
    target_swing_ratio: float = 0.38
    swing_ratio_tolerance: float = 0.12
    #: reference belt excursion (mm); ``0`` -> taken from the healthy reference
    #: or, failing that, from the patient baseline
    reference_excursion_mm: float = 0.0
    excursion_tolerance: float = 0.5   #: relative deficit that saturates the term


@dataclass
class AssistConfig:
    """Assist gain, swing profile and belt target (spec 20-22)."""

    max_gain: float = 0.8
    max_gain_delta: float = 0.1
    ood_max_gain: float = 0.2         #: hard cap while out of distribution
    #: what the gain does while out of distribution: hold the previous value or
    #: drop to ``ood_safe_gain``.  Aggressive assistance is refused either way.
    ood_policy: OodAssistPolicy = OodAssistPolicy.HOLD
    ood_safe_gain: float = 0.0
    #: a deficit must persist this many consecutive strides before the gain is
    #: allowed to rise, and the gait must be back inside its healthy intervals
    #: this many consecutive strides before it is allowed to fall.  Guards the
    #: policy against single-stride noise; works together with the rate limit.
    required_consecutive_deficit_strides: int = 3
    required_consecutive_recovery_strides: int = 3
    profile: str = "sine"             #: "sine" | "raised_cosine" | "trapezoid"
    max_retraction_mm: float = 30.0
    #: expected swing duration used to interpolate the profile before the first
    #: stride statistics are available (s)
    default_swing_duration_s: float = 0.4


@dataclass
class ImpedanceConfig:
    """Low-level impedance controller (spec 23)."""

    stiffness_n_per_mm: float = 0.005
    damping_n_per_mms: float = 0.005
    current_limit_a: float = 1.0
    #: use the PAD force->current map from ``pad_external_control_lib`` when
    #: available; otherwise ``current_per_newton`` is used
    use_pad_force_map: bool = True
    current_per_newton: float = 1.0 / 0.87


@dataclass
class SafetyConfig:
    """Safety envelope (spec 25)."""

    max_motor_position: float = 1.0e6
    max_motor_current_a: float = 10.0
    max_belt_length_mm: float = 400.0
    min_belt_length_mm: float = -400.0
    max_belt_velocity_mms: float = 2000.0
    max_accel_g: float = 16.0
    max_gyro_dps: float = 2000.0
    stale_data_s: float = 0.25
    #: faults stay latched until :meth:`SafetyMonitor.clear` is called
    latch_faults: bool = True


@dataclass
class LoopConfig:
    """Loop timing (spec 24)."""

    low_level_hz: float = 100.0
    high_level_in_thread: bool = True
    high_level_queue_size: int = 8


@dataclass
class LoggingConfig:
    """Stride-level CSV logging (spec 27)."""

    stride_csv: str = ""
    verbose: bool = True


@dataclass
class Config:
    """Aggregate configuration object."""

    mode: RunMode = RunMode.OFFLINE
    sensor: SensorConfig = field(default_factory=SensorConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    stride: StrideConfig = field(default_factory=StrideConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    patient: PatientConfig = field(default_factory=PatientConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    deviation: DeviationConfig = field(default_factory=DeviationConfig)
    assist: AssistConfig = field(default_factory=AssistConfig)
    impedance: ImpedanceConfig = field(default_factory=ImpedanceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-JSON representation of the configuration."""
        data = asdict(self)
        data["mode"] = self.mode.value
        return _tuples_to_lists(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Build a configuration from a (possibly partial) mapping."""
        return _build_dataclass(cls, data)

    def save(self, path: Union[str, Path]) -> None:
        """Write the configuration to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Read a configuration from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- convenience ------------------------------------------------------- #

    def override(self, updates: Mapping[str, Any]) -> "Config":
        """Return a copy with dotted-path ``updates`` applied.

        Args:
            updates: mapping such as ``{"assist.max_gain": 0.5}``.

        Returns:
            A new :class:`Config`; the receiver is left untouched.
        """
        data = self.to_dict()
        for dotted, value in updates.items():
            node: Any = data
            parts = dotted.split(".")
            for key in parts[:-1]:
                if key not in node:
                    raise KeyError(f"unknown configuration section: {dotted!r}")
                node = node[key]
            if parts[-1] not in node:
                raise KeyError(f"unknown configuration key: {dotted!r}")
            node[parts[-1]] = value
        return Config.from_dict(data)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists so the result is JSON serialisable."""
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _build_dataclass(cls: type, data: Mapping[str, Any]) -> Any:
    """Instantiate the dataclass ``cls`` from ``data``, recursing into fields."""
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, Mapping):
            kwargs[f.name] = _build_dataclass(f.type, value)
        elif isinstance(f.type, type) and issubclass(f.type, Enum):
            kwargs[f.name] = f.type(value)
        else:
            kwargs[f.name] = _coerce(cls, f.name, value)
    return cls(**kwargs)


def _coerce(cls: type, name: str, value: Any) -> Any:
    """Restore tuple-typed fields that JSON turned into lists."""
    default_holder = cls()
    default = getattr(default_holder, name, None)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(default, Enum) and not isinstance(value, Enum):
        return type(default)(value)
    if is_dataclass(default) and isinstance(value, Mapping):
        return _build_dataclass(type(default), value)
    return value


def default_config() -> Config:
    """Return a configuration with all defaults from the specification."""
    return Config()


__all__ = [
    "CHANNELS",
    "DEFAULT_CSV_ALIASES",
    "AssistConfig",
    "AssistMode",
    "ClusterConfig",
    "Config",
    "DeviationConfig",
    "FeatureConfig",
    "ImpedanceConfig",
    "LoggingConfig",
    "LoopConfig",
    "ManifoldConfig",
    "PatientConfig",
    "PhaseConfig",
    "ReferenceConfig",
    "RunMode",
    "SafetyConfig",
    "SensorConfig",
    "StrideConfig",
    "OodAssistPolicy",
    "TargetMode",
    "ThresholdMethod",
    "default_config",
]
