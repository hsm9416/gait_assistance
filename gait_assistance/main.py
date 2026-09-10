"""Command-line entry point (spec 33).

Subcommands::

    offline    replay a patient CSV: patient-specific baseline -> clustering
               -> online replay -> deviation -> assist gain -> stride log
               -> plots
    healthy    build a healthy reference from an able-bodied CSV
    simulate   run the whole stack against MockSensor / MockMotor
    live       run against the real PAD device over the serial link
    inspect    report the strides a recording yields (detector tuning)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .config import AssistMode, Config, RunMode, TargetMode
from .control.assistance_policy import PHASE_DERIVED_METRICS, TERM_METRICS
from .gait.phase_detector import create_phase_detector
from .loops import AssistCommand, LowLevelLoop, TwoLoopRuntime
from .manifold.reference import HealthyReference, ReferenceBank, load_reference
from .offline_sim import (
    OfflineSimulator,
    build_healthy_reference_from_csv,
    segment_csv,
)
from .patient.baseline import BaselineCollector
from .patient.patient_model import PatientModel
from .sensors.encoder import MockMotor, MotorInterface, PadMotor
from .sensors.sensor_manager import (
    MockSensorSource,
    PadSensorSource,
    SensorManager,
)
from .state_machine import SystemState
from .utils.logger import StrideLogger


def load_config(path: Optional[str], overrides: Sequence[str]) -> Config:
    """Load a configuration file and apply ``key=value`` overrides.

    Args:
        path: optional JSON configuration file.
        overrides: dotted assignments such as ``assist.max_gain=0.5``.

    Returns:
        The resulting :class:`~.config.Config`.

    Raises:
        SystemExit: on a malformed override.
    """
    config = Config.load(path) if path else Config()
    updates: Dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"malformed --set option: {item!r} (expected key=value)")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        updates[key.strip()] = value
    if updates:
        try:
            config = config.override(updates)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    return config


def cmd_offline(args: argparse.Namespace) -> int:
    """Run the offline replay pipeline."""
    config = load_config(args.config, args.set)
    config.mode = RunMode.OFFLINE
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    healthy: Optional[Union[HealthyReference, ReferenceBank]] = None
    if args.healthy_csv:
        healthy = build_healthy_reference_from_csv(args.healthy_csv, config)
        healthy.save(out_dir / "healthy_reference.npz")
        print(
            f"[healthy] {healthy.n_strides} strides  "
            f"mean={healthy.distance_mean:.3f}  std={healthy.distance_std:.3f}  "
            f"threshold={healthy.distance_threshold:.3f}"
        )
    elif args.healthy_ref:
        healthy = load_reference(args.healthy_ref)

    stride_csv = out_dir / "stride_log.csv"
    simulator = OfflineSimulator(config, args.csv, healthy, stride_csv=stride_csv)
    result = simulator.run(realtime=args.realtime)

    print(json.dumps(result.summary(), indent=2, default=str))
    if result.model is not None:
        _report_model_context(result.model, config)
    table_path = result.save_table(out_dir / "stride_table.csv")
    print(f"[out] stride log   : {stride_csv}")
    print(f"[out] stride table : {table_path}")
    if result.model is not None:
        model_path = out_dir / "patient_model.json"
        result.model.save(model_path)
        print(f"[out] patient model: {model_path}")
    if not args.no_plot:
        from .plotting import plot_results

        image = plot_results(result, out_dir / "results.png")
        if image is not None:
            print(f"[out] plots        : {image}")
    return 0 if result.model is not None else 1


def cmd_healthy(args: argparse.Namespace) -> int:
    """Build and store a healthy reference."""
    config = load_config(args.config, args.set)
    reference = build_healthy_reference_from_csv(
        args.csv, config, max_strides=args.max_strides
    )
    reference.save(args.out)
    print(
        f"[healthy] strides={reference.n_strides}  "
        f"mean={reference.distance_mean:.4f}  std={reference.distance_std:.4f}  "
        f"threshold={reference.distance_threshold:.4f}  -> {args.out}"
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Report how a recording segments, to help tune the phase detector."""
    config = load_config(args.config, args.set)
    strides, samples, rejected = segment_csv(args.csv, config)
    print(f"phase source     : {type(create_phase_detector(config.phase)).describe()}")
    print(f"samples          : {len(samples)}")
    print(f"strides accepted : {len(strides)}")
    print(f"strides rejected : {len(rejected)}")
    for stride in strides[:10]:
        print(
            f"  stride {stride.stride_id:3d}  n={stride.n_samples:4d}  "
            f"duration={stride.duration_s:.3f}s"
        )
    for stride_id, reason in rejected[:10]:
        print(f"  rejected {stride_id:3d}: {reason}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run the full stack against the mock sensors and motor."""
    config = load_config(args.config, args.set)
    config.mode = RunMode.SIMULATION
    if args.fast:
        # An unpaced replay outruns any worker thread, and a full queue drops
        # strides on purpose (the fast loop must never block).  Run the
        # high-level loop inline instead so a fast run stays deterministic.
        config.loop.high_level_in_thread = False
    motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
    source = MockSensorSource(
        config.sensor, motor=motor, drift_after_s=args.drift_after
    )
    sensors = SensorManager(source, config.sensor)
    sensors.open()
    logger = StrideLogger(args.stride_csv or None)
    runtime = _build_runtime(config, sensors, motor, logger)
    if args.fast:
        # unpaced: bound the run by simulated time instead of wall-clock time
        cycles = runtime.run(
            max_cycles=int(args.duration * config.sensor.sample_rate_hz),
            realtime=False,
        )
    else:
        cycles = runtime.run(duration_s=args.duration, realtime=True)
    logger.close()
    _report(runtime, cycles)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Run the full stack against the PAD hardware."""
    config = load_config(args.config, args.set)
    config.mode = RunMode.HARDWARE
    if args.port:
        config.sensor.serial_port = args.port
    source = PadSensorSource(config.sensor)
    sensors = SensorManager(source, config.sensor)
    sensors.open()
    motor: MotorInterface = PadMotor(
        source.controller, current_limit_a=config.impedance.current_limit_a
    )
    logger = StrideLogger(args.stride_csv or None)
    runtime = _build_runtime(config, sensors, motor, logger)
    try:
        cycles = runtime.run(duration_s=args.duration, realtime=True)
    except KeyboardInterrupt:
        print("\n[info] interrupted by the operator")
        runtime.low.safety.trigger_emergency_stop("keyboard interrupt")
        runtime.stop()
        cycles = runtime.low.cycles
    finally:
        logger.close()
        sensors.close()
    _report(runtime, cycles)
    return 0


def _build_runtime(
    config: Config,
    sensors: SensorManager,
    motor: MotorInterface,
    logger: StrideLogger,
) -> TwoLoopRuntime:
    """Assemble the two-loop runtime shared by the live and simulated modes."""
    healthy = (
        load_reference(config.reference.path) if config.reference.path else None
    )
    low = LowLevelLoop(config, sensors, motor, AssistCommand())
    return TwoLoopRuntime(
        config,
        low,
        baseline_collector=BaselineCollector(config),
        healthy=healthy,
        logger=logger,
        on_model_built=lambda model: print(
            "[model] " + json.dumps(model.summary(), default=str)
        ),
    )


def _report_model_context(model: "PatientModel", config: Config) -> None:
    """State what the numbers above mean before anyone reads them.

    Prints the assist mode, the phase-estimation caveat and every warning the
    model carries.  It deliberately reports values without grading them: no
    silhouette or stability figure is turned into a quality verdict here,
    because those scores describe the partition, not the patient's gait.
    """
    mode = model.assist_mode
    print(f"[mode] {mode.value}")
    if mode is AssistMode.BASELINE_STABILIZATION:
        print(
            "[mode] no healthy reference -> the target is this patient's own "
            "baseline.  d_healthy / e_healthy are undefined and the deviation "
            "score measures consistency, not pathology."
        )
    else:
        print(
            "[mode] healthy reference loaded -> d_healthy / e_healthy carry the "
            "pathological deviation, reported separately from the baseline one."
        )
    print(f"[target] {model.target_mode.value}")
    if model.target_mode is TargetMode.HEALTHY_REGION:
        print(
            f"[target] healthy region boundary d_H <= "
            f"{model.healthy_region_threshold:.4f}; a stride inside it has "
            "manifold deviation 0 and is never pushed towards the centroid."
        )
    detector = type(create_phase_detector(config.phase))
    print(f"[phase] {detector.describe()}")
    if detector.is_placeholder:
        dead = [
            term
            for term, metric in TERM_METRICS.items()
            if metric in PHASE_DERIVED_METRICS
            and config.deviation.biomech_weights.get(term, 0.0) > 0.0
        ]
        print(
            "[phase] the cadence is fixed, so every phase-derived metric is "
            "constant by construction and cannot show a deficit."
        )
        if dead:
            print(
                f"[warn] these weighted terms cannot fire until the real "
                f"detector lands: {', '.join(sorted(dead))}.  Assistance can "
                f"currently only be driven by the sensor-derived terms."
            )
    else:
        print(
            "[phase] estimated_swing_time / estimated_stance_time / "
            "estimated_swing_ratio are estimated on an uncalibrated scale; "
            "compare them within this patient only."
        )
    print(
        "[assist] E_R (manifold) is a severity factor only. Assistance needs a "
        "device-actionable deficit E_B > 0: "
        "E_assist = E_B * (base_factor + manifold_factor * E_R)."
    )
    for warning in model.warnings:
        # The missing-reference caveat is already the [mode] block above; the
        # rest (sample size) has not been said yet.
        if warning.startswith("no healthy reference"):
            continue
        print(f"[warn] {warning}")


def _report(runtime: TwoLoopRuntime, cycles: int) -> None: 
    """Print a short end-of-run report."""
    print(f"[run] cycles={cycles}  state={runtime.states.state.value}")
    print(f"[run] baseline strides={len(runtime.baseline_stride_ids)}")
    if runtime.high is not None:
        outputs = runtime.high.outputs
        print(f"[run] analysed strides={len(outputs)}")
        if outputs:
            gains = [o.assist_gain for o in outputs]
            print(
                f"[run] gain mean={sum(gains) / len(gains):.3f}  max={max(gains):.3f}  "
                f"ood={sum(o.analysis.is_ood for o in outputs)}"
                # only a streak of flagged strides actually caps the gain
                f" (capping={sum(1 for o in outputs if o.assessment is not None and o.assessment.ood_engaged)})"
            )
            counts: Dict[str, int] = {}
            for output in outputs:
                counts[output.decision_state] = counts.get(output.decision_state, 0) + 1
            summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k)
            if summary:
                print(f"[run] decision states: {summary}")
    if runtime.model is not None:
        _report_model_context(runtime.model, runtime.config)
    if runtime.dropped_strides:
        print(f"[warn] dropped strides: {runtime.dropped_strides}")
    for error in runtime.errors:
        print(f"[error] {error}")
        print(f"[safety] latched: {runtime.low.safety.status.fault_names}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gait_assistance",
        description="Real-time hemiparetic gait assistance on the Log-Euclidean SPD manifold",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        """Add the configuration options shared by every subcommand."""
        sub.add_argument("--config", help="JSON configuration file")
        sub.add_argument(
            "--set", action="append", default=[], metavar="KEY=VALUE",
            help="configuration override, e.g. --set assist.max_gain=0.5",
        )

    offline = subparsers.add_parser("offline", help="replay a patient CSV recording")
    offline.add_argument("csv", help="patient recording")
    offline.add_argument("--healthy-csv", help="able-bodied recording for the reference")
    offline.add_argument("--healthy-ref", help="stored healthy reference (.npz)")
    offline.add_argument("--out-dir", default="results", help="output directory")
    offline.add_argument("--realtime", action="store_true", help="replay at real speed")
    offline.add_argument("--no-plot", action="store_true", help="skip the figures")
    add_common(offline)
    offline.set_defaults(func=cmd_offline)

    healthy = subparsers.add_parser("healthy", help="build a healthy reference")
    healthy.add_argument("csv", help="able-bodied recording")
    healthy.add_argument("--out", default="healthy_reference.npz", help="destination")
    healthy.add_argument("--max-strides", type=int, default=None, help="stride cap")
    add_common(healthy)
    healthy.set_defaults(func=cmd_healthy)

    inspect = subparsers.add_parser("inspect", help="report stride segmentation")
    inspect.add_argument("csv", help="recording to inspect")
    add_common(inspect)
    inspect.set_defaults(func=cmd_inspect)

    simulate = subparsers.add_parser("simulate", help="run against mock hardware")
    simulate.add_argument("--duration", type=float, default=60.0, help="run time (s)")
    simulate.add_argument(
        "--drift-after", type=float, default=0.0,
        help="change the synthetic gait after this time (s) to trigger OOD",
    )
    simulate.add_argument("--stride-csv", default="", help="stride log destination")
    simulate.add_argument("--fast", action="store_true", help="do not pace the loop")
    add_common(simulate)
    simulate.set_defaults(func=cmd_simulate)

    live = subparsers.add_parser("live", help="run against the PAD device")
    live.add_argument("--port", help="serial port (auto-detected when omitted)")
    live.add_argument("--duration", type=float, default=None, help="run time (s)")
    live.add_argument("--stride-csv", default="", help="stride log destination")
    add_common(live)
    live.set_defaults(func=cmd_live)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
