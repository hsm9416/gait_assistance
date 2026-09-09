#!/usr/bin/env python3
"""Experiment runner. Edit the parameters below, then run by name.

    python exp.py            every offline run, then the summary
    python exp.py 0p4 1p0    only those recordings
    python exp.py noref      offline without the healthy reference (control)
    python exp.py step0      live: device connected, current clamped to zero
    python exp.py step1      live: worn, current barely perceptible
    python exp.py step4      live: minimum real assistance
    python exp.py sim:step1  same monitor against the mock device (no hardware)
    python exp.py summary    re-print the table
    python exp.py torque:step1   what the motor did, from the cycle log
"""
import csv as _csv
import sys, time, pathlib, pandas as pd
from gait_assistance.main import main as cli

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
OUT = "results"
HEALTHY_CSV = "csv/imu_hs_torque_20260826_110500.csv"

# A reference is only comparable to a run that used the SAME phase detector:
# swing_ratio from a fixed clock and from belt velocity are different
# quantities, and mixing them silently zeroes the swing_ratio deficit term.
# One reference file per detector, so the two can never be confused.
# The real swing/stance detector is merged later.  Until then the timing is
# set here, not measured: ScheduledPhaseDetector emits a heel strike every
# LIVE_PERIOD_S seconds and spends LIVE_SWING_RATIO of it in swing.  A clock
# always produces strides, so the baseline completes on schedule.
LIVE_DETECTOR = "scheduled"          # detector used by every live/sim stage
LIVE_PERIOD_S = 1.2                  # stride period, heel strike to heel strike
LIVE_SWING_RATIO = 0.40              # fraction of the period spent in swing

def ref_path(detector):
    """Healthy reference built with ``detector``."""
    return f"healthy_1p0_{detector}.npz"

BASELINE_STRIDES = 15      # 0.4 m/s only yields ~31 strides in 60 s
SWING_RATIO = 0.40
PLOTS = False
MONITOR_HZ = 10            # status-line refresh rate during live/sim runs
CYCLE_LOG = True           # also record every control cycle (force, current, target)

# Assist force = stiffness x gain x max_retraction.  max_retraction scales the
# assistance alone; stiffness also scales the spring pulling the belt back to
# baseline, which the wearer feels as resistance rather than help - so most of
# the increase goes into max_retraction.
#
# stiffness stays low until a run is observed with gain > 0.  Raised to 0.04 it
# turned the device into a stiff spring: with gain 0 the target sits on the
# baseline while the belt swings +-80 mm around it, so 0.04 x 97 mm = 3.9 N of
# pure resistance was produced in stance and swing alike, with zero assistance
# (measured: force +-5.6 N, 43.6% of cycles clipped by the current limit).
# Raise it again only after the gain is confirmed non-zero.
ASSIST = {
    "assist.max_retraction_mm": 100.0,     # displacement commanded at gain 1.0
    "assist.max_gain": 0.8,
    "impedance.stiffness_n_per_mm": 0.005,
    "impedance.damping_n_per_mms": 0.005,
    "impedance.current_limit_a": 1.0,      # hard ceiling on what reaches the motor
}

# The gain cannot rise until a model exists and the deficit persists.  A 60 s
# run at 30 baseline strides never left BASELINE_COLLECTION, so the assistance
# path was never reached at all and the gain stayed 0 for the whole run.
UNBLOCK = {
    "patient.baseline_strides": 15,                     # 18 s at a 1.2 s period
    # Default 3 was never satisfied: the observed deficits came in runs of two
    # (strides 30-31 and 40-41), and one in-range stride resets the streak.
    # 2 clears both of those episodes and still rejects a single noisy stride.
    "assist.required_consecutive_deficit_strides": 2,
}

# the dead swing_ratio term is zeroed: a fixed clock cannot produce a deficit
WEIGHTS = {"swing_ratio": 0.0, "belt_excursion": 1.0,
           "temporal_symmetry": 0.0, "trunk_compensation": 0.0}

# name -> (recording, measured stride period in s)
RUNS = {
    "0p4": ("csv/imu_hs_torque_20260826_105902.csv", 1.892),
    "0p6": ("csv/imu_hs_torque_20260826_110130.csv", 1.600),
    "0p8": ("csv/imu_hs_torque_20260826_110324.csv", 1.332),
    "1p0": ("csv/imu_hs_torque_20260826_110500.csv", 1.225),
}

# live stages: name -> (seconds, overrides).  Read the protocol before running.
LIVE = {
    "step0": (10, {"impedance.current_limit_a": 0.0}),
    "step1": (20, {"impedance.current_limit_a": 0.05}),
    "step4": (60, {"assist.max_gain": 0.1,
                   "impedance.current_limit_a": 0.3}),
    # step4 with the persistence gate lowered, so a two-stride deficit can
    # actually reach the motor.  Everything else comes from ASSIST: max_gain
    # 0.8, max_retraction 100 mm, stiffness 0.005, current ceiling 1.0 A.
    "step5": (180, dict(UNBLOCK)),
}

# --------------------------------------------------------------------------- #


def sets(**kw):
    """Turn parameters into the --set arguments the CLI expects."""
    import json
    out = []
    for k, v in kw.items():
        out += ["--set", f"{k}={json.dumps(v)}"]
    return out


def healthy(detector="scheduled"):
    """Build the healthy reference for ``detector`` if it is not on disk yet."""
    out = ref_path(detector)
    if not pathlib.Path(out).exists():
        print(f"[healthy] building {out} with the {detector} detector")
        cli(["healthy", HEALTHY_CSV, "--out", out,
             *sets(**{"phase.detector": detector,
                      "phase.scheduled_period_s": RUNS["1p0"][1]})])
    return out


def offline(name, out=OUT, ref=True):
    """Replay one recording."""
    csv, period = RUNS[name]
    print(f"\n=== {name}  {csv}  period={period}s  swing={SWING_RATIO}  "
          f"baseline={BASELINE_STRIDES}  ref={'yes' if ref else 'no'}")
    argv = ["offline", csv, "--out-dir", f"{out}/{name}"]
    if ref:
        argv += ["--healthy-ref", healthy("scheduled")]
    if not PLOTS:
        argv += ["--no-plot"]
    cli(argv + sets(**{"phase.scheduled_period_s": period,
                       "phase.scheduled_swing_ratio": SWING_RATIO,
                       "patient.baseline_strides": BASELINE_STRIDES,
                       "deviation.biomech_weights": WEIGHTS}))


def summary(out=OUT):
    """One line per run: what the deviation terms and the gain came out as."""
    for f in sorted(pathlib.Path(out).glob("*/stride_log.csv")):
        d = pd.read_csv(f)
        print(f"{f.parent.name:5s} n={len(d):3d} E_B={d.biomechanical_deviation.mean():.4f} "
              f"E_R={d.manifold_deviation.mean():.4f} gain_mean={d.assist_gain.mean():.4f} "
              f"gain_max={d.assist_gain.max():.4f} excursion={d.belt_excursion.mean():6.1f}mm "
              f"ood={int(d.ood.sum())}")


# --------------------------------------------------------------------------- #
# terminal monitor
# --------------------------------------------------------------------------- #


def _monitor(runtime, duration, cycle_writer=None):
    """Return an on_cycle callback printing one live status line.

    The line is rewritten in place; strides, state changes and faults scroll
    past above it so the run leaves a readable record.
    """
    t0 = time.monotonic()
    seen = {"strides": 0, "state": "", "faults": ""}

    def emit(text):
        sys.stdout.write("\r\033[K" + text + "\n")

    def on_cycle(out):
        t = time.monotonic() - t0
        state = runtime.states.state.name
        gain = out.target.assist_gain

        if cycle_writer is not None:
            cycle_writer.writerow([
                f"{t:.4f}", f"{out.sample.timestamp:.4f}", state,
                out.phase.phase.name,
                f"{out.phase.phase_progress:.4f}" if out.phase.phase_progress is not None else "",
                f"{out.sample.belt_length:.3f}", f"{out.sample.belt_velocity:.3f}",
                f"{out.target.target_belt_length:.3f}", f"{out.target.profile_value:.4f}",
                f"{gain:.4f}", f"{out.command.force_n:.4f}",
                f"{out.command.current_a:.4f}", int(out.command.saturated),
                out.safety.fault_names,
            ])

        if out.completed_stride is not None:
            seen["strides"] += 1
            emit(f"[{t:6.1f}s] stride {seen['strides']:3d} recorded  "
                 f"n={len(out.completed_stride.samples)} samples  gain={gain:.3f}")
        if state != seen["state"]:
            emit(f"[{t:6.1f}s] state -> {state}")
            seen["state"] = state
        if out.safety.fault_names != seen["faults"]:
            if out.safety.fault_names:
                emit(f"[{t:6.1f}s] FAULT {out.safety.fault_names}: {out.safety.message}")
            seen["faults"] = out.safety.fault_names

        if t - on_cycle.last < 1.0 / MONITOR_HZ:
            return
        on_cycle.last = t
        sys.stdout.write(
            f"\r\033[K[{t:6.1f}/{duration:5.1f}s] {state:20s} "
            f"{out.phase.phase.name:6s} "
            f"belt {out.sample.belt_length:+7.1f} -> {out.target.target_belt_length:+7.1f}mm  "
            f"F={out.command.force_n:+6.3f}N  I={out.command.current_a:+6.3f}A"
            f"{'!' if out.command.saturated else ' '}  gain={gain:.3f}  "
            f"strides={seen['strides']:3d}"
        )
        sys.stdout.flush()

    on_cycle.last = -1.0
    return on_cycle


def _run_monitored(stage, duration, overrides, simulated=False):
    """Build the runtime directly so the cycle callback can be attached."""
    from gait_assistance.config import RunMode
    from gait_assistance.main import _build_runtime, load_config
    from gait_assistance.sensors.encoder import MockMotor, PadMotor
    from gait_assistance.sensors.sensor_manager import (
        MockSensorSource, PadSensorSource, SensorManager,
    )
    from gait_assistance.utils.logger import StrideLogger

    config = load_config(None, [a for k, v in overrides.items()
                                for a in (f"{k}={_json(v)}",)])
    pathlib.Path("logs").mkdir(exist_ok=True)
    if simulated:
        config.mode = RunMode.SIMULATION
        motor = MockMotor(current_limit_a=config.impedance.current_limit_a)
        source = MockSensorSource(config.sensor, motor=motor)
    else:
        config.mode = RunMode.HARDWARE
        source = PadSensorSource(config.sensor)
        motor = None
    sensors = SensorManager(source, config.sensor)
    sensors.open()
    if motor is None:
        motor = PadMotor(source.controller,
                         current_limit_a=config.impedance.current_limit_a)
    logger = StrideLogger(f"logs/{stage}.csv")
    runtime = _build_runtime(config, sensors, motor, logger)

    cycle_file = cycle_writer = None
    if CYCLE_LOG:
        cycle_file = open(f"logs/{stage}_cycles.csv", "w", newline="")
        cycle_writer = _csv.writer(cycle_file)
        cycle_writer.writerow([
            "t_s", "timestamp", "state", "phase", "phase_progress",
            "belt_length", "belt_velocity", "target_belt_length", "profile",
            "assist_gain", "force_n", "current_a", "saturated", "faults",
        ])

    print(f"[{stage}] {duration:.0f}s  logs/{stage}.csv")
    for k, v in overrides.items():
        print(f"         {k} = {v}")
    print("         Ctrl+C = emergency stop\n")
    try:
        runtime.run(duration_s=duration, realtime=True,
                    on_cycle=_monitor(runtime, duration, cycle_writer))
    except KeyboardInterrupt:
        runtime.low.safety.trigger_emergency_stop("keyboard interrupt")
        runtime.stop()
        print("\n[info] interrupted by the operator")
    finally:
        logger.close()
        sensors.close()
        if cycle_file is not None:
            cycle_file.close()
    print(f"\n[{stage}] done -> logs/{stage}.csv"
          + (f" + logs/{stage}_cycles.csv" if CYCLE_LOG else ""))
    if CYCLE_LOG:
        torque(stage)


def _json(v):
    import json
    return json.dumps(v)


def live(stage, simulated=False):
    """Run one live stage against the worn device (or the mock, to rehearse)."""
    duration, overrides = LIVE[stage]
    detector = {"phase.detector": LIVE_DETECTOR,
                "phase.scheduled_period_s": LIVE_PERIOD_S,
                "phase.scheduled_swing_ratio": LIVE_SWING_RATIO,
                # a clock makes swing_ratio constant, so its deficit term can
                # only inject a fixed offset; belt_excursion drives the gain
                "deviation.biomech_weights": WEIGHTS}
    if stage != "step0":                      # step0 never builds a model
        detector["reference.path"] = healthy(LIVE_DETECTOR)
    _run_monitored(stage, duration, {**ASSIST, **detector, **overrides},
                   simulated=simulated)


def torque(stage):
    """What the motor actually did, from the cycle log."""
    f = pathlib.Path(f"logs/{stage}_cycles.csv")
    if not f.exists():
        return
    d = pd.read_csv(f)
    sw = d[d.phase == "SWING"]
    print(f"[{stage}] current  mean={d.current_a.mean():.3f}A  max={d.current_a.max():.3f}A  "
          f"saturated={100*d.saturated.mean():.1f}% of cycles")
    print(f"[{stage}] force    mean={d.force_n.mean():+.3f}N  max={d.force_n.max():+.3f}N")
    if len(sw):
        print(f"[{stage}] in swing current mean={sw.current_a.mean():.3f}A  "
              f"gain mean={sw.assist_gain.mean():.3f}  "
              f"assist displacement max={(sw.assist_gain * sw.profile).max() * ASSIST['assist.max_retraction_mm']:.1f}mm")


def main(names):
    if not names:
        names = list(RUNS) + ["summary"]
    for name in names:
        if name in RUNS:
            offline(name)
        elif name in LIVE:
            live(name)
        elif name == "noref":
            for r in RUNS:
                offline(r, out=f"{OUT}/noref", ref=False)
            summary(f"{OUT}/noref")
        elif name.startswith("sim:"):
            live(name[4:], simulated=True)
        elif name.startswith("torque:"):
            torque(name[7:])
        elif name == "summary":
            summary()
        else:
            sys.exit(f"unknown: {name}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv[1:])
