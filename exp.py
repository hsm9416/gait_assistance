#!/usr/bin/env python3
"""Experiment runner. Edit the parameters below, then run by name.

    python exp.py            every offline run, then the summary
    python exp.py 0p4 1p0    only those recordings
    python exp.py noref      offline without the healthy reference (control)
    python exp.py step0      live: telemetry + safety            (motor off)
    python exp.py step1      live: phase + stride segmentation    (motor off)
    python exp.py step2      live: baseline + patient model       (motor off)
    python exp.py step3      live: deviation + gain, computed     (motor off)
    python exp.py step4      live: first current the wearer feels
    python exp.py step5      live: minimum real assistance
    python exp.py step6      live: full assistance
    python exp.py feel       live: bring-up, gain held at 0.8 so the force is
                             deterministic - use it when nothing is felt
    python exp.py sim:step3  same monitor against the mock device (no hardware)
    python exp.py summary    re-print the table

Every invocation that drives the device writes its CSVs to a fresh
``logs/<date>_<time>/`` folder and points ``logs/latest`` at it, so re-running a
stage never overwrites the previous test.  ``plot_exp.py`` reads the newest run
by default; pass a folder name to read an older one.

The live stages walk up the algorithm one layer at a time: step0-step3 run the
whole pipeline with the motor clamped to zero, so every layer is checked on the
wearer before any current can reach them.  Run them in order and do not skip.

A sim: run rehearses the plumbing, not the physiology.  The healthy reference
is built from a real recording, so the mock's synthetic gait sits far outside
it: expect the run to report OOD on almost every stride and a saturated
manifold deviation.  That is the mock, not the detector - the same stage
replayed offline against real recordings (`python exp.py 1p0`) puts the
deviation back near zero.
    python exp.py torque:step1   what the motor did, from the cycle log
"""
import csv as _csv
import sys, time, pathlib, pandas as pd
from gait_assistance.control.impedance_controller import IMPEDANCE_HEADROOM_N
from gait_assistance.manifold.reference import ReferenceBank
from gait_assistance.main import main as cli

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
OUT = "results"
HEALTHY_CSV = "csv/imu_hs_torque_20260826_110500.csv"

# A reference is only comparable to a run that used the SAME phase detector:
# swing_ratio from a fixed clock and from the belt are different quantities,
# and mixing them silently distorts the swing_ratio deficit term.  One
# reference file per detector, picked by ``stage_detector`` below, so the two
# can never be confused.
#
# ScheduledPhaseDetector emits a heel strike every LIVE_PERIOD_S seconds and
# spends LIVE_SWING_RATIO of it in swing: the timing is set here, not measured.
# A clock always produces strides, so the baseline completes on schedule, which
# is what makes it useful for the early stages where nothing may reach the
# motor anyway.  step6 replaces it with BeltLengthPhaseDetector, which reads
# belt_length and nothing else, so swing/stance follow the wearer.
LIVE_DETECTOR = "belt_cycle"           # default for stages that do not override it
OFFLINE_DETECTOR = "belt_cycle"     # detector every replay in RUNS uses
# only read when a stage selects "scheduled": the clock has no wearer to follow
LIVE_PERIOD_S = 1.2                  # stride period, heel strike to heel strike
LIVE_SWING_RATIO = 0.40              # fraction of the period spent in swing

def ref_path(detector):
    """Healthy reference built with ``detector``."""
    return f"healthy_1p0_{detector}.npz"

BASELINE_STRIDES = 15      # 0.4 m/s only yields ~31 strides in 60 s

# "bank"   one healthy reference per recorded cadence, selected per stride by
#          the stride's own duration.  A slow stride measured against a fast
#          reference reads as a deficit that is not there.
# "single" the old behaviour: the 1.0 m/s recording stands in for every speed.
REFERENCE = "bank"
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
    "impedance.damping_n_per_mms": 0.0015,
    "impedance.current_limit_a": 3.0,      # hard ceiling on what reaches the motor
    # A single OOD stride used to cap the gain at ood_max_gain (0.2), which the
    # wearer feels as the assistance cutting out mid-walk.  Wait for a run of
    # five: the model has then really lost the gait, not just one stride.
    "assist.required_consecutive_ood_strides": 5,
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

# A fixed clock makes swing_ratio identical on every stride, so its deficit
# term can never fire; carrying weight on it would silently drop that share of
# the score.  Zero it and let belt_excursion carry the whole thing.
WEIGHTS_SCHEDULED = {"swing_ratio": 0.0, "belt_excursion": 1.0,
                     "temporal_symmetry": 0.0, "trunk_compensation": 0.0}

# A signal-driven detector measures swing/stance per stride, so the term is
# live again and gets weight back.  belt_excursion keeps the larger share: it
# is the deficit the device can act on directly, while swing_ratio from a belt
# event sits on an uncalibrated scale (see the detector's docstring).
WEIGHTS_MEASURED = {"swing_ratio": 0.3, "belt_excursion": 0.7,
                    "temporal_symmetry": 0.0, "trunk_compensation": 0.0}


def stage_detector(overrides):
    """Phase detector a stage runs with: its own override, else the default."""
    return overrides.get("phase.detector", LIVE_DETECTOR)


def weights_for(detector):
    """Biomechanical weights matching what ``detector`` can actually measure."""
    return dict(WEIGHTS_SCHEDULED if detector == "scheduled" else WEIGHTS_MEASURED)

# name -> (recording, measured stride period in s)
RUNS = {
    "0p4": ("csv/imu_hs_torque_20260826_105902.csv", 1.892),
    "0p6": ("csv/imu_hs_torque_20260826_110130.csv", 1.600),
    "0p8": ("csv/imu_hs_torque_20260826_110324.csv", 1.332),
    "1p0": ("csv/imu_hs_torque_20260826_110500.csv", 1.225),
}

# Live stages: name -> (seconds, overrides).  Read the protocol before running.
#
# One stage of the algorithm per step, in the order the pipeline computes them.
# step0-step3 clamp the current to zero, so the whole chain runs and is logged
# while nothing reaches the wearer; only step4 onwards opens the motor, and it
# opens it in three graded increments.  A stage is finished when its own layer
# is confirmed in the log, not when the timer runs out.
#
#   layer                             step  what the log has to show
#   --------------------------------  ----  ------------------------------------
#   acquisition, safety               0     frames arriving, no faults
#   phase detection, segmentation     1     heel strikes, plausible stride times
#   features, SPD, log, baseline, K   2     "[model]" line, K and cluster sizes
#   state, OOD, deviation, gain       3     gain > 0 computed, OOD rate sane
#   belt target, impedance, current   4-6   current at the motor, graded
LIVE = {
    # --- motor clamped off: the chain is verified before it can act --------- #

    # Acquisition and safety only.  Nothing else is trusted until frames arrive
    # cleanly: no faults, no stale telemetry, belt and motor inside the envelope.
    "step0": (10, {"impedance.current_limit_a": 0.0}),

    # Phase detection and stride segmentation on this wearer.  What to read:
    # heel strikes appear at a plausible cadence, stride times are consistent,
    # and standing still produces no strides at all (the belt-length detector
    # gates on its own excursion amplitude).
    "step1": (60, {"impedance.current_limit_a": 0.0}),

    # Baseline collection and model building: features -> z-score -> SPD
    # covariance -> matrix log -> clustering.  Long enough to collect the
    # baseline strides and print the "[model]" summary with K and its clusters.
    "step2": (90, {**UNBLOCK, "impedance.current_limit_a": 0.0}),

    # Online classification, OOD, deviation and assist gain - all computed and
    # logged, none of it delivered.  This is where a gain of exactly zero, or an
    # OOD rate near 1.0, has to be understood BEFORE the motor is opened.
    "step3": (180, {**UNBLOCK, "impedance.current_limit_a": 0.0}),

    # --- motor open, in three graded increments ---------------------------- #
    #
    # What escalates here is impedance.assist_force_n - the peak force the
    # wearer feels at gain 1.0.
    #
    # The force has to be sized against max_gain, not against the gain a past
    # run happened to show.  The earlier 10/20/30 N were sized for the 0.15
    # gains measured at the time; once the policy started producing 0.59 the
    # same settings asked for 11.7 N = 13.5 A against a 4.0 A ceiling, and the
    # ceiling - not the profile - decided the output: 51% of the swing cycles
    # of logs/20260910_1037/step5 sat pinned at exactly 4.00 A, so the wearer
    # felt a square pulse with steep edges instead of the commanded sine.
    #
    # The values below come from ImpedanceController.assist_force_limit_n(0.8)
    # for each ceiling: assist_force_n x max_gain + 0.8 N of impedance stays
    # under the force the ceiling can pass, so the sine survives at every gain
    # the policy can produce.  _report_force_budget prints the margin at the
    # start of every stage, and exp.py warns when a setting cannot fit.
    #
    # Raising a ceiling is the only way to raise the peak force, and the device
    # clamps at 13 A = 11.3 N whatever this file says.
    #
    # These are starting points, not clinical values.  Go up one step at a
    # time, with the wearer able to stop, and stay at the lowest force that
    # does something useful.

    # First force the wearer can clearly feel.  Confirms the direction and the
    # timing of the pull, not its usefulness.
    "step4": (60, {**UNBLOCK,
                   "impedance.assist_force_n": 1.6,      # peak 1.3 N = 1.5 A
                   "impedance.current_limit_a": 2.5}),   # passes 2.1 N

    # The smallest force that actually assists the swing.
    "step5": (120, {**UNBLOCK,
                    "impedance.assist_force_n": 3.2,     # peak 2.6 N = 3.0 A
                    "impedance.current_limit_a": 4.0}),  # passes 3.4 N

    # Full assistance.
    "step6": (180, {**UNBLOCK,
                    "impedance.assist_force_n": 4.8,     # peak 3.8 N = 4.5 A
                    "impedance.current_limit_a": 5.5}),  # passes 4.7 N

    # --- bring-up, outside the graded protocol ------------------------------ #

    # "Can the wearer feel anything at all?"  The deficit-proportional gain
    # cannot answer that on a wearer with almost no deficit: measured gains sat
    # at 0.09-0.18 out of 0.8, so a 6 N setting delivered 1.1 N.  Here the gain
    # is HELD at 0.8, which takes the deviation score and the OOD cap out of
    # the loop, so the force is deterministic: 4.8 N x 0.8 x sin(pi x) -> 3.8 N
    # at the middle of every swing = 4.5 A, just under the 5.5 A ceiling, so
    # nothing is clipped and the shape is exactly the commanded sine.
    #
    # This is a diagnostic, not a therapy mode - it assists regardless of
    # whether the wearer needs it.  Use it to confirm the direction and the
    # magnitude, then go back to the steps.
    "feel": (45, {**UNBLOCK,
                  "assist.fixed_gain": 0.8,
                  "impedance.assist_force_n": 4.8,
                  "impedance.current_limit_a": 5.5}),
}

#: Stages that run before a patient model exists, so they load no reference.
PRE_MODEL = {"step0", "step1"}

# --------------------------------------------------------------------------- #


LOGS = pathlib.Path("logs")
_RUN_DIR = None            # this invocation's folder, created on first use


def run_dir():
    """Folder for this invocation's logs: ``logs/YYYYmmdd_HHMM/``.

    One folder per ``exp.py`` invocation, so every test keeps its own CSVs and
    a re-run can never overwrite the previous one.  A second run inside the
    same minute gets a ``-2`` suffix.  ``logs/latest`` is pointed at the
    folder, which is what the plotting and torque commands read by default.

    Returns:
        The created directory.
    """
    global _RUN_DIR
    if _RUN_DIR is None:
        stamp = time.strftime("%Y%m%d_%H%M")
        folder = LOGS / stamp
        # Minute resolution means two runs can fall in the same minute - step0
        # only takes ten seconds.  Never write into a folder that already holds
        # a test: suffix instead, so no recorded run is ever overwritten.
        suffix = 2
        while any(folder.glob("*.csv")):
            folder = LOGS / f"{stamp}-{suffix}"
            suffix += 1
        _RUN_DIR = folder
        _RUN_DIR.mkdir(parents=True, exist_ok=True)
        link = LOGS / "latest"
        try:                                   # a convenience, never fatal
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(_RUN_DIR.name)
        except OSError:
            pass
    return _RUN_DIR


def run_folders():
    """Every timestamped run folder under ``logs/``, oldest first."""
    # the trailing wildcard also picks up the "-2" suffix and the seconds-long
    # names written before the format was shortened
    return sorted(d for d in LOGS.glob("[0-9]" * 8 + "_" + "[0-9]" * 4 + "*")
                  if d.is_dir())


def find_log(stage, suffix=".csv", run=None):
    """Locate one stage's log.

    Args:
        stage: stage name, e.g. ``"step6"``.
        suffix: ``".csv"`` for the stride log, ``"_cycles.csv"`` for the
            cycle log.
        run: a specific run folder; the newest one that has the file is used
            when omitted.

    Returns:
        The path, or ``None`` when no run holds that file.  Logs written
        before the per-run folders existed are still found at ``logs/``.
    """
    name = f"{stage}{suffix}"
    if run is not None:
        f = pathlib.Path(run) / name
        return f if f.exists() else None
    for folder in reversed(run_folders()):
        f = folder / name
        if f.exists():
            return f
    flat = LOGS / name                          # pre-folder layout
    return flat if flat.exists() else None


def sets(**kw):
    """Turn parameters into the --set arguments the CLI expects."""
    import json
    out = []
    for k, v in kw.items():
        out += ["--set", f"{k}={json.dumps(v)}"]
    return out


def healthy(detector="scheduled"):
    """Build the single 1.0 m/s healthy reference if it is not on disk yet."""
    out = ref_path(detector)
    if not pathlib.Path(out).exists():
        print(f"[healthy] building {out} with the {detector} detector")
        cli(["healthy", HEALTHY_CSV, "--out", out,
             *sets(**{"phase.detector": detector,
                      "phase.scheduled_period_s": RUNS["1p0"][1]})])
    return out


def healthy_bank(detector="belt_cycle"):
    """Build the cadence-conditioned reference bank if it is not on disk yet.

    One reference per recorded walking speed, all four from the same person.  A
    stride is then scored against the cadence it was actually walked at, which
    is what the single 1.0 m/s reference could not do: the 14:42 run (stride
    1.76 s) measured E_B = 0.105 against 1p0 and 0.004 against 0p4 - the whole
    "deficit" was the speed mismatch.

    Args:
        detector: phase detector the references must be built with; a reference
            is only comparable to a run that used the same one.

    Returns:
        Path of the bank archive.
    """
    out = f"healthy_bank_{detector}.npz"
    if pathlib.Path(out).exists():
        return out
    paths = {}
    for name, (csv, period) in RUNS.items():
        one = f"healthy_{name}_{detector}.npz"
        if not pathlib.Path(one).exists():
            print(f"[bank] building {one}")
            cli(["healthy", csv, "--out", one,
                 *sets(**{"phase.detector": detector,
                          "phase.scheduled_period_s": period})])
        paths[name] = one
    bank = ReferenceBank.from_files(paths)
    bank.save(out)
    print(f"[bank] {out}\n" + bank.describe())
    return out


def reference_for(detector):
    """Reference path a run should use: the bank unless REFERENCE says single."""
    return healthy(detector) if REFERENCE == "single" else healthy_bank(detector)


def offline(name, out=OUT, ref=True):
    """Replay one recording."""
    csv, period = RUNS[name]
    print(f"\n=== {name}  {csv}  detector={OFFLINE_DETECTOR}  period={period}s  "
          f"swing={SWING_RATIO}  baseline={BASELINE_STRIDES}  "
          f"ref={'yes' if ref else 'no'}")
    argv = ["offline", csv, "--out-dir", f"{out}/{name}"]
    if ref:
        argv += ["--healthy-ref", reference_for(OFFLINE_DETECTOR)]
    if not PLOTS:
        argv += ["--no-plot"]
    # The detector is pinned rather than left to the package default: the
    # reference above was built with OFFLINE_DETECTOR, and a run on a different
    # detector would be compared against a reference it does not match.
    cli(argv + sets(**{"phase.detector": OFFLINE_DETECTOR,
                       "phase.scheduled_period_s": period,
                       "phase.scheduled_swing_ratio": SWING_RATIO,
                       "patient.baseline_strides": BASELINE_STRIDES,
                       "deviation.biomech_weights": weights_for(OFFLINE_DETECTOR)}))


def summary(out=OUT):
    """One line per run: what the deviation terms and the gain came out as."""
    for f in sorted(pathlib.Path(out).glob("*/stride_log.csv")):
        d = pd.read_csv(f)
        print(f"{f.parent.name:5s} n={len(d):3d} E_B={d.biomechanical_deviation.mean():.4f} "
              f"E_R={d.manifold_deviation.mean():.4f} gain_mean={d.assist_gain.mean():.4f} "
              f"gain_max={d.assist_gain.max():.4f} excursion={d.belt_excursion.mean():6.1f}mm "
              f"ood={int(d.ood.sum())}"
              f"/capping={int(d.ood_engaged.sum()) if 'ood_engaged' in d else 0}")


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
                f"{out.command.assist_force_n:.4f}",
                f"{out.command.impedance_force_n:.4f}",
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
    out_dir = run_dir()
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
    stride_csv = out_dir / f"{stage}.csv"
    cycles_csv = out_dir / f"{stage}_cycles.csv"
    logger = StrideLogger(str(stride_csv))
    runtime = _build_runtime(config, sensors, motor, logger)

    cycle_file = cycle_writer = None
    if CYCLE_LOG:
        cycle_file = open(cycles_csv, "w", newline="")
        cycle_writer = _csv.writer(cycle_file)
        cycle_writer.writerow([
            "t_s", "timestamp", "state", "phase", "phase_progress",
            "belt_length", "belt_velocity", "target_belt_length", "profile",
            "assist_gain", "force_n", "assist_force_n", "impedance_force_n",
            "current_a", "saturated", "faults",
        ])

    print(f"[{stage}] {duration:.0f}s  {stride_csv}")
    for k, v in overrides.items():
        print(f"         {k} = {v}")
    _report_force_budget(stage, runtime.low.impedance, config)
    print("         Ctrl+C = emergency stop\n")
    try:
        runtime.run(duration_s=duration, realtime=True,
                    on_cycle=_monitor(runtime, duration, cycle_writer))
    except KeyboardInterrupt:
        runtime.low.safety.trigger_emergency_stop("keyboard interrupt")
        runtime.stop()
        print("\n[info] interrupted by the operator")
    except OSError as exc:
        # the serial link died mid-run (USB unplugged, device reset): every
        # later write fails with Errno 5.  runtime.run() has already stopped
        # the loop in its own finally, so there is nothing left to command -
        # report it plainly and keep the log written up to that point.
        print(f"\n[error] the device link failed mid-run: {exc}")
        print("        the log is kept up to that point; check the USB cable "
              "and the device power, then re-run the stage")
    finally:
        # every step has to run even when an earlier one fails: a dead link
        # makes sensors.close() raise, and skipping cycle_file.close() would
        # lose the buffered tail of the very run that needs looking at
        problems = _close_each(
            ("stride log", logger.close),
            ("sensor link", sensors.close),
            *((("cycle log", cycle_file.close),) if cycle_file is not None else ()),
        )
        problems += sensors.close_errors
        problems += [f"motor {e}" for e in runtime.low.stop_errors]
        if problems:
            print("\n[warn] the shutdown did not complete cleanly:")
            for problem in problems:
                print(f"         {problem}")
            if runtime.low.stop_errors:
                print("         the motor was NOT confirmed at 0 A - power the "
                      "device off before taking it off")
    print(f"\n[{stage}] done -> {stride_csv}"
          + (f" + {cycles_csv}" if CYCLE_LOG else ""))
    if CYCLE_LOG:
        torque(stage, run=out_dir)


def _report_force_budget(stage, impedance, config):
    """State up front whether the force command can leave the amplifier.

    ``assist_force_n`` and ``current_limit_a`` are set independently, and a
    force above what the ceiling passes does not assist harder: the ceiling
    takes over at the peak of the swing profile, so the sine is delivered as a
    plateau with steep edges.  Printing the margin before the stage runs makes
    that visible in the terminal instead of only in the log afterwards.

    Args:
        stage: stage name, for the warning line.
        impedance: the controller the run will use, which owns the force map.
        config: the resolved configuration of this stage.
    """
    limit = config.impedance.current_limit_a
    if limit <= 0.0:                       # step0-step3: the motor is clamped
        return
    gain = config.assist.fixed_gain or config.assist.max_gain
    force = config.impedance.assist_force_n
    peak = force * gain + IMPEDANCE_HEADROOM_N
    ceiling = impedance.max_force_n()
    print(f"         force budget: {force:.1f}N x gain {gain:.2f} + "
          f"{IMPEDANCE_HEADROOM_N:.1f}N impedance = {peak:.2f}N  vs  "
          f"{ceiling:.2f}N passed by {limit:.1f}A"
          f"  -> {'fits' if peak <= ceiling + 1e-9 else 'WILL BE CLIPPED'}")
    if peak > ceiling + 1e-9:
        print(f"[warn] {stage}: the swing profile will be delivered as a "
              f"plateau at {limit:.1f} A, not as the commanded sine; "
              f"assist_force_n <= {impedance.assist_force_limit_n(gain):.1f} N "
              f"keeps its shape")


def _close_each(*steps):
    """Run every shutdown step, even when an earlier one raises.

    Args:
        steps: ``(label, callable)`` pairs, in shutdown order.

    Returns:
        ``["label: error", ...]`` for the steps that failed; empty when the
        shutdown was clean.
    """
    problems = []
    for label, call in steps:
        try:
            call()
        except Exception as exc:
            problems.append(f"{label}: {exc}")
    return problems


def _json(v):
    import json
    return json.dumps(v)


def live(stage, simulated=False):
    """Run one live stage against the worn device (or the mock, to rehearse)."""
    duration, overrides = LIVE[stage]
    name = stage_detector(overrides)
    detector = {"phase.detector": name,
                "phase.scheduled_period_s": LIVE_PERIOD_S,
                "phase.scheduled_swing_ratio": LIVE_SWING_RATIO,
                "deviation.biomech_weights": weights_for(name)}
    if stage not in PRE_MODEL:                # step0/step1 build no model
        # the reference has to come from the same detector as the run
        detector["reference.path"] = reference_for(name)
    _run_monitored(stage, duration, {**ASSIST, **detector, **overrides},
                   simulated=simulated)


def torque(stage, run=None):
    """What the motor actually did, from the cycle log.

    Args:
        stage: stage name.
        run: run folder to read; the most recent run holding the log is used
            when omitted.
    """
    f = find_log(stage, "_cycles.csv", run)
    if f is None:
        print(f"[{stage}] no cycle log")
        return
    d = pd.read_csv(f)
    sw = d[d.phase == "SWING"]
    # With the current limit at zero (step0-step3) every request is clipped, so
    # the saturation figure would read like a fault when it is the clamp doing
    # its job.  Say which one it is.
    clamped = float(d.current_a.abs().max()) == 0.0
    if clamped:
        print(f"[{stage}] motor clamped off - no current reached the wearer "
              f"({100*d.saturated.mean():.1f}% of cycles hit the zero clamp)")
    else:
        print(f"[{stage}] current  mean={d.current_a.mean():.3f}A  max={d.current_a.max():.3f}A  "
              f"saturated={100*d.saturated.mean():.1f}% of cycles")
    print(f"[{stage}] force    mean={d.force_n.mean():+.3f}N  max={d.force_n.max():+.3f}N")
    if "assist_force_n" in d.columns and not clamped:
        total = d.assist_force_n.abs().mean() + d.impedance_force_n.abs().mean()
        share = 100 * d.assist_force_n.abs().mean() / total if total else 0.0
        print(f"[{stage}] of the force asked for, {share:.1f}% was assistance "
              f"and {100 - share:.1f}% was the impedance reacting to the wearer")
    if len(sw) and not clamped:
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
