#!/usr/bin/env python3
"""Plot the live logs offline.  Needs the CSVs only, never the device.

    python plot_exp.py                    every stage of the newest run
    python plot_exp.py step5              one stage of the newest run
    python plot_exp.py step1 step5        several
    python plot_exp.py 20260910_1430    every stage of that run
    python plot_exp.py 20260910_1430 step5   one stage of that run

Each ``exp.py`` invocation writes its CSVs to ``logs/<date>_<time>/`` and points
``logs/latest`` at it, so a re-run never overwrites the previous test.  The
figures are written next to the CSVs they came from.

Two figures per stage:

    logs/<stage>_torque.png    from <stage>_cycles.csv, one row per control
                               cycle: belt target vs measured, gain, assist
                               displacement, current.  Says whether the motor
                               was driven and when.
    logs/<stage>_strides.png   from <stage>.csv, one row per stride: excursion
                               against the healthy interval, E_B, E_R, and the
                               gain before and after the persistence gate.
                               Says why the motor was driven.

``gait_assistance/plotting.py`` is the offline-replay figure and takes a
``SimulationResult`` object; it cannot read these logs.
"""
import pathlib
import sys

import os

import matplotlib
# Show the figures when there is a display; fall back to file-only over SSH or
# in CI, where selecting an interactive backend would fail on import.
HEADLESS = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp import ASSIST, LIVE, LOGS, find_log, ref_path, run_folders, stage_detector

ZOOM_S = 12.0          # width of the zoomed panel


def resolve(names):
    """Split the command line into a run folder and the stages to plot.

    Args:
        names: raw arguments - an optional run folder (or its name) followed by
            stage names; either part may be omitted.

    Returns:
        ``(run, pinned, stages)``.  ``pinned`` is True only when the run folder
        was named explicitly; otherwise the newest run is merely where the
        stage list comes from, and a stage missing from it is still looked up
        in older runs and in the flat pre-folder layout.
    """
    names = list(names)
    run, pinned = None, False
    if names:
        given = pathlib.Path(names[0])
        candidate = given if given.exists() else LOGS / names[0]
        if candidate.is_dir():
            run, pinned, names = candidate, True, names[1:]
    if run is None:
        folders = run_folders()
        run = folders[-1] if folders else None
    if not names:
        source = run if run is not None else LOGS
        names = sorted({f.name.split("_cycles")[0].removesuffix(".csv")
                        for f in source.glob("*.csv")})
    return run, pinned, names


def stage_params(stage):
    """Current limit and phase detector the stage actually ran with.

    The early stages of the protocol override the current limit - step0 to
    step3 clamp it to zero - so drawing the limit from ASSIST would put a
    1.0 A reference line on a run that could never exceed 0 A.

    Args:
        stage: stage name, e.g. ``"step3"``.

    Returns:
        ``(current_limit_a, detector_name)``, falling back to the ASSIST
        defaults for a log whose stage is no longer in ``LIVE``.
    """
    _duration, overrides = LIVE.get(stage, (0, {}))
    limit = overrides.get("impedance.current_limit_a",
                          ASSIST["impedance.current_limit_a"])
    return limit, stage_detector(overrides)


def healthy_interval(d, detector, metric):
    """Reference interval the strides in ``d`` were actually scored against.

    With a cadence-conditioned bank the interval is not a property of the run
    but of each stride's walking speed, so drawing the 1.0 m/s band under a
    slow walk would show a deficit the policy never saw.  The stride log
    records the cadence that scored it in ``reference_source``, e.g.
    ``healthy_reference:0p4``; the band drawn is the one most of the strides
    used.

    Args:
        d: stride log.
        detector: phase detector the stage ran with.
        metric: metric whose interval is wanted.

    Returns:
        ``(MetricRange, label)``, the label naming the cadence when the run used
        a bank and empty otherwise.
    """
    from gait_assistance.manifold.reference import ReferenceBank, load_reference

    labels = []
    if "reference_source" in d:
        labels = [s.split(":", 1)[1] for s in d.reference_source.dropna()
                  if isinstance(s, str) and ":" in s]
    if labels:
        chosen = max(set(labels), key=labels.count)
        bank = ReferenceBank.load(f"healthy_bank_{detector}.npz")
        share = 100 * labels.count(chosen) / len(labels)
        return (bank.references[chosen].metric_ranges[metric],
                f" of the {chosen} cadence ({share:.0f}% of strides)")
    ref = load_reference(ref_path(detector))
    if isinstance(ref, ReferenceBank):
        ref = ref.reference(d.estimated_stride_time.median()
                            if "estimated_stride_time" in d else None)
    return ref.metric_ranges[metric], ""


def torque_figure(stage, run=None):
    """Draw the sample-rate view: what the motor was told to do.

    Returns:
        The written path in a list, empty when the stage has no cycle log.
    """
    f = find_log(stage, "_cycles.csv", run)
    if f is None:
        print(f"[{stage}] no cycle log")
        return []
    d = pd.read_csv(f)
    swing = d.phase == "SWING"
    retract = ASSIST["assist.max_retraction_mm"]
    limit, _detector = stage_params(stage)
    clamped = limit == 0.0

    # centre the zoom on the strongest assistance; a stage with the motor
    # clamped off (step0-step3) has no such point, so show the middle instead
    assist = d.assist_gain * d.profile
    if assist.max() > 0:
        centre, label = d.t_s[assist.idxmax()], "around the strongest assist"
    else:
        centre, label = d.t_s.median(), "mid-run (this stage never assisted)"
    t0 = max(d.t_s.min(), centre - ZOOM_S / 2)
    windows = [(d.t_s.min(), d.t_s.max(), "whole run"),
               (t0, t0 + ZOOM_S, f"{ZOOM_S:.0f} s {label}")]

    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex="col")
    for col, (lo, hi, title) in enumerate(windows):
        w = d[(d.t_s >= lo) & (d.t_s <= hi)]
        sw = swing[w.index]

        axes[0, col].plot(w.t_s, w.belt_length, lw=0.8, label="measured")
        axes[0, col].plot(w.t_s, w.target_belt_length, lw=1.2, label="target")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("belt (mm)")
        axes[0, col].legend(fontsize=8, loc="upper right")

        axes[1, col].plot(w.t_s, w.assist_gain, lw=1.0, color="tab:green")
        axes[1, col].set_ylabel("assist gain")

        axes[2, col].plot(w.t_s, w.assist_gain * w.profile * retract,
                          lw=1.0, color="tab:purple")
        axes[2, col].set_ylabel("assist disp (mm)")

        axes[3, col].plot(w.t_s, w.current_a, lw=0.8, color="tab:red")
        axes[3, col].axhline(limit, ls="--", lw=0.8, color="gray")
        axes[3, col].set_ylabel("current (A)")
        axes[3, col].set_xlabel("time (s)")
        axes[3, col].annotate(
            "motor clamped off (limit 0 A)" if clamped else f"limit {limit:g} A",
            xy=(0.99, 0.92), xycoords="axes fraction", ha="right", fontsize=8,
            color="gray")

        # shade the swing intervals: assistance may only appear inside them
        for ax in axes[:, col]:
            ax.fill_between(w.t_s, *ax.get_ylim(), where=sw,
                            color="tab:blue", alpha=0.07, step="mid")
            ax.margins(x=0)

    headline = (f"{stage}: motor clamped off - the chain runs, nothing reaches "
                f"the wearer" if clamped else
                f"{stage}: assistance is applied in the shaded swing intervals only")
    fig.suptitle(headline)
    fig.tight_layout()
    out = f.parent / f"{stage}_torque.png"
    fig.savefig(out, dpi=110)
    if HEADLESS:
        plt.close(fig)

    stance_assist = int(((~swing) & (assist > 0)).sum())
    # At a 0 A limit every request is clipped, so the saturation figure would
    # read like a fault when it is the clamp doing its job.
    clip = ("motor clamped off" if clamped
            else f"saturated={100 * d.saturated.mean():.1f}%")
    print(f"[{stage}] -> {out}   {clip}  "
          f"assist during STANCE={stance_assist}"
          f"{'  <- should be 0' if stance_assist else '  (correct)'}")
    return [out]


def stride_figure(stage, run=None):
    """Draw the stride-rate view: why the motor was told to do it.

    Returns:
        The written path in a list, empty when the stage has no stride log.
    """
    f = find_log(stage, ".csv", run)
    if f is None:
        print(f"[{stage}] no stride log")
        return []
    d = pd.read_csv(f)
    x = d.stride_id
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    ax[0].plot(x, d.belt_excursion, "o-", ms=3, lw=0.8)
    try:
        _limit, detector = stage_params(stage)
        r, label = healthy_interval(d, detector, "belt_excursion")
        ax[0].axhspan(r.lower, r.upper, color="tab:green", alpha=0.12)
        ax[0].axhline(r.lower, color="tab:green", lw=0.8, ls="--")
        ax[0].set_title(f"{stage}: shaded band is the healthy interval{label}; "
                        "only a stride below it is a deficit")
    except Exception as exc:                       # the reference is optional
        ax[0].set_title(f"{stage}: no healthy interval to draw ({exc})")
    ax[0].set_ylabel("belt excursion\n(mm)")

    ax[1].plot(x, d.belt_excursion_error, "o-", ms=3, lw=0.8, color="tab:orange",
               label="belt_excursion_error")
    ax[1].plot(x, d.biomechanical_deviation, "o-", ms=3, lw=0.8, color="tab:red",
               label="E_B")
    ax[1].legend(fontsize=8)
    ax[1].set_ylabel("deficit")

    ax[2].plot(x, d.manifold_deviation, "o-", ms=3, lw=0.8, color="tab:purple")
    # a flagged stride only caps the gain once the flag has repeated
    # required_consecutive_ood_strides times, so the two are drawn apart;
    # ood_engaged is absent from logs recorded before that rule existed
    ood = d[d.ood == 1]
    engaged = d[d.ood_engaged == 1] if "ood_engaged" in d else d.iloc[:0]
    if len(ood):
        ax[2].plot(ood.stride_id, ood.manifold_deviation, "x", ms=5,
                   color="tab:orange", label="OOD flagged")
    if len(engaged):
        ax[2].plot(engaged.stride_id, engaged.manifold_deviation, "X", ms=8,
                   color="tab:red", label="OOD capping the gain")
    if len(ood) or len(engaged):
        ax[2].legend(fontsize=8)
    ax[2].set_ylabel("E_R (manifold)")

    ax[3].step(x, d.raw_assist_gain, where="post", lw=0.9, color="gray",
               label="raw (before the gate)")
    ax[3].step(x, d.assist_gain, where="post", lw=1.4, color="tab:green",
               label="applied")
    ax[3].legend(fontsize=8)
    ax[3].set_ylabel("assist gain")
    ax[3].set_xlabel("stride")

    fig.tight_layout()
    out = f.parent / f"{stage}_strides.png"
    fig.savefig(out, dpi=110)
    if HEADLESS:
        plt.close(fig)

    blocked = int(((d.raw_assist_gain > 0) & (d.assist_gain == 0)).sum())
    print(f"[{stage}] -> {out}   strides={len(d)}  "
          f"deficit={(d.biomechanical_deviation > 0).sum()}  "
          f"blocked by the gate={blocked}  "
          f"OOD flagged={int((d.ood == 1).sum())}"
          f"/capping={len(engaged)}")
    return [out]


def main(argv):
    run, pinned, stages = resolve(argv)
    if not stages:
        sys.exit(f"no logs in {run or LOGS}/")
    # Only a folder the caller named pins the search; otherwise a stage that
    # the newest run does not hold is still found in an older one.
    search = run if pinned else None
    print(f"[run] {run or LOGS}" + ("" if pinned else "  (newest; older runs searched as needed)"))
    written = []
    for stage in stages:
        written += torque_figure(stage, search) + stride_figure(stage, search)

    # every figure is on disk before anything is shown, so closing a window
    # (or interrupting the run) cannot lose one
    where = written[0].parent.resolve() if written else (run or LOGS).resolve()
    print(f"\nsaved {len(written)} figure(s) to {where}")
    for p in written:
        print(f"  {p.name}")
    if not HEADLESS:
        plt.show()          # blocks until every window is closed


if __name__ == "__main__":
    main(sys.argv[1:])
