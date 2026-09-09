#!/usr/bin/env python3
"""Plot the live logs offline.  Needs the CSVs only, never the device.

    python plot_exp.py                 every stage that has a log
    python plot_exp.py step5           one stage
    python plot_exp.py step1 step5     several

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

from exp import ASSIST, LIVE_DETECTOR, ref_path

LOGS = pathlib.Path("logs")
ZOOM_S = 12.0          # width of the zoomed panel


def torque_figure(stage):
    """Draw the sample-rate view: what the motor was told to do.

    Returns:
        The written path in a list, empty when the stage has no cycle log.
    """
    f = LOGS / f"{stage}_cycles.csv"
    if not f.exists():
        print(f"[{stage}] no cycle log")
        return []
    d = pd.read_csv(f)
    swing = d.phase == "SWING"
    retract = ASSIST["assist.max_retraction_mm"]
    limit = ASSIST["impedance.current_limit_a"]

    # centre the zoom on the strongest assistance; a stage that never assisted
    # (step0, step1) has no such point, so show the middle of the run instead
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

        # shade the swing intervals: assistance may only appear inside them
        for ax in axes[:, col]:
            ax.fill_between(w.t_s, *ax.get_ylim(), where=sw,
                            color="tab:blue", alpha=0.07, step="mid")
            ax.margins(x=0)

    fig.suptitle(f"{stage}: assistance is applied in the shaded swing intervals only")
    fig.tight_layout()
    out = LOGS / f"{stage}_torque.png"
    fig.savefig(out, dpi=110)
    if HEADLESS:
        plt.close(fig)

    stance_assist = int(((~swing) & (assist > 0)).sum())
    print(f"[{stage}] -> {out}   saturated={100 * d.saturated.mean():.1f}%  "
          f"assist during STANCE={stance_assist}"
          f"{'  <- should be 0' if stance_assist else '  (correct)'}")
    return [out]


def stride_figure(stage):
    """Draw the stride-rate view: why the motor was told to do it.

    Returns:
        The written path in a list, empty when the stage has no stride log.
    """
    f = LOGS / f"{stage}.csv"
    if not f.exists():
        print(f"[{stage}] no stride log")
        return []
    d = pd.read_csv(f)
    x = d.stride_id
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    ax[0].plot(x, d.belt_excursion, "o-", ms=3, lw=0.8)
    try:
        from gait_assistance.manifold.reference import HealthyReference
        r = HealthyReference.load(ref_path(LIVE_DETECTOR)).metric_ranges["belt_excursion"]
        ax[0].axhspan(r.lower, r.upper, color="tab:green", alpha=0.12)
        ax[0].axhline(r.lower, color="tab:green", lw=0.8, ls="--")
        ax[0].set_title(f"{stage}: shaded band is the healthy interval; "
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
    ood = d[d.ood == 1]
    if len(ood):
        ax[2].plot(ood.stride_id, ood.manifold_deviation, "x", color="tab:red",
                   label="OOD")
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
    out = LOGS / f"{stage}_strides.png"
    fig.savefig(out, dpi=110)
    if HEADLESS:
        plt.close(fig)

    blocked = int(((d.raw_assist_gain > 0) & (d.assist_gain == 0)).sum())
    print(f"[{stage}] -> {out}   strides={len(d)}  "
          f"deficit={(d.biomechanical_deviation > 0).sum()}  "
          f"blocked by the gate={blocked}")
    return [out]


def main(stages):
    if not stages:
        stages = sorted({f.name.split("_cycles")[0].removesuffix(".csv")
                         for f in LOGS.glob("*.csv")})
        if not stages:
            sys.exit(f"no logs in {LOGS}/")
    LOGS.mkdir(exist_ok=True)
    written = []
    for stage in stages:
        written += torque_figure(stage) + stride_figure(stage)

    # every figure is on disk before anything is shown, so closing a window
    # (or interrupting the run) cannot lose one
    print(f"\nsaved {len(written)} figure(s) to {LOGS.resolve()}")
    for p in written:
        print(f"  {p.name}")
    if not HEADLESS:
        plt.show()          # blocks until every window is closed


if __name__ == "__main__":
    main(sys.argv[1:])
