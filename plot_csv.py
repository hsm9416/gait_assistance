#!/usr/bin/env python3
"""저장된 heelstrike CSV를 빠르게 확인하는 플롯."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from heelstrike_detect import HeelStrikeDetector
from heelstrike_impedance import (
    BIAS_N,
    D_N_PER_MMS,
    K_N_PER_MM,
    LOADING_VEL,
    MAX_FORCE_N,
    MIN_BELT_MM,
    REFRACTORY_S,
    STRIKE_VEL,
    WINDOW_S,
    X_REF_OFFSET_MM,
)


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
        if key != "assist_active"
    }


def replay(data: dict[str, list[float]]) -> tuple[list[int], list[float]]:
    detector = HeelStrikeDetector(MIN_BELT_MM, REFRACTORY_S, LOADING_VEL, STRIKE_VEL)
    hs_count = 0
    last_hs_t = -999.0
    x_ref = 0.0
    counts: list[int] = []
    forces: list[float] = []

    for t, belt, vel in zip(data["host_time_s"], data["belt_length"], data["belt_velocity"]):
        if detector.update(belt, vel, t):
            last_hs_t = t
            x_ref = belt + X_REF_OFFSET_MM
            hs_count += 1
        if t - last_hs_t > WINDOW_S:
            force = 0.0
        else:
            force = K_N_PER_MM * (x_ref - belt) + D_N_PER_MMS * (-vel) + BIAS_N
            force = max(0.0, min(MAX_FORCE_N, force))
        counts.append(hs_count)
        forces.append(force)
    return counts, forces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=Path("csv/TEST1.csv"))
    args = parser.parse_args()

    data = read_csv(args.csv)
    t = data["host_time_s"]
    replay_hs_count, replay_force = replay(data)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(t, data["belt_length"], label="belt_length")
    axes[0].axhline(-130.0, color="orange", linestyle="--", linewidth=1, label="HS threshold")
    axes[0].set_ylabel("mm")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, data["belt_velocity"], label="belt_velocity")
    axes[1].axhline(10.0, color="green", linestyle="--", linewidth=1)
    axes[1].axhline(-5.0, color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("mm/s")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, replay_hs_count, label="replay_hs_count")
    axes[2].plot(t, replay_force, label="replay_force")
    axes[2].set_ylabel("count / N")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, data["assist_current"], label="assist_current")
    axes[3].plot(t, data["output_force"], label="output_force")
    axes[3].set_ylabel("A / N")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(str(args.csv))
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
