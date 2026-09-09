#!/usr/bin/env python3
"""사전 보행 CSV에서 heel strike 타이밍 모델을 저장합니다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, median

from heelstrike_detect import (
    HS_LOADING_VEL,
    HS_MIN_BELT_MM,
    HS_REFRACTORY_S,
    HS_STRIKE_VEL,
    HeelStrikeDetector,
)


def detect_hs(csv_path: Path, min_belt: float, refractory: float, loading_vel: float, strike_vel: float) -> list[dict[str, float]]:
    detector = HeelStrikeDetector(min_belt, refractory, loading_vel, strike_vel)
    hits: list[dict[str, float]] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["host_time_s"])
            belt = float(row["belt_length"])
            vel = float(row["belt_velocity"])
            if detector.update(belt, vel, t):
                hits.append({"t": t, "belt": belt, "vel": vel})
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=Path("csv/TEST1.csv"))
    parser.add_argument("-o", "--out", type=Path, default=Path("hs_model.json"))
    parser.add_argument("--min-belt", type=float, default=HS_MIN_BELT_MM)
    parser.add_argument("--refractory", type=float, default=HS_REFRACTORY_S)
    parser.add_argument("--loading-vel", type=float, default=HS_LOADING_VEL)
    parser.add_argument("--strike-vel", type=float, default=HS_STRIKE_VEL)
    args = parser.parse_args()

    hits = detect_hs(args.csv, args.min_belt, args.refractory, args.loading_vel, args.strike_vel)
    if len(hits) < 2:
        raise SystemExit(f"HS가 부족합니다: {len(hits)}개")

    intervals = [hits[i]["t"] - hits[i - 1]["t"] for i in range(1, len(hits))]
    model = {
        "source_csv": str(args.csv),
        "hs_count": len(hits),
        "stride_s": median(intervals),
        "stride_mean_s": fmean(intervals),
        "min_belt_mm": args.min_belt,
        "refractory_s": args.refractory,
        "loading_vel": args.loading_vel,
        "strike_vel": args.strike_vel,
        "first_hs_s": hits[0]["t"],
        "last_hs_s": hits[-1]["t"],
    }

    args.out.write_text(json.dumps(model, indent=2) + "\n")
    print(f"[OK] HS {len(hits)}개, stride={model['stride_s']:.3f}s 저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
