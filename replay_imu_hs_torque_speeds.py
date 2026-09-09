#!/usr/bin/env python3
"""Replay four treadmill recordings with the current IMU torque algorithm."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import shutil

from imu_hs_torque import (
    AdaptiveImuHeelStrikeDetector,
    FIELDS,
    NOMINAL_GAIT_CYCLE_S,
    gait_cycle_pulse_s,
    plot_csv,
    should_apply_torque,
)


CSV_DIR = Path("csv")
OUTPUT_DIR = CSV_DIR / "imu_hs_torque_expected_20260826"
EXPERIMENTS = (
    ("0p4", CSV_DIR / "imu_hs_torque_20260826_105902.csv"),
    ("0p6", CSV_DIR / "imu_hs_torque_20260826_110130.csv"),
    ("0p8", CSV_DIR / "imu_hs_torque_20260826_110324.csv"),
    ("1p0", CSV_DIR / "imu_hs_torque_20260826_110500.csv"),
)


def replay(source_path: Path, output_csv: Path) -> tuple[int, int]:
    with source_path.open(newline="") as source:
        rows = list(csv.DictReader(source))

    detector = AdaptiveImuHeelStrikeDetector(
        calibration_s=6.0,
        acc_threshold=0.2,
        gyro_threshold=1.0,
        belt_threshold=-160.0,
        belt_side="left",
        use_gyro_gate=False,
        use_acc_peak=False,
        acc_percentile=87.0,
    )
    t0 = float(rows[0]["host_time_s"])
    torque_until = -999.0
    last_assist_t = -999.0
    assist_count = 0

    with output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            now = float(row["host_time_s"]) - t0
            gyro = tuple(float(row[key]) for key in ("gyro_x", "gyro_y", "gyro_z"))
            accel = tuple(float(row[key]) for key in ("accel_x", "accel_y", "accel_z"))
            belt = float(row["belt_length"])
            imu_hs, side, gyro_corrected, residual, belt_ok = detector.update(
                now, gyro, accel, belt
            )
            gait_cycle_s = 2.0 * detector.step_period_s
            cadence_spm = 60.0 / detector.step_period_s
            pulse_s = gait_cycle_pulse_s(detector.step_period_s, 0.18, 0.10, 0.30)
            hs_for_torque = should_apply_torque(
                imu_hs,
                side,
                "left",
                now,
                last_assist_t,
                gait_cycle_s,
            )
            if hs_for_torque:
                assist_count += 1
                last_assist_t = now
                torque_until = now + pulse_s
            torque = 1.5 if now < torque_until else 0.0

            output_row = {key: row.get(key, "") for key in FIELDS}
            output_row.update(
                {
                    "host_time_s": now,
                    "acc_mag": math.sqrt(sum(value * value for value in accel)),
                    "gyro_mag": math.sqrt(sum(value * value for value in gyro)),
                    "gyro_corrected": gyro_corrected,
                    "gyro_side_score": detector.gyro_side_score,
                    "gyro_side_separation": 2.0 * detector.gyro_side_scale,
                    "dynamic_refractory_s": detector.dynamic_refractory_s,
                    "step_period_s": detector.step_period_s,
                    "gait_cycle_s": gait_cycle_s,
                    "cadence_spm": cadence_spm,
                    "gait_speed_ratio": NOMINAL_GAIT_CYCLE_S / gait_cycle_s,
                    "assist_pulse_s": pulse_s,
                    "acc_residual": residual,
                    "acc_impact_peak": detector.acc_peak_value if imu_hs else 0.0,
                    "acc_peak_rise_s": detector.acc_peak_rise_s if imu_hs else 0.0,
                    "acc_peak_confirm_delay_s": detector.acc_peak_delay_s if imu_hs else 0.0,
                    "acc_impact_time_s": detector.acc_impact_time_s if imu_hs else -1.0,
                    "acc_peak_candidate": 1 if detector.acc_peak_candidate else 0,
                    "threshold": detector.acc_threshold,
                    "acc_threshold_percentile": detector.acc_percentile,
                    "gyro_threshold": detector.gyro_threshold,
                    "belt_threshold": detector.belt_threshold,
                    "belt_ok": 1 if belt_ok else 0,
                    "belt_ready": 1 if detector.belt_rearmed else 0,
                    "calibrated": 1 if detector.ready else 0,
                    "imu_side": side,
                    "imu_heel_strike": 1 if imu_hs else 0,
                    "heel_strike": 1 if hs_for_torque else 0,
                    "hs_count": detector.count,
                    "left_hs_count": detector.left_count,
                    "right_hs_count": detector.right_count,
                    "torque_cmd_nm": torque,
                }
            )
            writer.writerow(output_row)

    return detector.count, assist_count


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for speed, source_path in EXPERIMENTS:
        raw_path = OUTPUT_DIR / f"raw_{speed}_mps_{source_path.name}"
        replay_path = OUTPUT_DIR / f"expected_{speed}_mps.csv"
        png_path = OUTPUT_DIR / f"expected_{speed}_mps.png"
        shutil.copy2(source_path, raw_path)
        hs_count, assist_count = replay(source_path, replay_path)
        plot_csv(
            replay_path,
            png_path,
            show=False,
            title=f"{speed.replace('p', '.')} m/s | raw: {source_path.name}",
        )
        replay_path.unlink()
        print(
            f"[INFO] {speed.replace('p', '.')} m/s: "
            f"HS={hs_count}, assist={assist_count}, raw={raw_path}, plot={png_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
