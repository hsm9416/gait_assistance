#!/usr/bin/env python3
"""Heel strike 감지 후 짧게 impedance assist를 거는 예제."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time

from connect import PadController, PadSensorData
from heelstrike_detect import HeelStrikeDetector
from pad_external_control_lib import force_to_current_a, list_serial_ports


# Detection
MIN_BELT_MM = -130.0
REFRACTORY_S = 0.5
LOADING_VEL = 10.0
STRIKE_VEL = -5.0

# Control: F = K * (x_ref - x) + D * (0 - v) + BIAS
WINDOW_S = 0.35
X_REF_OFFSET_MM = 20.0
K_N_PER_MM = 0.5
D_N_PER_MMS = 0.01
BIAS_N = 0.0
MAX_FORCE_N = 30.0
CURRENT_LIMIT_A = 8.0
CSV_DIR = Path("csv")


def next_csv_path() -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    i = 1
    while (CSV_DIR / f"TEST{i}.csv").exists():
        i += 1
    return CSV_DIR / f"TEST{i}.csv"


def auto_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("연결된 시리얼 포트가 없습니다.")
    print(f"[INFO] 포트 자동 선택: {ports[0]}")
    return ports[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-d", "--duration", type=float, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-c", "--csv", type=Path, default=None, help="CSV 저장 경로. 생략하면 csv/TEST1.csv부터 자동 저장")
    args = parser.parse_args()

    detector = HeelStrikeDetector(MIN_BELT_MM, REFRACTORY_S, LOADING_VEL, STRIKE_VEL)
    t0 = time.monotonic()
    last_hs_t = -999.0
    hs_count = 0
    x_ref = 0.0
    last_force = 0.0
    last_current = 0.0
    csv_file = None
    csv_writer = None

    def algo(data: PadSensorData, _: PadController) -> float:
        nonlocal last_hs_t, hs_count, x_ref, last_force, last_current

        now = time.monotonic() - t0
        if detector.update(data.belt_length, data.belt_velocity, now):
            last_hs_t = now
            x_ref = data.belt_length + X_REF_OFFSET_MM
            hs_count += 1
            print(f"[HS ] #{hs_count} t={now:.2f}s x_ref={x_ref:.1f}mm", flush=True)

        if now - last_hs_t > WINDOW_S:
            last_force = 0.0
            last_current = 0.0
            return 0.0

        force = K_N_PER_MM * (x_ref - data.belt_length) + D_N_PER_MMS * (-data.belt_velocity) + BIAS_N
        last_force = max(0.0, min(MAX_FORCE_N, force))
        last_current = force_to_current_a(last_force)
        return last_current

    def on_data(data: PadSensorData, _: PadController) -> None:
        now = time.monotonic() - t0
        active = "ON " if last_current > 0.0 else "OFF"
        if csv_writer is not None:
            row = asdict(data)
            row.update(
                host_time_s=now,
                assist_active=last_current > 0.0,
                hs_count=hs_count,
                x_ref=x_ref,
                assist_force=last_force,
                assist_current=last_current,
            )
            csv_writer.writerow(row)
            if csv_file is not None:
                csv_file.flush()
        if not args.quiet:
            print(
                f"[TEL] {active} belt={data.belt_length:7.1f}mm "
                f"vel={data.belt_velocity:+7.1f}mm/s "
                f"F={last_force:5.2f}N I={last_current:4.2f}A hs={hs_count}",
                flush=True,
            )

    ctrl = PadController(args.port or auto_port(), baudrate=args.baud)
    try:
        csv_path = args.csv or next_csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        try:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "host_time_s",
                    *PadSensorData.__dataclass_fields__.keys(),
                    "assist_active",
                    "hs_count",
                    "x_ref",
                    "assist_force",
                    "assist_current",
                ],
            )
            csv_writer.writeheader()
            print(f"[INFO] CSV 저장: {csv_path}")
        except Exception:
            csv_file.close()
            raise
        ctrl.open()
        ctrl.run_control_loop(
            algo,
            duration_s=args.duration,
            current_limit_a=CURRENT_LIMIT_A,
            on_data=on_data,
        )
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        ctrl.close()
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
