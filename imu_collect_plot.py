#!/usr/bin/env python3
"""Collect PAD IMU telemetry to CSV and plot it."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from connect import PadController
from pad_external_control_lib import list_serial_ports


CSV_DIR = Path("csv")
FIELDS = [
    "host_time_s",
    "seq",
    "device_time_s",
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
]


def auto_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("연결된 시리얼 포트가 없습니다.")
    print(f"[INFO] 포트 자동 선택: {ports[0]}")
    return ports[0]


def default_csv_path() -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    return CSV_DIR / f"imu_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def collect_imu(port: str, baud: int, duration_s: float, csv_path: Path, quiet: bool) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    t0 = time.monotonic()

    with csv_path.open("w", newline="") as f, PadController(port, baudrate=baud) as ctrl:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        if not ctrl.start_session():
            raise RuntimeError("PAD 세션 시작 실패")

        print(f"[INFO] {duration_s:.1f}초 IMU 수집 시작: {csv_path}")
        while time.monotonic() - t0 < duration_s:
            data = ctrl.poll()
            if data is None:
                time.sleep(0.002)
                continue

            now = time.monotonic() - t0
            writer.writerow(
                {
                    "host_time_s": now,
                    "seq": data.seq,
                    "device_time_s": data.time_s,
                    "roll": data.roll,
                    "pitch": data.pitch,
                    "yaw": data.yaw,
                    "gyro_x": data.gyro_x,
                    "gyro_y": data.gyro_y,
                    "gyro_z": data.gyro_z,
                    "accel_x": data.accel_x,
                    "accel_y": data.accel_y,
                    "accel_z": data.accel_z,
                }
            )
            count += 1

            if not quiet:
                print(
                    f"[IMU] t={now:6.2f}s "
                    f"rpy=({data.roll:7.2f}, {data.pitch:7.2f}, {data.yaw:7.2f}) "
                    f"gyro=({data.gyro_x:7.2f}, {data.gyro_y:7.2f}, {data.gyro_z:7.2f}) "
                    f"acc=({data.accel_x:7.3f}, {data.accel_y:7.3f}, {data.accel_z:7.3f})",
                    flush=True,
                )

        ctrl.stop_session(timeout_s=1.0)

    print(f"[INFO] 저장 완료: {csv_path} ({count} samples)")
    return count


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV에 데이터가 없습니다: {path}")
    return {
        key: [float(row[key]) for row in rows]
        for key in FIELDS
        if key != "seq"
    }


def plot_imu(csv_path: Path, png_path: Path | None = None, show: bool = True) -> Path:
    data = read_csv(csv_path)
    t = data["host_time_s"]
    png_path = png_path or csv_path.with_suffix(".png")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(t, data["roll"], label="roll")
    axes[0].plot(t, data["pitch"], label="pitch")
    axes[0].plot(t, data["yaw"], label="yaw")
    axes[0].set_ylabel("orientation (deg)")

    axes[1].plot(t, data["gyro_x"], label="gyro_x")
    axes[1].plot(t, data["gyro_y"], label="gyro_y")
    axes[1].plot(t, data["gyro_z"], label="gyro_z")
    axes[1].set_ylabel("angular velocity (deg/s)")

    axes[2].plot(t, data["accel_x"], label="accel_x")
    axes[2].plot(t, data["accel_y"], label="accel_y")
    axes[2].plot(t, data["accel_z"], label="accel_z")
    axes[2].set_ylabel("linear acc")
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle(str(csv_path))
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"[INFO] plot 저장: {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IMU 데이터를 CSV로 저장하고 orientation/gyro/accel 플롯을 그립니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", help="시리얼 포트")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-d", "--duration", type=float, default=10.0)
    parser.add_argument("-c", "--csv", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--plot-only", type=Path, metavar="CSV")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="PNG만 저장하고 창은 띄우지 않음")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        print("\n".join(ports) if ports else "(사용 가능한 포트 없음)")
        return 0

    if args.plot_only:
        plot_imu(args.plot_only, args.png, show=not args.no_show)
        return 0

    csv_path = args.csv or default_csv_path()
    collect_imu(args.port or auto_port(), args.baud, args.duration, csv_path, args.quiet)
    plot_imu(csv_path, args.png, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
