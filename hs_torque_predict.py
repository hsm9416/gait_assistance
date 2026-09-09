#!/usr/bin/env python3
"""사전 보행 HS 모델을 이용해 예상 heel strike 타이밍에 torque를 출력합니다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from connect import PadController, PadSensorData
from heelstrike_detect import HeelStrikeDetector
from pad_external_control_lib import list_serial_ports


RISE_TIME_S = 0.05
PEAK_TIME_S = 0.08
FALL_TIME_S = 0.18
PEAK_TORQUE_NM = 2.0
TORQUE_LIMIT_NM = 5.0


class TorqueProfile:
    def __init__(self, rise_s: float, peak_s: float, fall_s: float, peak_nm: float) -> None:
        self.rise_s = rise_s
        self.peak_s = peak_s
        self.fall_s = fall_s
        self.peak_nm = peak_nm
        self.duration_s = rise_s + peak_s + fall_s

    def torque_at(self, elapsed_s: float) -> float:
        if elapsed_s < 0.0 or elapsed_s >= self.duration_s:
            return 0.0
        if elapsed_s < self.rise_s:
            return self.peak_nm * elapsed_s / self.rise_s if self.rise_s > 0 else self.peak_nm
        elapsed_s -= self.rise_s
        if elapsed_s < self.peak_s:
            return self.peak_nm
        elapsed_s -= self.peak_s
        return self.peak_nm * (1.0 - elapsed_s / self.fall_s) if self.fall_s > 0 else 0.0


def auto_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("연결된 포트 없음")
    print(f"[INFO] 포트 자동 선택: {ports[0]}")
    return ports[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=Path, default=Path("hs_model.json"))
    parser.add_argument("-p", "--port")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-d", "--duration", type=float, default=None)
    parser.add_argument("--offset", type=float, default=0.0, help="예상 HS 기준 토크 시작 오프셋(s). 음수면 미리 시작")
    parser.add_argument("--rise", type=float, default=RISE_TIME_S)
    parser.add_argument("--peak-time", type=float, default=PEAK_TIME_S)
    parser.add_argument("--fall", type=float, default=FALL_TIME_S)
    parser.add_argument("--torque", type=float, default=PEAK_TORQUE_NM)
    parser.add_argument("--limit", type=float, default=TORQUE_LIMIT_NM)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    model = json.loads(args.model.read_text())
    stride_s = float(model["stride_s"])
    detector = HeelStrikeDetector(
        float(model["min_belt_mm"]),
        float(model["refractory_s"]),
        float(model["loading_vel"]),
        float(model["strike_vel"]),
    )
    profile = TorqueProfile(args.rise, args.peak_time, args.fall, args.torque)

    t0 = time.monotonic()
    hs_count = 0
    next_hs_t: float | None = None
    last_torque = 0.0

    def algo(data: PadSensorData, _: PadController) -> float:
        nonlocal hs_count, next_hs_t, last_torque
        now = time.monotonic() - t0

        if detector.update(data.belt_length, data.belt_velocity, now):
            hs_count += 1
            next_hs_t = now + stride_s
            print(f"[HS ] #{hs_count} t={now:.2f}s next={next_hs_t:.2f}s", flush=True)

        if next_hs_t is None:
            last_torque = 0.0
            return 0.0

        start_t = next_hs_t + args.offset
        elapsed = now - start_t
        torque = min(profile.torque_at(elapsed), args.limit)

        if now > start_t + profile.duration_s:
            next_hs_t += stride_s

        last_torque = torque
        return torque

    def on_data(data: PadSensorData, _: PadController) -> None:
        if args.quiet:
            return
        now = time.monotonic() - t0
        next_text = f"{next_hs_t:.2f}" if next_hs_t is not None else "wait"
        print(
            f"[TEL] t={now:6.2f}s belt={data.belt_length:7.1f}mm "
            f"vel={data.belt_velocity:+7.1f}mm/s torque={last_torque:5.2f}Nm "
            f"hs={hs_count} next={next_text}",
            flush=True,
        )

    ctrl = PadController(args.port or auto_port(), baudrate=args.baud)
    try:
        print(
            f"[INFO] model={args.model} stride={stride_s:.3f}s "
            f"torque={args.torque}Nm offset={args.offset}s"
        )
        ctrl.open()
        ctrl.run_control_loop(
            algo,
            duration_s=args.duration,
            current_limit_a=args.limit + 1.0,
            on_data=on_data,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
