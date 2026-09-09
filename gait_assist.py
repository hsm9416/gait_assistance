#!/usr/bin/env python3
"""
gait_assist.py - Heel Strike 기반 보행 보조 토크 제어

Heel strike 감지 후 사다리꼴 토크 프로파일을 적용합니다.
발을 들어올리는 구간(swing phase)에서 모터 토크를 보조합니다.

토크 프로파일:
        peak_torque
            │    ┌─────┐
            │   /       \
            │  /         \
            │ /           \
       0 ───┤──────────────────→ time
            ↑  ↑         ↑   ↑
           HS  rise    fall  end
                ←rise→←pk→←fall→

파라미터는 아래 PARAMETERS 섹션에서 조정하세요.

실행:
    python3 gait_assist.py
    python3 gait_assist.py --port /dev/ttyUSB0

그래프와 함께:
    python3 gait_assist.py | python3 belt_plot.py --target 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pad_external_control_lib import list_serial_ports
from connect import PadController, PadSensorData
from heelstrike_detect import HeelStrikeDetector


# ── PARAMETERS ────────────────────────────────────────────────────────────────

# [Heel Strike 감지]
HS_MIN_BELT_MM  = 50.0   # heel strike 인정 최소 벨트 길이 (mm)
HS_REFRACTORY_S = 0.5    # 연속 감지 최소 간격 (s)
HS_LOADING_VEL  = 10.0   # Loading 진입 속도 임계값 (mm/s, 양수)
HS_STRIKE_VEL   = -5.0   # Strike 판별 속도 임계값 (mm/s, 음수)

# [토크 프로파일]
RISE_TIME_S     = 0.10   # 토크 상승 시간 (s): 0 → peak
PEAK_TIME_S     = 0.05   # 피크 유지 시간 (s)
FALL_TIME_S     = 0.20   # 토크 하강 시간 (s): peak → 0
PEAK_TORQUE_NM  = 2.0    # 피크 토크 (N·m)

# [안전]
TORQUE_LIMIT_NM = 5.0    # 절대 최대 토크 (N·m)

# ─────────────────────────────────────────────────────────────────────────────


class TorqueProfile:
    """
    사다리꼴 토크 프로파일.

    heel strike 후 elapsed_s 만큼 시간이 지났을 때의 토크를 반환합니다.
    """

    def __init__(
        self,
        rise_s: float = RISE_TIME_S,
        peak_s: float = PEAK_TIME_S,
        fall_s: float = FALL_TIME_S,
        peak_nm: float = PEAK_TORQUE_NM,
    ) -> None:
        self.rise_s    = rise_s
        self.peak_s    = peak_s
        self.fall_s    = fall_s
        self.peak_nm   = peak_nm
        self.duration_s = rise_s + peak_s + fall_s

    def torque_at(self, elapsed_s: float) -> float:
        """elapsed_s: heel strike 이후 경과 시간 (s)"""
        if elapsed_s < 0 or elapsed_s >= self.duration_s:
            return 0.0
        if elapsed_s < self.rise_s:
            frac = elapsed_s / self.rise_s if self.rise_s > 0 else 1.0
            return self.peak_nm * frac
        elapsed_s -= self.rise_s
        if elapsed_s < self.peak_s:
            return self.peak_nm
        elapsed_s -= self.peak_s
        if self.fall_s > 0:
            return self.peak_nm * (1.0 - elapsed_s / self.fall_s)
        return 0.0

    def is_active(self, elapsed_s: float) -> bool:
        return 0.0 <= elapsed_s < self.duration_s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heel Strike 기반 보행 보조 토크 제어",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",          help="시리얼 포트 (생략 시 자동 감지)")
    parser.add_argument("--baud",          type=int,   default=115200)
    parser.add_argument("--duration",      type=float, default=None,           help="실행 시간(s). 생략 시 Ctrl+C까지")
    parser.add_argument("--min-belt",      type=float, default=HS_MIN_BELT_MM,  help="최소 벨트 길이 (mm)")
    parser.add_argument("--refractory",    type=float, default=HS_REFRACTORY_S, help="최소 감지 간격 (s)")
    parser.add_argument("--loading-vel",   type=float, default=HS_LOADING_VEL,  help="Loading 속도 임계값 (mm/s)")
    parser.add_argument("--strike-vel",    type=float, default=HS_STRIKE_VEL,   help="Strike 속도 임계값 (mm/s)")
    parser.add_argument("--rise-time",     type=float, default=RISE_TIME_S,     help="토크 상승 시간 (s)")
    parser.add_argument("--peak-time",     type=float, default=PEAK_TIME_S,     help="피크 유지 시간 (s)")
    parser.add_argument("--fall-time",     type=float, default=FALL_TIME_S,     help="토크 하강 시간 (s)")
    parser.add_argument("--peak-torque",   type=float, default=PEAK_TORQUE_NM,  help="피크 토크 (N·m)")
    parser.add_argument("--torque-limit",  type=float, default=TORQUE_LIMIT_NM, help="토크 상한 (N·m)")
    args = parser.parse_args()

    if not args.port:
        ports = list_serial_ports()
        if not ports:
            print("오류: 연결된 포트 없음", file=sys.stderr)
            return 1
        args.port = ports[0]
        print(f"[INFO] 포트 자동 선택: {args.port}")

    detector = HeelStrikeDetector(
        min_belt_mm  = args.min_belt,
        refractory_s = args.refractory,
        loading_vel  = args.loading_vel,
        strike_vel   = args.strike_vel,
    )
    profile = TorqueProfile(
        rise_s   = args.rise_time,
        peak_s   = args.peak_time,
        fall_s   = args.fall_time,
        peak_nm  = args.peak_torque,
    )

    t0 = time.monotonic()
    _hs_time: list[float] = [-999.0]   # 마지막 heel strike 시각
    _hs_count: list[int]  = [0]

    def _algo(data: PadSensorData, _: PadController) -> float:
        t = time.monotonic() - t0

        # Heel strike 판별
        if detector.update(data.belt_length, data.belt_velocity, t):
            _hs_time[0] = t
            _hs_count[0] += 1
            print(
                f"[HS ] #{_hs_count[0]:3d}  t={t:6.2f}s  "
                f"belt={data.belt_length:.1f}mm",
                flush=True,
            )

        # 토크 프로파일 출력
        elapsed = t - _hs_time[0]
        torque = profile.torque_at(elapsed)
        torque = min(torque, args.torque_limit)
        return torque

    def _on_data(data: PadSensorData, _: PadController) -> None:
        t = time.monotonic() - t0
        elapsed = t - _hs_time[0]
        torque = profile.torque_at(elapsed)
        print(
            f"[TEL] seq={data.seq:5d}  "
            f"belt={data.belt_length:7.1f}mm  "
            f"vel={data.belt_velocity:+7.1f}mm/s  "
            f"torque={torque:5.3f}N·m  "
            f"hs={_hs_count[0]:3d}",
            flush=True,
        )

    print(f"[INFO] 토크 프로파일: rise={args.rise_time}s  peak={args.peak_time}s  "
          f"fall={args.fall_time}s  peak_torque={args.peak_torque}N·m  "
          f"total={profile.duration_s:.2f}s")
    print(f"[INFO] HS 파라미터: min_belt={args.min_belt}mm  "
          f"loading_vel={args.loading_vel}mm/s  strike_vel={args.strike_vel}mm/s")

    ctrl = PadController(port=args.port, baudrate=args.baud)
    try:
        ctrl.open()
        print(f"[INFO] 연결됨: {args.port}  — Ctrl+C로 종료")
        ctrl.run_control_loop(
            _algo,
            duration_s=args.duration,
            current_limit_a=args.torque_limit + 1.0,
            on_data=_on_data,
        )
    except KeyboardInterrupt:
        print(f"\n[INFO] 종료. 총 heel strike 횟수: {_hs_count[0]}")
    finally:
        ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
