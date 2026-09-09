#!/usr/bin/env python3
"""
heelstrike_detect.py - Heel Strike 감지 모듈

벨트 속도 기반 상태머신으로 heel strike를 판별합니다.
벨트가 빠르게 늘어난(발이 땅을 향함) 뒤, 다시 빠르게 줄어드는 변곡점이 heel strike입니다.

상태 흐름:
    IDLE  →  (vel > LOADING_VEL)  →  LOADING
    LOADING  →  (vel < STRIKE_VEL)  →  [Heel Strike!]  →  IDLE

단독 실행 (감지 파라미터 튜닝용):
    python3 heelstrike_detect.py
    python3 heelstrike_detect.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── 감지 파라미터 ───────────────────────────────────────
HS_MIN_BELT_MM  = -130.0 # 음수이면 belt <= 값, 양수이면 belt >= 값에서 감지
HS_REFRACTORY_S = 0.5    # 연속 heel strike 감지 최소 간격 (s)
                          # 너무 짧으면 노이즈로 중복 감지됨
HS_LOADING_VEL  = 10.0   # 이 속도(mm/s) 이상이면 Loading 시작
                          # 벨트가 빠르게 늘어나는 중 (발이 지면으로)
HS_STRIKE_VEL   = -5.0   # 이 속도(mm/s) 이하로 꺾이면 heel strike
                          # 벨트가 다시 짧아지기 시작
# ──────────────────────────────────────────────────────


class HeelStrikeDetector:
    """
    벨트 속도 변화로 heel strike 시점을 판별하는 상태머신.

    사용법:
        det = HeelStrikeDetector()
        # 매 텔레메트리 프레임마다:
        if det.update(belt_mm, vel_mm_s, time.monotonic()):
            print("Heel Strike!")
    """

    def __init__(
        self,
        min_belt_mm: float = HS_MIN_BELT_MM,
        refractory_s: float = HS_REFRACTORY_S,
        loading_vel: float = HS_LOADING_VEL,
        strike_vel: float = HS_STRIKE_VEL,
    ) -> None:
        self.min_belt_mm  = min_belt_mm
        self.refractory_s = refractory_s
        self.loading_vel  = loading_vel
        self.strike_vel   = strike_vel

        self._loading: bool = False
        self._last_hs_t: float = -999.0
        self.last_hs_belt_mm: float = 0.0   # 마지막 heel strike 시 벨트 길이

    def update(self, belt_mm: float, vel_mm_s: float, t: float) -> bool:
        """
        새 텔레메트리 샘플을 입력하고 heel strike 감지 여부를 반환합니다.

        Returns:
            True  : 이 샘플에서 heel strike 감지됨
            False : 아직 미감지
        """
        # 벨트가 빠르게 늘어나고 있으면 Loading 상태 진입
        if vel_mm_s >= self.loading_vel:
            self._loading = True

        if self._loading and vel_mm_s <= self.strike_vel:
            self._loading = False
            if self.min_belt_mm < 0:
                belt_ok = belt_mm <= self.min_belt_mm
            else:
                belt_ok = belt_mm >= self.min_belt_mm
            if (belt_ok
                    and t - self._last_hs_t >= self.refractory_s):
                self._last_hs_t = t
                self.last_hs_belt_mm = belt_mm
                return True

        return False

    def reset(self) -> None:
        self._loading = False
        self._last_hs_t = -999.0


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행: 감지 테스트
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def main() -> int:
    from pad_external_control_lib import list_serial_ports
    from connect import PadController, PadSensorData

    parser = argparse.ArgumentParser(
        description="Heel Strike 감지 테스트 (파라미터 튜닝용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",            help="시리얼 포트 (생략 시 자동 감지)")
    parser.add_argument("--baud",            type=int,   default=115200)
    parser.add_argument("--min-belt",        type=float, default=HS_MIN_BELT_MM,  help="최소 벨트 길이 (mm)")
    parser.add_argument("--refractory",      type=float, default=HS_REFRACTORY_S, help="최소 감지 간격 (s)")
    parser.add_argument("--loading-vel",     type=float, default=HS_LOADING_VEL,  help="Loading 진입 속도 (mm/s)")
    parser.add_argument("--strike-vel",      type=float, default=HS_STRIKE_VEL,   help="Strike 판별 속도 (mm/s)")
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

    t0 = time.monotonic()
    hs_count = 0

    def _on_data(data: PadSensorData, _: PadController) -> None:
        nonlocal hs_count
        t = time.monotonic() - t0
        detected = detector.update(data.belt_length, data.belt_velocity, t)

        marker = "  <<< HEEL STRIKE!" if detected else ""
        if detected:
            hs_count += 1

        print(
            f"[TEL] t={t:6.2f}s  "
            f"belt={data.belt_length:7.1f}mm  "
            f"vel={data.belt_velocity:+7.1f}mm/s  "
            f"hs={hs_count:3d}"
            f"{marker}"
        )

    ctrl = PadController(port=args.port, baudrate=args.baud)
    try:
        ctrl.open()
        print(f"[INFO] 연결됨: {args.port}  — Ctrl+C로 종료")
        print(f"[INFO] 파라미터: min_belt={args.min_belt}mm  refractory={args.refractory}s  "
              f"loading_vel={args.loading_vel}mm/s  strike_vel={args.strike_vel}mm/s")
        ctrl.run_control_loop(
            lambda _d, _c: 0.0,   # 토크 없음, 모니터링만
            current_limit_a=0.1,
            on_data=_on_data,
        )
    except KeyboardInterrupt:
        print(f"\n[INFO] 종료. 총 감지 횟수: {hs_count}")
    finally:
        ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
