#!/usr/bin/env python3
"""
임피던스 컨트롤러 — 목표 위치: belt_length = -50 mm

제어 법칙:
    F = K * (x_d - x) + D * (v_d - v)
    x_d = -50 mm,  v_d = 0 mm/s

CLI 실행:
    python impedance_controller.py --port /dev/ttyUSB0
    python impedance_controller.py --port /dev/ttyUSB0 --stiffness 2.0 --damping 0.05
"""

from __future__ import annotations

import argparse
from typing import Optional

from connect import PadController, PadSensorData
from pad_external_control_lib import force_to_current_a


# ──────────────────────────────────────────────────────────────────────────────
# 임피던스 파라미터 (여기서 조정)
# ──────────────────────────────────────────────────────────────────────────────

STIFFNESS_N_PER_MM = 0.005                                                                                              # K: 위치 오차 게인 (N/mm)
DAMPING_N_PER_MMS  = 0.005  # D: 속도 오차 게인 (N·s/mm)
CURRENT_LIMIT_A    = 1.0   # 안전 전류 상한 (A)
DESIRED_POS_MM     = 0.0   # 목표 벨트 길이 (mm)
DESIRED_VEL_MMS    = 0.0   # 목표 벨트 속도 (mm/s)


# ──────────────────────────────────────────────────────────────────────────────
# 제어 알고리즘
# ──────────────────────────────────────────────────────────────────────────────

def impedance_algorithm(
    data: PadSensorData,
    ctrl: PadController,
    stiffness: float = STIFFNESS_N_PER_MM,  
    damping: float = DAMPING_N_PER_MMS,
) -> Optional[float]:
    """
    목표 벨트 길이를 기준으로 하는 임피던스 제어.

    Returns:
        전류 명령 (A).
    """
    pos_error = DESIRED_POS_MM - data.belt_length    # mm
    vel_error = DESIRED_VEL_MMS - data.belt_velocity # mm/s

    force_n = stiffness * pos_error + damping * vel_error
    return force_to_current_a(force_n)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 출력 콜백
# ──────────────────────────────────────────────────────────────────────────────

def _on_data(data: PadSensorData, ctrl: PadController) -> None:
    pos_error = DESIRED_POS_MM - data.belt_length
    vel_error = DESIRED_VEL_MMS - data.belt_velocity
    force_n   = STIFFNESS_N_PER_MM * pos_error + DAMPING_N_PER_MMS * vel_error
    print(
        f"[TEL] seq={data.seq:5d}  "
        f"belt={data.belt_length:7.2f}mm  vel={data.belt_velocity:6.2f}mm/s  "
        f"pos_err={pos_error:7.2f}mm  vel_err={vel_error:6.2f}mm/s  "
        f"F={force_n:6.3f}N"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="임피던스 컨트롤러",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",     help="직렬 포트 (예: /dev/ttyUSB0)")
    p.add_argument("--baud",     type=int,   default=115200,            help="보드레이트")
    p.add_argument("--stiffness",type=float, default=STIFFNESS_N_PER_MM,help="위치 게인 K (N/mm)")
    p.add_argument("--damping",  type=float, default=DAMPING_N_PER_MMS, help="속도 게인 D (N·s/mm)")
    p.add_argument("--duration", type=float, default=None,              help="실행 시간(초). 생략 시 Ctrl+C 까지")
    p.add_argument("--period-ms",type=float, default=10.0,              help="제어 루프 주기 (ms)")
    p.add_argument("--quiet",    action="store_true",                   help="텔레메트리 로그 숨기기")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    def algo(data: PadSensorData, ctrl: PadController) -> Optional[float]:
        return impedance_algorithm(data, ctrl, args.stiffness, args.damping)

    ctrl = PadController(port=args.port or _auto_port(), baudrate=args.baud)
    try:
        ctrl.open()
        print(
            f"[INFO] 연결됨: port={args.port}  baud={args.baud}  "
            f"K={args.stiffness} N/mm  D={args.damping} N·s/mm"
        )
        ctrl.run_control_loop(
            algo,
            control_period_s=args.period_ms / 1000.0,
            duration_s=args.duration,
            current_limit_a=CURRENT_LIMIT_A,
            on_data=None if args.quiet else _on_data,
            on_ascii_reply=lambda line, _: print(f"[RX ] {line}"),
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except KeyboardInterrupt:
        print("[INFO] 사용자 중단 (Ctrl+C)")
    finally:
        ctrl.close()
        print("[INFO] 시리얼 포트 종료")

    return 0


def _auto_port() -> str:
    from pad_external_control_lib import list_serial_ports
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("연결된 시리얼 포트가 없습니다.")
    if len(ports) == 1:
        print(f"[INFO] 포트 자동 선택: {ports[0]}")
        return ports[0]
    print("사용 가능한 포트:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    try:
        return ports[int(input("포트 번호 선택: "))]
    except (ValueError, IndexError):
        raise RuntimeError("올바른 번호를 입력하세요.")


if __name__ == "__main__":
    raise SystemExit(main())
