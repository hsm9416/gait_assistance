#!/usr/bin/env python3
"""
주기적 토크 ON/OFF 테스트 알고리즘 (torque.py)

지정한 토크를 ON_DURATION_S 동안 출력하고,
OFF_DURATION_S 동안 0으로 반복합니다.

CLI 실행:
    python torque.py               # 포트 자동 감지
    python torque.py --port /dev/ttyUSB0
    python torque.py --list-ports
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pad_external_control_lib import clamp_current_a, list_serial_ports, run_self_test
from connect import PadController, PadSensorData


# ──────────────────────────────────────────────────────────────────────────────
# 파라미터 — 여기서 조정
# ──────────────────────────────────────────────────────────────────────────────

TORQUE_ON_NM          = 0.1    # ON 구간에 출력할 토크 (N·m)
ON_DURATION_S         = 10.0    # 토크 켜짐 지속 시간 (s)
OFF_DURATION_S        = 1.0    # 토크 꺼짐 지속 시간 (s)

TORQUE_CONST_NM_PER_A = 0.1    # 모터 토크 상수 (N·m/A) — 사양서 또는 실측값으로 교체
CURRENT_LIMIT_A       = 1.0    # 안전 전류 상한 (A)

CONTROL_PERIOD_MS     = 10.0   # 제어 루프 주기 (ms)


# ──────────────────────────────────────────────────────────────────────────────
# 토크 ↔ 전류 변환
# ──────────────────────────────────────────────────────────────────────────────

def torque_to_current_a(
    torque_nm: float,
    torque_const_nm_per_a: float,
    limit_a: float = 13.0,
) -> float:
    if torque_nm <= 0.0 or torque_const_nm_per_a <= 0.0:
        return 0.0
    return clamp_current_a(torque_nm / torque_const_nm_per_a, limit_a)


# ──────────────────────────────────────────────────────────────────────────────
# 알고리즘
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TorqueParams:
    """외부 코드에서 파라미터를 묶어 전달할 때 사용."""
    torque_on_nm: float = TORQUE_ON_NM
    on_duration_s: float = ON_DURATION_S
    off_duration_s: float = OFF_DURATION_S
    torque_const_nm_per_a: float = TORQUE_CONST_NM_PER_A
    current_limit_a: float = CURRENT_LIMIT_A


class TorqueAlgorithm:
    """
    주기적 토크 ON/OFF 알고리즘.

    ON_DURATION_S 동안 TORQUE_ON_NM 출력 → OFF_DURATION_S 동안 0 출력 → 반복.

    사용법::

        algo = TorqueAlgorithm()
        ctrl.run_control_loop(algo.compute)
    """

    def __init__(self, params: Optional[TorqueParams] = None) -> None:
        self.params = params or TorqueParams()
        self._start_time: Optional[float] = None
        self._is_on: bool = False

    def reset(self) -> None:
        """내부 상태 초기화 (세션 재시작 시 호출)."""
        self._start_time = None
        self._is_on = False

    @property
    def is_on(self) -> bool:
        return self._is_on

    def compute(self, *_: object) -> Optional[float]:
        p = self.params
        now = time.monotonic()

        if self._start_time is None:
            self._start_time = now

        period = p.on_duration_s + p.off_duration_s
        phase = (now - self._start_time) % period

        self._is_on = phase < p.on_duration_s
        torque = p.torque_on_nm if self._is_on else 0.0

        return torque_to_current_a(torque, p.torque_const_nm_per_a, p.current_limit_a)


# ──────────────────────────────────────────────────────────────────────────────
# 콜백: 수신 데이터 출력
# ──────────────────────────────────────────────────────────────────────────────

def _make_on_data(algo: TorqueAlgorithm):
    def _on_data(data: PadSensorData, _: PadController) -> None:
        state_str = " ON" if algo.is_on else "OFF"
        print(
            f"[TEL] seq={data.seq:5d}  [{state_str}]  "
            f"motor_iq_meas={data.motor_iq_meas:5.2f}A  "
            f"motor_vel={data.motor_vel:6.2f}  "
            f"belt={data.belt_length:7.1f}mm  "
            f"F_out={data.output_force:5.2f}N  "
            f"spring_torque={data.spring_torque:6.3f}N·m"
        )
    return _on_data


def _on_ascii_reply(line: str, _: PadController) -> None:
    print(f"[RX ] {line}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PAD 주기적 토크 ON/OFF 테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", help="직렬 포트 (예: /dev/ttyUSB0, COM3). 생략하면 자동 감지")
    p.add_argument("--baud", type=int, default=115200, help="보드레이트")
    p.add_argument("--quiet", action="store_true", help="텔레메트리 로그 숨기기")
    p.add_argument("--list-ports", action="store_true", help="사용 가능한 시리얼 포트 목록 출력 후 종료")
    p.add_argument("--self-test", action="store_true", help="프로토콜 파서 자체 테스트 후 종료")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.list_ports:
        ports = list_serial_ports()
        if ports:
            for port in ports:
                print(port)
        else:
            print("(사용 가능한 포트 없음)")
        return 0

    if not args.port:
        ports = list_serial_ports()
        if not ports:
            print("오류: 연결된 시리얼 포트가 없습니다.")
            return 1
        if len(ports) == 1:
            args.port = ports[0]
            print(f"[INFO] 포트 자동 선택: {args.port}")
        else:
            print("사용 가능한 포트:")
            for i, port in enumerate(ports):
                print(f"  [{i}] {port}")
            try:
                idx = int(input("포트 번호 선택: "))
                args.port = ports[idx]
            except (ValueError, IndexError):
                print("오류: 올바른 번호를 입력하세요.")
                return 1

    params = TorqueParams()
    algo = TorqueAlgorithm(params)

    print(
        f"[INFO] ON={params.on_duration_s}s / OFF={params.off_duration_s}s  "
        f"torque={params.torque_on_nm}N·m  "
        f"→ {torque_to_current_a(params.torque_on_nm, params.torque_const_nm_per_a, params.current_limit_a):.2f}A"
    )

    ctrl = PadController(
        port=args.port,
        baudrate=args.baud,
        read_timeout_s=0.01,
        ack_timeout_s=1.0,
        telemetry_stale_s=0.25,
    )

    try:
        ctrl.open()
        print(f"[INFO] 연결됨: port={args.port}  baud={args.baud}")

        ctrl.run_control_loop(
            algo.compute,
            control_period_s=CONTROL_PERIOD_MS / 1000.0,
            start_session=True,
            stop_on_exit=True,
            current_limit_a=params.current_limit_a,
            on_data=None if args.quiet else _make_on_data(algo),
            on_ascii_reply=_on_ascii_reply,
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


if __name__ == "__main__":
    raise SystemExit(main())
