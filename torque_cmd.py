#!/usr/bin/env python3
"""
고정 토크 명령 (torque_cmd.py)

PADCTRL TORQUE <value> 를 지정한 값으로 계속 전송합니다.
음수 토크도 지원합니다.

CLI:
    python3 torque_cmd.py --torque 1.0
    python3 torque_cmd.py --torque -1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pad_external_control_lib import list_serial_ports
from connect import PadController, PadSensorData


CONTROL_PERIOD_MS = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PAD 고정 토크 명령",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--torque",   type=float, required=True, help="출력 토크 (N·m, 음수 가능)")
    parser.add_argument("--port",     help="직렬 포트 (생략 시 자동 감지)")
    parser.add_argument("--baud",     type=int, default=115200)
    parser.add_argument("--duration", type=float, default=None, help="실행 시간(s). 생략 시 Ctrl+C까지")
    args = parser.parse_args()

    if not args.port:
        ports = list_serial_ports()
        if not ports:
            print("오류: 연결된 포트 없음", file=sys.stderr)
            return 1
        if len(ports) == 1:
            args.port = ports[0]
            print(f"[INFO] 포트 자동 선택: {args.port}")
        else:
            for i, pt in enumerate(ports):
                print(f"  [{i}] {pt}")
            try:
                args.port = list_serial_ports()[int(input("포트 번호: "))]
            except (ValueError, IndexError):
                print("오류: 올바른 번호를 입력하세요.", file=sys.stderr)
                return 1

    torque_nm = args.torque
    print(f"[INFO] 토크 명령: {torque_nm:+.3f} N·m")

    _status_requested = False
    _last_tel_time: list[float] = []   # [prev_time, count, window_start]

    def _algo(_: PadSensorData, c: PadController):
        nonlocal _status_requested
        if not _status_requested:
            c.request_status()
            _status_requested = True
        return torque_nm

    def _on_data(data: PadSensorData, _: PadController) -> None:
        now = time.monotonic()
        if not _last_tel_time:
            _last_tel_time.extend([now, 0, now])
        prev, count, win_start = _last_tel_time
        dt_ms = (now - prev) * 1000.0
        count += 1
        elapsed = now - win_start
        hz = count / elapsed if elapsed > 0.1 else 0.0
        _last_tel_time[0] = now
        _last_tel_time[1] = count
        print(
            f"[TEL] seq={data.seq:5d}  "
            f"belt={data.belt_length:7.1f}mm  "
            f"vel={data.belt_velocity:+7.1f}mm/s  "
            f"spring_torque={data.spring_torque:6.3f}N·m  "
            f"dt={dt_ms:6.1f}ms  hz={hz:5.1f}"
        )

    def _on_ascii(line: str, _: PadController) -> None:
        print(f"[RX ] {line}")

    ctrl = PadController(
        port=args.port,
        baudrate=args.baud,
        read_timeout_s=0.01,
        ack_timeout_s=2.0,
        telemetry_stale_s=0.25,
    )

    try:
        ctrl.open()
        print(f"[INFO] 연결됨: port={args.port}  baud={args.baud}")
        ctrl.run_control_loop(
            _algo,
            control_period_s=CONTROL_PERIOD_MS / 1000.0,
            duration_s=args.duration,
            start_session=True,
            stop_on_exit=True,
            current_limit_a=abs(torque_nm) + 1.0,
            on_data=_on_data,
            on_ascii_reply=_on_ascii,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[INFO] 사용자 중단 (Ctrl+C)")
    finally:
        ctrl.close()
        print("[INFO] 포트 종료")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
