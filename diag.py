#!/usr/bin/env python3
"""
Serial diagnostic: 포트를 열고 raw 바이트를 5초 동안 출력합니다.
PADCTRL STATUS / PADCTRL START 를 보내고 장치 응답을 확인합니다.

실행:
    python3 diag.py
    python3 diag.py --port /dev/ttyUSB0 --baud 115200
    python3 diag.py --no-send    # 명령 전송 없이 수신만
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial.tools import list_ports


def hex_dump(data: bytes, prefix: str = "") -> None:
    if not data:
        return
    hex_part = " ".join(f"{b:02X}" for b in data)
    try:
        ascii_part = data.decode("ascii", errors="replace")
        ascii_part = ascii_part.replace("\r", "\\r").replace("\n", "\\n")
    except Exception:
        ascii_part = "?"
    print(f"{prefix}[{len(data):3d}B] {hex_part}  |  {ascii_part}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PAD 시리얼 진단")
    parser.add_argument("--port", help="시리얼 포트 (자동 감지)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--listen-s", type=float, default=5.0, help="수신 대기 시간(s)")
    parser.add_argument("--no-send", action="store_true", help="명령 전송 안 함")
    parser.add_argument("--delay", type=float, default=2.0, help="포트 오픈 후 대기(s)")
    args = parser.parse_args()

    if not args.port:
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            print("오류: 연결된 포트 없음")
            return 1
        args.port = ports[0]
        print(f"[DIAG] 자동 선택: {args.port}")

    print(f"[DIAG] 포트 열기: {args.port}  baud={args.baud}")

    # DTR=False 로 열어서 ESP32/Arduino 리셋 방지
    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = args.baud
    ser.timeout = 0.05
    ser.dtr = False
    ser.rts = False
    ser.open()

    print(f"[DIAG] 오픈 완료. {args.delay}초 대기 중...")
    time.sleep(args.delay)

    # 버퍼 비우기
    ser.reset_input_buffer()
    print("[DIAG] 입력 버퍼 초기화 완료")

    # 명령 전송
    if not args.no_send:
        cmds = ["PADCTRL START", "PADCTRL PADMODE ENABLE"]
        for cmd in cmds:
            payload = (cmd + "\n").encode("ascii")
            ser.write(payload)
            ser.flush()
            print(f"[TX ] {cmd!r}  ({len(payload)}B)")
            time.sleep(0.1)

    # 수신 루프
    print(f"\n[DIAG] {args.listen_s}초 동안 수신 대기...\n")
    deadline = time.monotonic() + args.listen_s
    total_bytes = 0
    line_buf = bytearray()

    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            total_bytes += len(chunk)
            hex_dump(chunk, prefix="[RX ] ")

            # ASCII 라인 감지
            for b in chunk:
                if b == 0x0A:  # \n
                    line = line_buf.decode("ascii", errors="replace").strip()
                    if line:
                        print(f"[LINE] {line!r}")
                    line_buf.clear()
                elif b != 0x0D:  # \r 제외
                    line_buf.append(b)
        else:
            time.sleep(0.02)

    ser.close()
    print(f"\n[DIAG] 완료. 수신 총 {total_bytes}바이트")
    if total_bytes == 0:
        print("[DIAG] 장치가 아무 데이터도 보내지 않음 → baudrate 확인 or 장치 미동작")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
