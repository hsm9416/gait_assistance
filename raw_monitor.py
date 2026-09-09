#!/usr/bin/env python3
"""
RAW 시리얼 모니터 — 수신 바이트를 hex + ASCII로 실시간 출력.

실행:
    python3 raw_monitor.py
    python3 raw_monitor.py --port /dev/ttyUSB0
    python3 raw_monitor.py --port /dev/ttyUSB0 --send "PADCTRL START"
    python3 raw_monitor.py --port /dev/ttyUSB0 --send "PADCTRL START" --send "PADCTRL SENSOR_START"
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial.tools import list_ports


RESET = "\033[0m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
GRAY  = "\033[90m"
RED   = "\033[31m"


def hex_line(data: bytes, width: int = 16) -> str:
    hex_part   = " ".join(f"{b:02X}" for b in data).ljust(width * 3 - 1)
    ascii_part = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data)
    return f"{hex_part}  |  {ascii_part}"


def main() -> int:
    p = argparse.ArgumentParser(description="RAW 시리얼 모니터")
    p.add_argument("--port",  help="시리얼 포트 (생략 시 자동 감지)")
    p.add_argument("--baud",  type=int, default=115200)
    p.add_argument("--send",  action="append", metavar="CMD",
                   help="오픈 후 전송할 ASCII 명령 (여러 번 지정 가능)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="포트 오픈 후 명령 전송까지 대기 시간(s)")
    p.add_argument("--width", type=int, default=16, help="hex 줄당 바이트 수")
    args = p.parse_args()

    # 포트 자동 감지
    if not args.port:
        ports = [pt.device for pt in list_ports.comports()]
        if not ports:
            print(f"{RED}[ERR] 연결된 시리얼 포트 없음{RESET}", file=sys.stderr)
            return 1
        args.port = ports[0]
        print(f"{GRAY}[INFO] 자동 선택: {args.port}{RESET}")

    print(f"{GRAY}[INFO] 포트 열기: {args.port}  baud={args.baud}{RESET}")

    ser = serial.Serial()
    ser.port     = args.port
    ser.baudrate = args.baud
    ser.timeout  = 0.05
    ser.dtr      = False
    ser.rts      = False
    ser.open()

    print(f"{GRAY}[INFO] 오픈 완료. {args.delay}s 대기 중...{RESET}")
    time.sleep(args.delay)
    ser.reset_input_buffer()

    # 명령 전송
    if args.send:
        for cmd in args.send:
            payload = (cmd.rstrip("\r\n") + "\n").encode("ascii")
            ser.write(payload)
            ser.flush()
            ts = time.strftime("%H:%M:%S")
            print(f"{GREEN}[{ts}][TX ] {cmd!r}{RESET}")
            time.sleep(0.05)

    print(f"\n{GRAY}수신 대기 중... (Ctrl+C 로 종료){RESET}\n")

    line_buf  = bytearray()
    total_rx  = 0

    try:
        while True:
            waiting = ser.in_waiting
            if not waiting:
                time.sleep(0.01)
                continue

            chunk = ser.read(waiting)
            total_rx += len(chunk)
            ts = time.strftime("%H:%M:%S")

            # hex dump (width 단위로 줄 분할)
            for i in range(0, len(chunk), args.width):
                row = chunk[i:i + args.width]
                offset = f"{i:04X}" if len(chunk) > args.width else "    "
                print(f"{CYAN}[{ts}][RX ][{offset}] {hex_line(row, args.width)}{RESET}")

            # ASCII 라인 감지
            for b in chunk:
                if b == 0x0A:
                    line = line_buf.decode("ascii", errors="replace").strip()
                    if line:
                        print(f"        {GREEN}→ ASCII: {line!r}{RESET}")
                    line_buf.clear()
                elif b == 0x0D:
                    pass
                elif b == 0x00:
                    if line_buf:
                        print(f"        {GRAY}→ binary frame boundary (buf={len(line_buf)}B flushed){RESET}")
                        line_buf.clear()
                    else:
                        print(f"        {GRAY}→ 0x00 binary frame delimiter{RESET}")
                else:
                    line_buf.append(b)

    except KeyboardInterrupt:
        print(f"\n{GRAY}[INFO] 종료. 수신 총 {total_rx}B{RESET}")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
