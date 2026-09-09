#!/usr/bin/env python3
"""
belt_plot.py - [TEL] stdout을 실시간 그래프로 표시

사용법:
    python3 torque_cmd.py --torque 0 | python3 belt_plot.py
    python3 torque_cmd.py --torque 1.5 | python3 belt_plot.py --window 600
    python3 torque_cmd.py --torque 0 | python3 belt_plot.py --target 100

옵션:
    --window N     표시할 최대 샘플 수 (기본 300 = 약 30초 @ 10Hz)
    --target MM    목표 벨트 길이 기준선 표시 (mm)
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker


TEL_RE = re.compile(
    r"\[TEL\].*?belt=\s*([+-]?\d+\.?\d*)mm"
    r".*?vel=\s*([+-]?\d+\.?\d*)mm/s"
    r"(?:.*?spring_torque=\s*([+-]?\d+\.?\d*))?",
    re.ASCII,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAD belt length 실시간 그래프")
    p.add_argument("--time-window", type=float, default=60.0, help="표시할 시간 범위 (초, 기본 60s)")
    p.add_argument("--target", type=float, default=None, help="목표 belt 길이 기준선 (mm)")
    p.add_argument("--interval-ms", type=int, default=33, help="화면 갱신 주기 ms (기본 33 ≈ 30Hz)")
    return p.parse_args()


def reader_thread(
    buf_t: deque,
    buf_belt: deque,
    buf_vel: deque,
    buf_spring: deque,
    stop_event: threading.Event,
) -> None:
    t0 = None
    for raw in sys.stdin:
        if stop_event.is_set():
            break
        m = TEL_RE.search(raw)
        if not m:
            sys.stderr.write(raw)      # [CMD]/[RX] 등은 stderr로 전달
            continue
        now = time.monotonic()
        if t0 is None:
            t0 = now
        buf_t.append(now - t0)
        buf_belt.append(float(m.group(1)))
        buf_vel.append(float(m.group(2)))
        spring = float(m.group(3)) if m.group(3) else 0.0
        buf_spring.append(spring)


def main() -> None:
    args = parse_args()
    time_window_s = args.time_window
    # 100Hz 기준으로 버퍼 크기 설정 (최소 1000)
    N = max(1000, int(time_window_s * 110))

    buf_t      = deque(maxlen=N)
    buf_belt   = deque(maxlen=N)
    buf_vel    = deque(maxlen=N)
    buf_spring = deque(maxlen=N)
    stop_event = threading.Event()

    t = threading.Thread(
        target=reader_thread,
        args=(buf_t, buf_belt, buf_vel, buf_spring, stop_event),
        daemon=True,
    )
    t.start()

    n_axes = 3 if True else 2
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("PAD Belt Monitor", fontsize=13)

    ax_belt, ax_vel, ax_spring = axes

    (ln_belt,) = ax_belt.plot([], [], "b-", linewidth=1.5, label="belt length")
    if args.target is not None:
        ax_belt.axhline(args.target, color="orange", linewidth=1.0, linestyle="--", label=f"target {args.target}mm")
    ax_belt.set_ylabel("Belt Length (mm)")
    ax_belt.grid(True, which="major", alpha=0.4)
    ax_belt.grid(True, which="minor", alpha=0.15)
    ax_belt.legend(loc="upper right", fontsize=8)

    (ln_vel,) = ax_vel.plot([], [], "r-", linewidth=1.2, label="velocity")
    ax_vel.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_vel.set_ylabel("Velocity (mm/s)")
    ax_vel.grid(True, which="major", alpha=0.4)
    ax_vel.grid(True, which="minor", alpha=0.15)
    ax_vel.legend(loc="upper right", fontsize=8)

    (ln_spring,) = ax_spring.plot([], [], "g-", linewidth=1.2, label="spring torque")
    ax_spring.set_ylabel("Spring Torque (N·m)")
    ax_spring.set_xlabel("Time (s)")
    ax_spring.grid(True, which="major", alpha=0.4)
    ax_spring.grid(True, which="minor", alpha=0.15)
    ax_spring.legend(loc="upper right", fontsize=8)

    # 1초 단위 major tick, 0.5초 minor tick
    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.minorticks_on()

    def update(_frame):
        if not buf_t:
            return ln_belt, ln_vel, ln_spring
        ts = list(buf_t)
        t_now = ts[-1]
        t_min = t_now - time_window_s

        ln_belt.set_data(ts, list(buf_belt))
        ln_vel.set_data(ts, list(buf_vel))
        ln_spring.set_data(ts, list(buf_spring))

        # X축: 항상 최근 time_window_s 초만 표시 (슬라이딩)
        for ax in axes:
            ax.set_xlim(t_min, t_now)
            ax.relim()
            ax.autoscale_view(scalex=False)
        return ln_belt, ln_vel, ln_spring

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=args.interval_ms,
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
