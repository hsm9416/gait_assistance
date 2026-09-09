#!/usr/bin/env python3
"""
Live PAD external-control runner for a PC + TTL-to-USB adapter.

Edit only `compute_current_a()` to test a host-side control algorithm while
reusing the serial protocol handling from `pad_external_control_lib.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pad_external_control_lib import (  # noqa: E402
    PadAsciiReply,
    PadLinkController,
    PadLinkState,
    PadTelemetry,
    PadTelemetryFrame,
    clamp_current_a,
    force_to_current_a,
    list_serial_ports,
    run_self_test,
)


def compute_current_a(telemetry: PadTelemetry, state: PadLinkState) -> float:
    """
    Replace this function with your control algorithm.

    Input:
      - telemetry: latest telemetry frame from the target board
      - state: current host-side PAD link state

    Output:
      - current command in ampere that will be sent as:
        `PADCTRL CURRENT <value>`

    Helpful conversions:
      - telemetry.belt_length
      - telemetry.belt_velocity
      - telemetry.input_force
      - telemetry.output_force
      - telemetry.spring_torque
      - force_to_current_a(force_value)
    """

    _ = state

    # Example:
    # if telemetry.belt_velocity < -5.0:
    #     return 0.15
    #
    # If your algorithm is force-based instead of current-based:
    # desired_force = max(0.0, telemetry.output_force * 0.5)
    # return force_to_current_a(desired_force)
    return 0.0


def load_algorithm_from_file(path: Path, function_name: str) -> Callable[[PadTelemetry, PadLinkState], Optional[float]]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load algorithm file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    algorithm = getattr(module, function_name, None)
    if algorithm is None:
        raise RuntimeError(f"Function '{function_name}' was not found in {path}")
    return algorithm


def print_event(event, state: PadLinkState, quiet_telemetry: bool) -> None:
    if isinstance(event, PadAsciiReply):
        print(f"[ASCII] {event.line}")
        return

    if quiet_telemetry:
        return

    telemetry = event.telemetry
    print(f"[TELEM] {telemetry.summary()} state={state.session_state.value}")


def print_current_command(current_a: float, telemetry: PadTelemetry, state: PadLinkState) -> None:
    print(
        "[CTRL] "
        f"current={current_a:.2f}A "
        f"seq={telemetry.seq} "
        f"belt_len={telemetry.belt_length:.1f} "
        f"belt_vel={telemetry.belt_velocity:.1f} "
        f"state={state.session_state.value}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a host-side PAD control algorithm over a TTL-to-USB serial link."
    )
    parser.add_argument("--port", help="Serial port connected to the target USART3 line, for example COM12.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate. Default: 115200.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds. If omitted, runs until Ctrl+C.",
    )
    parser.add_argument(
        "--control-period-ms",
        type=float,
        default=10.0,
        help="Control loop period in milliseconds. Default: 10.",
    )
    parser.add_argument(
        "--current-limit",
        type=float,
        default=13.0,
        help="Clamp outgoing current commands to this maximum. Default: 13.0A.",
    )
    parser.add_argument(
        "--quiet-telemetry",
        action="store_true",
        help="Hide per-frame telemetry logs and print only ASCII replies and outgoing control commands.",
    )
    parser.add_argument(
        "--no-start-session",
        action="store_true",
        help="Do not send PADCTRL START automatically.",
    )
    parser.add_argument(
        "--no-stop-on-exit",
        action="store_true",
        help="Do not send PADCTRL STOP automatically on exit.",
    )
    parser.add_argument(
        "--algorithm-file",
        type=Path,
        help="Optional external Python file that defines the algorithm function.",
    )
    parser.add_argument(
        "--algorithm-function",
        default="compute_current_a",
        help="Function name inside --algorithm-file. Default: compute_current_a.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run protocol parser self-test without opening a serial port.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.list_ports:
        for port in list_serial_ports():
            print(port)
        return 0

    if not args.port:
        parser.error("--port is required unless --self-test or --list-ports is used.")

    algorithm: Callable[[PadTelemetry, PadLinkState], Optional[float]]
    if args.algorithm_file is not None:
        algorithm = load_algorithm_from_file(args.algorithm_file.resolve(), args.algorithm_function)
    else:
        algorithm = compute_current_a

    controller = PadLinkController(
        port=args.port,
        baudrate=args.baud,
    )

    try:
        controller.open()
        print(f"[INFO] open port={args.port} baud={args.baud}")
        controller.run_control_loop(
            algorithm,
            control_period_s=args.control_period_ms / 1000.0,
            duration_s=args.duration,
            start_session=not args.no_start_session,
            stop_on_exit=not args.no_stop_on_exit,
            current_limit_a=clamp_current_a(args.current_limit, args.current_limit),
            on_event=lambda event, state: print_event(event, state, args.quiet_telemetry),
            on_current_command=print_current_command,
        )
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        controller.close()
        print("[INFO] serial port closed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

