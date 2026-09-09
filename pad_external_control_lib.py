#!/usr/bin/env python3
"""
Host-side PAD external control utilities.

This module ports the mixed ASCII + COBS telemetry handling from the ESP32
`PortableAssistDeviceControlTest` project so a PC can replace the ESP32 bridge
through a TTL-to-USB serial adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct
import time
from typing import Callable, List, Optional, Sequence, Union


PAD_TELEMETRY_STRUCT = struct.Struct("<BBHI24h")
PAD_TELEMETRY_TYPE = 0x01
PAD_TELEMETRY_VERSION = 0x01


class PadProtocolError(RuntimeError):
    """Raised when the PAD mixed stream contains invalid ASCII or telemetry."""


class PadSessionState(str, Enum):
    IDLE = "IDLE"
    AWAITING_START_ACK = "AWAITING_START_ACK"
    CONFIG_WAIT = "CONFIG_WAIT"          # after PADCTRL START
    SENSOR_STREAMING = "SENSOR_STREAMING"  # after PADCTRL SENSOR_START
    STREAMING = "STREAMING"              # CONTROL_ACTIVE (after CONTROL_START + first sample)
    AWAITING_STOP_ACK = "AWAITING_STOP_ACK"
    STALE = "STALE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class PadTelemetry:
    type: int
    version: int
    seq: int
    time10ms: int
    motor_pos_x100: int
    motor_vel_x100: int
    motor_iq_meas_x100: int
    motor_iq_set_x100: int
    belt_length_x10: int
    belt_vel_x10: int
    belt_acc_x1: int
    roll_x10: int
    pitch_x10: int
    yaw_x10: int
    gax_x100: int
    gay_x100: int
    gaz_x100: int
    ax_x100: int
    ay_x100: int
    az_x100: int
    input_force_x100: int
    output_force_x100: int
    spring_torque_x100: int
    bus_voltage_x100: int
    bus_current_x100: int
    driver_error: int
    detect_state: int
    control_state: int

    @classmethod
    def from_payload(cls, payload: bytes) -> "PadTelemetry":
        if len(payload) != PAD_TELEMETRY_STRUCT.size:
            raise PadProtocolError(
                f"Telemetry payload size mismatch: got {len(payload)}, "
                f"expected {PAD_TELEMETRY_STRUCT.size}"
            )

        values = PAD_TELEMETRY_STRUCT.unpack(payload)
        telemetry = cls(*values)
        if telemetry.type != PAD_TELEMETRY_TYPE or telemetry.version != PAD_TELEMETRY_VERSION:
            raise PadProtocolError(
                f"Unexpected telemetry header: type={telemetry.type} version={telemetry.version}"
            )
        return telemetry

    def to_payload(self) -> bytes:
        return PAD_TELEMETRY_STRUCT.pack(
            self.type,
            self.version,
            self.seq,
            self.time10ms,
            self.motor_pos_x100,
            self.motor_vel_x100,
            self.motor_iq_meas_x100,
            self.motor_iq_set_x100,
            self.belt_length_x10,
            self.belt_vel_x10,
            self.belt_acc_x1,
            self.roll_x10,
            self.pitch_x10,
            self.yaw_x10,
            self.gax_x100,
            self.gay_x100,
            self.gaz_x100,
            self.ax_x100,
            self.ay_x100,
            self.az_x100,
            self.input_force_x100,
            self.output_force_x100,
            self.spring_torque_x100,
            self.bus_voltage_x100,
            self.bus_current_x100,
            self.driver_error,
            self.detect_state,
            self.control_state,
        )

    @property
    def motor_pos(self) -> float:
        return self.motor_pos_x100 / 100.0

    @property
    def motor_vel(self) -> float:
        return self.motor_vel_x100 / 100.0

    @property
    def motor_iq_meas(self) -> float:
        return self.motor_iq_meas_x100 / 100.0

    @property
    def motor_iq_set(self) -> float:
        return self.motor_iq_set_x100 / 100.0

    @property
    def belt_length(self) -> float:
        return self.belt_length_x10 / 10.0

    @property
    def belt_velocity(self) -> float:
        return self.belt_vel_x10 / 10.0

    @property
    def belt_acceleration(self) -> float:
        return float(self.belt_acc_x1)

    @property
    def roll(self) -> float:
        return self.roll_x10 / 10.0

    @property
    def pitch(self) -> float:
        return self.pitch_x10 / 10.0

    @property
    def yaw(self) -> float:
        return self.yaw_x10 / 10.0

    @property
    def input_force(self) -> float:
        return self.input_force_x100 / 100.0

    @property
    def output_force(self) -> float:
        return self.output_force_x100 / 100.0

    @property
    def spring_torque(self) -> float:
        return self.spring_torque_x100 / 100.0

    @property
    def bus_voltage(self) -> float:
        return self.bus_voltage_x100 / 100.0

    @property
    def bus_current(self) -> float:
        return self.bus_current_x100 / 100.0

    def summary(self) -> str:
        return (
            f"seq={self.seq} time10ms={self.time10ms} "
            f"belt_len={self.belt_length:.1f} belt_vel={self.belt_velocity:.1f} "
            f"roll={self.roll:.1f} pitch={self.pitch:.1f} yaw={self.yaw:.1f} "
            f"input_force={self.input_force:.2f} output_force={self.output_force:.2f} "
            f"spring_torque={self.spring_torque:.2f}"
        )


@dataclass(frozen=True)
class PadAsciiReply:
    line: str


@dataclass(frozen=True)
class PadTelemetryFrame:
    telemetry: PadTelemetry


PadEvent = Union[PadAsciiReply, PadTelemetryFrame]


@dataclass
class PadLinkState:
    session_active: bool = False
    awaiting_start_ack: bool = False
    awaiting_stop_ack: bool = False
    pending_since_s: Optional[float] = None
    last_ascii_reply: str = ""
    last_tx_line: str = ""
    last_tx_s: Optional[float] = None
    last_telemetry_s: Optional[float] = None
    last_seq: int = 0
    telemetry_valid: bool = False
    requested_current_a: float = 0.0
    current_dirty: bool = False
    session_state: PadSessionState = PadSessionState.IDLE
    latest_telemetry: Optional[PadTelemetry] = None

    def _mark_streaming(self, session_state: PadSessionState = PadSessionState.STREAMING) -> None:
        self.session_active = True
        self.awaiting_start_ack = False
        self.awaiting_stop_ack = False
        self.pending_since_s = None
        self.session_state = session_state

    def _mark_idle(self) -> None:
        self.session_active = False
        self.awaiting_start_ack = False
        self.awaiting_stop_ack = False
        self.pending_since_s = None
        self.current_dirty = False
        self.session_state = PadSessionState.IDLE

    def record_tx(self, line: str, now_s: Optional[float] = None) -> None:
        self.last_tx_line = line
        self.last_tx_s = time.monotonic() if now_s is None else now_s

    def update_from_ascii_reply(self, line: str) -> None:
        self.last_ascii_reply = line

        if line.startswith("OK session_started"):
            # Device is in CONFIG_WAIT — port owned but telemetry not started yet.
            # Do NOT mark session_active yet; SENSOR_START and CONTROL_START still needed.
            self.awaiting_start_ack = False
            self.awaiting_stop_ack = False
            self.pending_since_s = None
            self.session_active = False
            self.requested_current_a = 0.0
            self.current_dirty = False
            self.session_state = PadSessionState.CONFIG_WAIT
            return

        if line.startswith("OK sensor_streaming"):
            # Device started binary telemetry — control not armed yet.
            self.session_state = PadSessionState.SENSOR_STREAMING
            return

        if line.startswith("OK control_started") or line.startswith("OK control_rearmed"):
            # CONTROL_START acknowledged — armed, waiting for first CURRENT/TORQUE sample.
            self._mark_streaming(PadSessionState.STREAMING)
            return

        if line.startswith("OK session_stopped"):
            self._mark_idle()
            return

        if line.startswith("OK state="):
            # Response to PADCTRL STATUS
            state_token = _extract_key_value(line, "state")
            if state_token in ("SENSOR_STREAMING", "CONTROL_ACTIVE"):
                self._mark_streaming(PadSessionState.STREAMING)
            elif state_token == "CONTROL_STALE":
                self._mark_streaming(PadSessionState.STALE)
            else:
                self._mark_idle()

            last_seq_text = _extract_key_value(line, "last_seq")
            if last_seq_text is not None:
                try:
                    self.last_seq = int(last_seq_text)
                except ValueError:
                    pass

            # STATUS response uses "value=" not "current="
            value_text = _extract_key_value(line, "value")
            if value_text is not None:
                try:
                    self.requested_current_a = float(value_text)
                except ValueError:
                    pass
            return

        if line.startswith("ERR "):
            self.awaiting_start_ack = False
            self.awaiting_stop_ack = False
            self.pending_since_s = None
            if not self.session_active:
                self.current_dirty = False
                self.session_state = PadSessionState.FAULT

    def update_from_telemetry(self, telemetry: PadTelemetry, now_s: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now_s is None else now_s
        self.latest_telemetry = telemetry
        self.telemetry_valid = True
        self.last_seq = telemetry.seq
        self.last_telemetry_s = timestamp
        # Do NOT touch session_state or session_active here.
        # State transitions are driven by ASCII replies (OK control_started etc.),
        # not by telemetry arrival. Mixing them caused SENSOR_STREAMING to be
        # silently overwritten when a binary frame raced the ASCII ACK.

    def refresh_timeouts(
        self,
        now_s: Optional[float] = None,
        ack_timeout_s: float = 1.0,
        telemetry_stale_s: float = 0.25,
    ) -> None:
        timestamp = time.monotonic() if now_s is None else now_s

        if self.pending_since_s is not None and (timestamp - self.pending_since_s) >= ack_timeout_s:
            start_timed_out = self.awaiting_start_ack
            stop_timed_out = self.awaiting_stop_ack
            self.awaiting_start_ack = False
            self.awaiting_stop_ack = False
            self.pending_since_s = None

            if start_timed_out:
                self.session_active = False
                self.current_dirty = False
                self.session_state = PadSessionState.FAULT
            elif stop_timed_out:
                if self.telemetry_is_fresh(timestamp, telemetry_stale_s):
                    self.session_active = True
                    self.session_state = PadSessionState.STREAMING
                else:
                    self.session_active = False
                    self.current_dirty = False
                    self.session_state = PadSessionState.IDLE

        if self.session_active and self.telemetry_valid and not self.telemetry_is_fresh(
            timestamp, telemetry_stale_s
        ):
            self.session_state = PadSessionState.STALE

    def telemetry_is_fresh(self, now_s: Optional[float] = None, telemetry_stale_s: float = 0.25) -> bool:
        if not self.telemetry_valid or self.last_telemetry_s is None:
            return False
        timestamp = time.monotonic() if now_s is None else now_s
        return (timestamp - self.last_telemetry_s) < telemetry_stale_s


def _extract_key_value(line: str, key: str) -> Optional[str]:
    token = f"{key}="
    start = line.find(token)
    if start < 0:
        return None
    value_start = start + len(token)
    value_end = line.find(" ", value_start)
    if value_end < 0:
        value_end = len(line)
    return line[value_start:value_end]


def cobs_decode(data: bytes) -> bytes:
    if not data:
        return b""

    output = bytearray()
    read_index = 0
    length = len(data)

    while read_index < length:
        code = data[read_index]
        read_index += 1
        if code == 0:
            raise PadProtocolError("COBS decode failed: zero code byte")

        for _ in range(1, code):
            if read_index >= length:
                raise PadProtocolError("COBS decode failed: truncated block")
            output.append(data[read_index])
            read_index += 1

        if code < 0xFF and read_index < length:
            output.append(0)

    return bytes(output)


def cobs_encode(data: bytes) -> bytes:
    output = bytearray()
    code_index = 0
    output.append(0)
    code = 1

    for value in data:
        if value == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
            continue

        output.append(value)
        code += 1
        if code == 0xFF:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1

    output[code_index] = code
    return bytes(output)


class PadLinkStreamParser:
    def __init__(self, ascii_limit: int = 192, binary_limit: int = 256) -> None:
        self.ascii_limit = ascii_limit
        self.binary_limit = binary_limit
        self.ascii_buffer = bytearray()
        self.binary_buffer = bytearray()
        self.binary_frame_active = False
        # UART 바이트 유실/노이즈로 깨진 프레임 수 (예외 대신 버리고 재동기화)
        self.dropped_frames = 0
        self.last_drop_reason: Optional[str] = None

    def feed(self, data: bytes) -> List[PadEvent]:
        events: List[PadEvent] = []

        for byte in data:
            if self.binary_frame_active:
                if byte == 0:
                    if self.binary_buffer:
                        # 프레임 하나가 깨져도 스트림 전체를 죽이지 않는다.
                        # 해당 프레임만 버리고 다음 구분자에서 재동기화한다.
                        try:
                            events.append(
                                PadTelemetryFrame(self._decode_binary_frame(bytes(self.binary_buffer)))
                            )
                        except PadProtocolError as exc:
                            self.dropped_frames += 1
                            self.last_drop_reason = str(exc)
                    self.binary_buffer.clear()
                    self.binary_frame_active = False
                    continue

                if len(self.binary_buffer) >= self.binary_limit:
                    self.binary_buffer.clear()
                    self.binary_frame_active = False
                    self.dropped_frames += 1
                    self.last_drop_reason = f"Binary frame overflow: limit={self.binary_limit}"
                    continue

                self.binary_buffer.append(byte)
                continue

            if byte == 0:
                if self.ascii_buffer:
                    implicit = self._try_decode_implicit_binary(bytes(self.ascii_buffer))
                    self.ascii_buffer.clear()
                    if implicit is not None:
                        events.append(PadTelemetryFrame(implicit))
                    continue

                self.binary_frame_active = True
                self.binary_buffer.clear()
                continue

            if byte == 13:
                continue

            if byte == 10:
                if self.ascii_buffer:
                    line = self.ascii_buffer.decode("utf-8", errors="replace")
                    self.ascii_buffer.clear()
                    events.append(PadAsciiReply(line))
                continue

            if len(self.ascii_buffer) >= self.ascii_limit:
                self.ascii_buffer.clear()
                self.dropped_frames += 1
                self.last_drop_reason = f"ASCII buffer overflow: limit={self.ascii_limit}"

            self.ascii_buffer.append(byte)

        return events

    def _decode_binary_frame(self, encoded_payload: bytes) -> PadTelemetry:
        decoded = cobs_decode(encoded_payload)
        return PadTelemetry.from_payload(decoded)

    def _try_decode_implicit_binary(self, encoded_payload: bytes) -> Optional[PadTelemetry]:
        try:
            return self._decode_binary_frame(encoded_payload)
        except PadProtocolError:
            return None


def clamp_current_a(current_a: float, limit_a: float = 13.0) -> float:
    if current_a < 0.0:
        return 0.0
    if current_a > limit_a:
        return limit_a
    return current_a


def force_to_current_a(force_value: float) -> float:
    if force_value <= 0.0:
        return 0.0
    return clamp_current_a((force_value + 0.04) / 0.87)


def format_padctrl_current(current_a: float) -> str:
    return f"PADCTRL CURRENT {current_a:.2f}"


class PadLinkController:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        read_timeout_s: float = 0.01,
        ack_timeout_s: float = 1.0,
        telemetry_stale_s: float = 0.25,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.read_timeout_s = read_timeout_s
        self.ack_timeout_s = ack_timeout_s
        self.telemetry_stale_s = telemetry_stale_s
        self.state = PadLinkState()
        self.parser = PadLinkStreamParser()
        self._serial = None

    @property
    def serial_port(self):
        if self._serial is None:
            raise RuntimeError("Serial port is not open")
        return self._serial

    def open(self, boot_delay_s: float = 2.0) -> None:
        if self._serial is not None:
            return

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for live serial use. Install it with "
                "`python -m pip install pyserial`."
            ) from exc

        # Open with port=None first so we can set DTR=False before the port
        # opens — prevents ESP32/Arduino reset triggered by the DTR edge.
        s = serial.Serial(
            baudrate=self.baudrate,
            timeout=self.read_timeout_s,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        s.dtr = False
        s.port = self.port
        s.open()
        self._serial = s

        if boot_delay_s > 0:
            time.sleep(boot_delay_s)

    def close(self) -> None:
        if self._serial is None:
            return
        self._serial.close()
        self._serial = None

    def poll(self) -> List[PadEvent]:
        waiting = getattr(self.serial_port, "in_waiting", 0)
        chunk = self.serial_port.read(waiting or 1)
        events = self.parser.feed(chunk)
        now_s = time.monotonic()

        for event in events:
            if isinstance(event, PadAsciiReply):
                self.state.update_from_ascii_reply(event.line)
            else:
                self.state.update_from_telemetry(event.telemetry, now_s=now_s)

        self.state.refresh_timeouts(
            now_s=now_s,
            ack_timeout_s=self.ack_timeout_s,
            telemetry_stale_s=self.telemetry_stale_s,
        )
        return events

    def send_line(self, line: str) -> None:
        payload = (line.rstrip("\r\n") + "\n").encode("ascii")
        self.serial_port.write(payload)
        self.serial_port.flush()
        self.state.record_tx(line)

    def send_padctrl_start(self) -> None:
        self.state.awaiting_start_ack = True
        self.state.awaiting_stop_ack = False
        self.state.pending_since_s = time.monotonic()
        self.state.session_active = False
        self.state.requested_current_a = 0.0
        self.state.current_dirty = False
        self.state.session_state = PadSessionState.AWAITING_START_ACK
        self.send_line("PADCTRL START")

    def send_padctrl_padmode_enable(self) -> None:
        self.send_line("PADCTRL PADMODE ENABLE")

    def send_padctrl_data_mode_current_only(self) -> None:
        self.send_line("PADCTRL DATA_MODE CURRENT_ONLY")

    def send_padctrl_sensor_start(self) -> None:
        self.send_line("PADCTRL SENSOR_START")

    def send_padctrl_control_start(self) -> None:
        self.send_line("PADCTRL CONTROL_START")

    def send_padctrl_stop(self) -> None:
        self.state.awaiting_start_ack = False
        self.state.awaiting_stop_ack = True
        self.state.pending_since_s = time.monotonic()
        self.state.current_dirty = False
        self.state.session_state = PadSessionState.AWAITING_STOP_ACK
        self.send_line("PADCTRL STOP")

    def send_padctrl_status(self) -> None:
        self.send_line("PADCTRL STATUS")

    def send_padctrl_current(self, current_a: float) -> None:
        current = clamp_current_a(float(current_a))
        self.state.requested_current_a = current
        self.state.current_dirty = True
        self.send_line(format_padctrl_current(current))
        self.state.current_dirty = False

    def send_padctrl_torque(self, torque_nm: float) -> None:
        torque = float(torque_nm)
        self.state.requested_current_a = torque
        self.state.current_dirty = True
        self.send_line(f"PADCTRL TORQUE {torque:.3f}")
        self.state.current_dirty = False

    def wait_until(
        self,
        predicate: Callable[[PadLinkState], bool],
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.poll()
            if predicate(self.state):
                return True
        return predicate(self.state)

    def start_session(self, timeout_s: float = 2.0) -> bool:
        # Pre-flight: flush any stale firmware session left from a previous run.
        # PADCTRL STOP is safe from any state (firmware returns OK session_stopped
        # even when already IDLE), so this is always harmless.
        self.state.session_state = PadSessionState.AWAITING_STOP_ACK
        self.send_line("PADCTRL STOP")
        self.wait_until(lambda s: s.session_state == PadSessionState.IDLE, 1.0)

        # Step 1: claim the port → CONFIG_WAIT
        self.send_padctrl_start()
        if not self.wait_until(
            lambda s: s.session_state == PadSessionState.CONFIG_WAIT, timeout_s
        ):
            return False

        # Step 2: switch to current-unit control (firmware default is TORQUE/DEBUG_V1)
        self.send_padctrl_data_mode_current_only()

        # Step 3: start binary telemetry → SENSOR_STREAMING
        # Also accept STREAMING in case a binary frame races the ASCII ACK.
        self.send_padctrl_sensor_start()
        if not self.wait_until(
            lambda s: s.session_state in (
                PadSessionState.SENSOR_STREAMING, PadSessionState.STREAMING
            ),
            timeout_s,
        ):
            return False

        # Step 3: arm control path → session_active=True (OK control_started)
        self.send_padctrl_control_start()
        return self.wait_until(lambda s: s.session_active, timeout_s)

    def stop_session(self, timeout_s: float = 2.0) -> bool:
        self.send_padctrl_stop()
        return self.wait_until(lambda state: not state.session_active, timeout_s)

    def run_control_loop(
        self,
        algorithm: Callable[[PadTelemetry, PadLinkState], Optional[float]],
        *,
        control_period_s: float = 0.01,
        duration_s: Optional[float] = None,
        start_session: bool = True,
        stop_on_exit: bool = True,
        current_limit_a: float = 13.0,
        on_event: Optional[Callable[[PadEvent, PadLinkState], None]] = None,
        on_current_command: Optional[Callable[[float, PadTelemetry, PadLinkState], None]] = None,
    ) -> None:
        if start_session and not self.start_session():
            raise RuntimeError(
                f"PAD session start timed out. Last reply: {self.state.last_ascii_reply!r}"
            )

        loop_start = time.monotonic()
        next_control_s = loop_start

        try:
            while True:
                for event in self.poll():
                    if on_event is not None:
                        on_event(event, self.state)

                now_s = time.monotonic()
                if duration_s is not None and (now_s - loop_start) >= duration_s:
                    break

                if (
                    now_s >= next_control_s
                    and self.state.session_active
                    and self.state.latest_telemetry is not None
                    and self.state.telemetry_is_fresh(now_s, self.telemetry_stale_s)
                ):
                    command = algorithm(self.state.latest_telemetry, self.state)
                    if command is not None:
                        clamped = max(-current_limit_a, min(current_limit_a, float(command)))
                        self.send_padctrl_current(clamped)
                        if on_current_command is not None:
                            on_current_command(clamped, self.state.latest_telemetry, self.state)

                    next_control_s += control_period_s
                    while next_control_s < now_s:
                        next_control_s += control_period_s
                else:
                    time.sleep(min(control_period_s / 5.0, 0.002))
        finally:
            if stop_on_exit and self._serial is not None and (
                self.state.session_active
                or self.state.awaiting_start_ack
                or self.state.awaiting_stop_ack
            ):
                try:
                    self.stop_session(timeout_s=1.0)
                except Exception:
                    pass


def list_serial_ports() -> Sequence[str]:
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial port discovery. Install it with "
            "`python -m pip install pyserial`."
        ) from exc

    ports = list(list_ports.comports())
    ports.sort(key=lambda port: (
        0 if "USB" in port.device or "ACM" in port.device else 1,
        port.device,
    ))
    return [port.device for port in ports]


def build_test_frame(telemetry: PadTelemetry, with_leading_delimiter: bool = True) -> bytes:
    encoded = cobs_encode(telemetry.to_payload())
    if with_leading_delimiter:
        return b"\x00" + encoded + b"\x00"
    return encoded + b"\x00"


def run_self_test() -> None:
    parser = PadLinkStreamParser()
    state = PadLinkState()

    telemetry = PadTelemetry(
        type=PAD_TELEMETRY_TYPE,
        version=PAD_TELEMETRY_VERSION,
        seq=12,
        time10ms=3456,
        motor_pos_x100=123,
        motor_vel_x100=-456,
        motor_iq_meas_x100=78,
        motor_iq_set_x100=90,
        belt_length_x10=1034,
        belt_vel_x10=-12,
        belt_acc_x1=5,
        roll_x10=12,
        pitch_x10=-8,
        yaw_x10=34,
        gax_x100=1,
        gay_x100=2,
        gaz_x100=3,
        ax_x100=4,
        ay_x100=5,
        az_x100=6,
        input_force_x100=15,
        output_force_x100=12,
        spring_torque_x100=99,
        bus_voltage_x100=2450,
        bus_current_x100=125,
        driver_error=0,
        detect_state=7,
        control_state=2,
    )

    mixed_stream = (
        b"OK session_started\n"
        + build_test_frame(telemetry, with_leading_delimiter=True)
        + b"OK current=0.25\n"
        + build_test_frame(telemetry, with_leading_delimiter=False)
        + b"OK session_stopped\n"
    )

    events = parser.feed(mixed_stream)
    if len(events) != 5:
        raise AssertionError(f"Unexpected event count: {len(events)}")

    for event in events:
        if isinstance(event, PadAsciiReply):
            state.update_from_ascii_reply(event.line)
        else:
            state.update_from_telemetry(event.telemetry)

    if state.last_seq != telemetry.seq:
        raise AssertionError(f"Unexpected last seq: {state.last_seq}")
    if abs(telemetry.belt_length - 103.4) > 1e-6:
        raise AssertionError(f"Unexpected belt length: {telemetry.belt_length}")
    if abs(force_to_current_a(0.30) - ((0.30 + 0.04) / 0.87)) > 1e-6:
        raise AssertionError("force_to_current_a conversion mismatch")

    print("[self-test] parser events:", len(events))
    print("[self-test] telemetry summary:", telemetry.summary())
    print("[self-test] final session state:", state.session_state.value)
