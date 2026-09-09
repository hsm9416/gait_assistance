#!/usr/bin/env python3
"""
PAD 보행보조 시리얼 제어

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다른 코드에서 임포트해서 사용하는 방법:

    from connect import PadController, PadSensorData

    ctrl = PadController(port="/dev/ttyUSB0")
    with ctrl:
        ctrl.start_session()
        while True:
            data = ctrl.poll()
            if data:
                print(f"roll={data.roll:.1f}  pitch={data.pitch:.1f}  "
                      f"gyro_x={data.gyro_x:.1f}  F_in={data.input_force:.2f}N")
                ctrl.set_current(1.0)

PadSensorData 필드:
    seq, time_s                          — 시퀀스 번호 / 장치 내부 시간(s)
    motor_pos, motor_vel                 — 모터 위치/속도
    motor_iq_meas, motor_iq_set          — 모터 실측/설정 전류(A)
    belt_length, belt_velocity,
        belt_acceleration                — 벨트 길이(mm)/속도/가속도
    roll, pitch, yaw                     — IMU 자세(°)
    gyro_x, gyro_y, gyro_z              — IMU 각속도(°/s)
    accel_x, accel_y, accel_z           — IMU 가속도
    input_force, output_force            — 입력/출력힘(N)
    spring_torque                        — 스프링 토크(N·m)
    bus_voltage, bus_current             — 버스 전압(V)/전류(A)
    driver_error, detect_state,
        control_state                    — 상태 코드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLI 실행:
    python connect.py --port /dev/ttyUSB0
    python connect.py --port /dev/ttyUSB0 --duration 30
    python connect.py --list-ports
    python connect.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pad_external_control_lib import (
    PadAsciiReply,
    PadEvent,
    PadLinkController,
    PadLinkState,
    PadTelemetry,
    PadTelemetryFrame,
    force_to_current_a,
    list_serial_ports,
    run_self_test,
)


# ──────────────────────────────────────────────────────────────────────────────
# 센서 데이터 — PadTelemetry 전체 필드를 실제 단위로 변환
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PadSensorData:
    """PAD 장치에서 수신한 전체 텔레메트리 (실제 단위 변환 완료)."""

    # 타이밍
    seq: int
    time_s: float            # 장치 내부 시간 (s)

    # 모터
    motor_pos: float         # 모터 위치
    motor_vel: float         # 모터 속도
    motor_iq_meas: float     # 모터 실측 전류 (A)
    motor_iq_set: float      # 모터 설정 전류 (A)

    # 벨트
    belt_length: float       # 벨트 길이 (mm)
    belt_velocity: float     # 벨트 속도 (mm/s)
    belt_acceleration: float # 벨트 가속도 (mm/s²)

    # IMU — 자세 (오일러각)
    roll: float              # 롤  (°)
    pitch: float             # 피치 (°)
    yaw: float               # 요  (°)

    # IMU — 각속도 (자이로스코프)
    gyro_x: float            # X축 각속도 (°/s)
    gyro_y: float            # Y축 각속도 (°/s)
    gyro_z: float            # Z축 각속도 (°/s)

    # IMU — 가속도계
    accel_x: float           # X축 가속도
    accel_y: float           # Y축 가속도
    accel_z: float           # Z축 가속도

    # 힘 / 토크
    input_force: float       # 입력힘 (N)
    output_force: float      # 출력힘 (N)
    spring_torque: float     # 스프링 토크 (N·m)

    # 전원
    bus_voltage: float       # 버스 전압 (V)
    bus_current: float       # 버스 전류 (A)

    # 상태 코드
    driver_error: int
    detect_state: int
    control_state: int

    @classmethod
    def from_telemetry(cls, t: PadTelemetry) -> "PadSensorData":
        return cls(
            seq=t.seq,
            time_s=t.time10ms / 100.0,
            motor_pos=t.motor_pos,
            motor_vel=t.motor_vel,
            motor_iq_meas=t.motor_iq_meas,
            motor_iq_set=t.motor_iq_set,
            belt_length=t.belt_length,
            belt_velocity=t.belt_velocity,
            belt_acceleration=t.belt_acceleration,
            roll=t.roll,
            pitch=t.pitch,
            yaw=t.yaw,
            gyro_x=t.gax_x100 / 100.0,
            gyro_y=t.gay_x100 / 100.0,
            gyro_z=t.gaz_x100 / 100.0,
            accel_x=t.ax_x100 / 100.0,
            accel_y=t.ay_x100 / 100.0,
            accel_z=t.az_x100 / 100.0,
            input_force=t.input_force,
            output_force=t.output_force,
            spring_torque=t.spring_torque,
            bus_voltage=t.bus_voltage,
            bus_current=t.bus_current,
            driver_error=t.driver_error,
            detect_state=t.detect_state,
            control_state=t.control_state,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 고수준 컨트롤러 — 다른 코드에서 import해서 사용
# ──────────────────────────────────────────────────────────────────────────────

class PadController:
    """
    PAD 장치 고수준 인터페이스.

    단순 폴링::

        ctrl = PadController(port="/dev/ttyUSB0")
        with ctrl:
            ctrl.start_session()
            while True:
                data = ctrl.poll()
                if data:
                    ctrl.set_current(my_algo(data))

    블로킹 제어 루프::

        def my_algo(data: PadSensorData, ctrl: PadController) -> float:
            return force_to_current_a(data.input_force * 0.5)

        with PadController(port="/dev/ttyUSB0") as ctrl:
            ctrl.run_control_loop(my_algo, duration_s=30)
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        read_timeout_s: float = 0.01,
        ack_timeout_s: float = 1.0,
        telemetry_stale_s: float = 0.25,
    ) -> None:
        self._link = PadLinkController(
            port=port,
            baudrate=baudrate,
            read_timeout_s=read_timeout_s,
            ack_timeout_s=ack_timeout_s,
            telemetry_stale_s=telemetry_stale_s,
        )
        self._latest: Optional[PadSensorData] = None

    # ── 연결 관리 ──────────────────────────────────────────────────────────

    def open(self) -> None:
        """시리얼 포트 열기."""
        self._link.open()

    def close(self) -> None:
        """시리얼 포트 닫기."""
        self._link.close()

    def __enter__(self) -> "PadController":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── 세션 관리 ──────────────────────────────────────────────────────────

    def start_session(self, timeout_s: float = 2.0) -> bool:
        """스트리밍 세션 시작. 성공 시 True 반환."""
        return self._link.start_session(timeout_s)

    def stop_session(self, timeout_s: float = 2.0) -> bool:
        """스트리밍 세션 종료. 성공 시 True 반환."""
        return self._link.stop_session(timeout_s)

    # ── 데이터 수신 ────────────────────────────────────────────────────────

    def poll(self) -> Optional[PadSensorData]:
        """
        시리얼 버퍼를 읽고 새 텔레메트리 프레임이 있으면 PadSensorData 반환.
        새 프레임이 없으면 None 반환. sensor_data 프로퍼티도 함께 갱신됨.
        """
        events = self._link.poll()
        new_data: Optional[PadSensorData] = None
        for event in events:
            if isinstance(event, PadTelemetryFrame):
                new_data = PadSensorData.from_telemetry(event.telemetry)
                self._latest = new_data
        return new_data

    @property
    def sensor_data(self) -> Optional[PadSensorData]:
        """poll() 호출 없이 가장 최근에 수신된 센서 데이터에 접근."""
        return self._latest

    # ── 상태 조회 ──────────────────────────────────────────────────────────

    @property
    def is_streaming(self) -> bool:
        """스트리밍 세션 활성 여부."""
        return self._link.state.session_active

    @property
    def session_state(self):
        """현재 세션 상태 (PadSessionState enum)."""
        return self._link.state.session_state

    @property
    def link_state(self) -> PadLinkState:
        """저수준 링크 상태 전체 접근."""
        return self._link.state

    @property
    def dropped_frames(self) -> int:
        """UART 바이트 유실 등으로 버려진 텔레메트리 프레임 수."""
        return self._link.parser.dropped_frames

    # ── 제어 명령 ──────────────────────────────────────────────────────────

    def set_current(self, current_a: float) -> None:
        """전류 명령 전송 (A)."""
        self._link.send_padctrl_current(current_a)

    def set_force(self, force_n: float) -> None:
        """보조힘 목표값(N)으로 전류 명령 전송 (N → A 자동 변환)."""
        self._link.send_padctrl_current(force_to_current_a(force_n))

    def request_status(self) -> None:
        """장치에 STATUS 쿼리 전송."""
        self._link.send_padctrl_status()

    # ── 블로킹 제어 루프 ────────────────────────────────────────────────────

    def run_control_loop(
        self,
        algorithm: Callable[["PadSensorData", "PadController"], Optional[float]],
        *,
        control_period_s: float = 0.01,
        duration_s: Optional[float] = None,
        start_session: bool = True,
        stop_on_exit: bool = True,
        current_limit_a: float = 13.0,
        on_data: Optional[Callable[["PadSensorData", "PadController"], None]] = None,
        on_ascii_reply: Optional[Callable[[str, "PadController"], None]] = None,
    ) -> None:
        """
        블로킹 제어 루프.

        Args:
            algorithm:      (data, ctrl) → float(A) | None.
                            None 반환 시 해당 주기에는 전류 명령 전송 안 함.
            on_data:        새 텔레메트리 수신마다 호출되는 콜백.
            on_ascii_reply: ASCII 응답 수신마다 호출되는 콜백.
        """
        def _on_event(event: PadEvent, _: PadLinkState) -> None:
            if isinstance(event, PadTelemetryFrame):
                data = PadSensorData.from_telemetry(event.telemetry)
                self._latest = data
                if on_data is not None:
                    on_data(data, self)
            elif isinstance(event, PadAsciiReply):
                if on_ascii_reply is not None:
                    on_ascii_reply(event.line, self)

        def _algo(*_: object) -> Optional[float]:
            if self._latest is None:
                return None
            result = algorithm(self._latest, self)
            if result is not None:
                return max(-current_limit_a, min(current_limit_a, float(result)))
            return None

        def _on_current_command(current_a: float, *_) -> None:
            print(f"[CMD] current={current_a:.3f}A")

        self._link.run_control_loop(
            _algo,
            control_period_s=control_period_s,
            duration_s=duration_s,
            start_session=start_session,
            stop_on_exit=stop_on_exit,
            current_limit_a=current_limit_a,
            on_event=_on_event,
            on_current_command=_on_current_command,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 알고리즘 파라미터 (여기서 조정)
# ──────────────────────────────────────────────────────────────────────────────

FORCE_GAIN      = 0.5   # 입력힘 → 보조힘 비율  (0.0 ~ 1.0)
CURRENT_LIMIT_A = 8.0   # 안전 전류 상한 (A)
DEADBAND_N      = 0.05  # 입력힘 데드밴드 (N) — 잡음 제거
SMOOTH_ALPHA    = 0.2   # 저역통과 필터 계수   (0=무변화, 1=필터 없음)


# ──────────────────────────────────────────────────────────────────────────────
# 알고리즘 내부 상태
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _AlgoState:
    smoothed_force: float = 0.0
    prev_seq: int = -1


_state = _AlgoState()


# ──────────────────────────────────────────────────────────────────────────────
# 제어 알고리즘 — 이 함수만 교체해서 알고리즘 변경 가능
# ──────────────────────────────────────────────────────────────────────────────

def compute_current_a(data: PadSensorData, ctrl: PadController) -> Optional[float]:
    """
    입력힘 비례 보행보조 알고리즘.

    흐름:
      입력힘(N)  →  데드밴드 제거  →  EMA 저역통과  →  보조힘 스케일링  →  전류(A)

    Returns:
      전류 명령(A). None 반환 시 해당 주기에는 명령을 보내지 않음.
    """
    if data.seq == _state.prev_seq:
        return None
    _state.prev_seq = data.seq

    raw_force = data.input_force
    if raw_force < DEADBAND_N:
        raw_force = 0.0

    _state.smoothed_force = (
        SMOOTH_ALPHA * raw_force + (1.0 - SMOOTH_ALPHA) * _state.smoothed_force
    )

    desired_force = _state.smoothed_force * FORCE_GAIN
    return force_to_current_a(desired_force)


# ──────────────────────────────────────────────────────────────────────────────
# 콜백: 수신 데이터 출력 (전체 센서 데이터)
# ──────────────────────────────────────────────────────────────────────────────

def _on_data(data: PadSensorData, ctrl: PadController) -> None:
    print(
        f"[TEL] seq={data.seq:5d}  t={data.time_s:8.2f}s  "
        f"sess={ctrl.session_state.value}\n"
        f"      belt={data.belt_length:7.1f}mm  vel={data.belt_velocity:6.1f}  acc={data.belt_acceleration:6.1f}\n"
        f"      roll={data.roll:6.1f}°  pitch={data.pitch:6.1f}°  yaw={data.yaw:6.1f}°\n"
        f"      gyro=({data.gyro_x:6.1f}, {data.gyro_y:6.1f}, {data.gyro_z:6.1f}) °/s\n"
        f"      accel=({data.accel_x:.3f}, {data.accel_y:.3f}, {data.accel_z:.3f})\n"
        f"      F_in={data.input_force:5.2f}N  F_out={data.output_force:5.2f}N  "
        f"torque={data.spring_torque:.2f}N·m\n"
        f"      motor_pos={data.motor_pos:.2f}  motor_vel={data.motor_vel:.2f}  "
        f"iq_meas={data.motor_iq_meas:.2f}A  iq_set={data.motor_iq_set:.2f}A\n"
        f"      V={data.bus_voltage:.2f}V  I={data.bus_current:.2f}A  "
        f"err={data.driver_error}  detect={data.detect_state}  ctrl_st={data.control_state}"
    )


def _on_ascii_reply(line: str, ctrl: PadController) -> None:
    print(f"[RX ] {line}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PAD 보행보조 시리얼 제어",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", help="직렬 포트 (예: /dev/ttyUSB0, COM3)")
    p.add_argument("--baud", type=int, default=115200, help="보드레이트")
    p.add_argument(
        "--duration", type=float, default=None,
        help="실행 시간(초). 생략하면 Ctrl+C 까지 실행",
    )
    p.add_argument(
        "--control-period-ms", type=float, default=10.0,
        help="제어 루프 주기 (ms)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="텔레메트리 로그 숨기기 (ASCII 응답만 출력)",
    )
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
            for i, p in enumerate(ports):
                print(f"  [{i}] {p}")
            try:
                idx = int(input("포트 번호 선택: "))
                args.port = ports[idx]
            except (ValueError, IndexError):
                print("오류: 올바른 번호를 입력하세요.")
                return 1

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
            compute_current_a,
            control_period_s=args.control_period_ms / 1000.0,
            duration_s=args.duration,
            start_session=True,
            stop_on_exit=True,
            current_limit_a=CURRENT_LIMIT_A,
            on_data=None if args.quiet else _on_data,
            on_ascii_reply=_on_ascii_reply,
        )

    except OSError as exc:
        # USB-UART 브리지가 사라진 경우 (장치 재열거, 케이블 접촉 불량, 전원 글리치).
        # pyserial의 SerialException도 OSError 서브클래스라 여기서 함께 잡힌다.
        print(f"[ERROR] 시리얼 장치 연결 끊김: {exc}")
        print("[HINT] USB 재열거 여부는 `dmesg | tail` 의 'USB disconnect' 로 확인하세요.")
        return 1
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except KeyboardInterrupt:
        print("[INFO] 사용자 중단 (Ctrl+C)")
    finally:
        dropped = ctrl.dropped_frames
        if dropped:
            print(f"[WARN] 손상되어 버린 프레임: {dropped}개 (UART 바이트 유실)")
        try:
            ctrl.close()
            print("[INFO] 시리얼 포트 종료")
        except OSError as exc:
            print(f"[WARN] 포트 종료 중 오류 (장치가 이미 사라짐): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
