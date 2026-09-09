#!/usr/bin/env python3
"""Adaptive left/right IMU heel-strike detection and torque assistance."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from connect import PadController
from pad_external_control_lib import list_serial_ports


CSV_DIR = Path("csv")
NOMINAL_GAIT_CYCLE_S = 1.2
ASSIST_REFRACTORY_RATIO = 0.65
ACC_PEAK_DROP_RATIO = 0.12
ACC_PEAK_AMPLITUDE_RATIO = 1.15
ACC_PEAK_TIMEOUT_S = 0.12
FIELDS = [
    "host_time_s",
    "seq",
    "device_time_s",
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "belt_length",
    "belt_velocity",
    "belt_acceleration",
    "motor_pos",
    "motor_vel",
    "motor_iq_meas",
    "motor_iq_set",
    "acc_mag",
    "gyro_mag",
    "gyro_corrected",
    "gyro_side_score",
    "gyro_side_separation",
    "dynamic_refractory_s",
    "step_period_s",
    "gait_cycle_s",
    "cadence_spm",
    "gait_speed_ratio",
    "assist_pulse_s",
    "acc_residual",
    "acc_impact_peak",
    "acc_peak_rise_s",
    "acc_peak_confirm_delay_s",
    "acc_impact_time_s",
    "acc_peak_candidate",
    "threshold",
    "acc_threshold_percentile",
    "gyro_threshold",
    "belt_threshold",
    "belt_ok",
    "belt_ready",
    "calibrated",
    "imu_side",
    "imu_heel_strike",
    "heel_strike",
    "hs_count",
    "left_hs_count",
    "right_hs_count",
    "torque_cmd_nm",
]


class AdaptiveImuHeelStrikeDetector:
    def __init__(
        self,
        *,
        calibration_s: float,
        acc_threshold: float,
        gyro_threshold: float,
        belt_threshold: float,
        belt_side: str,
        use_gyro_gate: bool = True,
        use_acc_peak: bool = True,
        acc_percentile: float = 82.0,
        refractory_s: float = 0.45,
        alpha: float = 0.02,
    ) -> None:
        self.calibration_s = calibration_s
        self.acc_threshold = acc_threshold
        self.gyro_threshold = gyro_threshold
        self.belt_threshold = belt_threshold
        self.belt_side = belt_side
        self.use_gyro_gate = use_gyro_gate
        self.use_acc_peak = use_acc_peak
        self.acc_percentile = acc_percentile
        self.refractory_s = refractory_s
        self.alpha = alpha
        self.ready = False
        self.calibration_start_t: float | None = None
        self.last_calibration_attempt_t = -999.0
        self.samples: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        self.gravity = np.array([0.0, 0.0, 1.0])
        self.gyro_bias = np.zeros(3)
        self.gyro_axis = np.array([1.0, 0.0, 0.0])
        self.gyro_side_axis = np.array([1.0, 0.0, 0.0])
        self.gyro_side_center = np.zeros(3)
        self.gyro_side_scale = 1.0
        self.step_period_s = 0.6
        self.gait_cycle_intervals: deque[float] = deque(maxlen=9)
        self.dynamic_refractory_s = refractory_s
        self.side_confidence_limit = 0.75
        self.belt_lower = belt_threshold < 0
        self.belt_reset_threshold = belt_threshold
        self.belt_rearmed = True
        self.acc_baseline: float | None = None
        self.prev_residual = 0.0
        self.acc_peak_candidate = False
        self.acc_peak_candidate_start_t = -999.0
        self.acc_peak_candidate_threshold = acc_threshold
        self.acc_peak_candidate_value = 0.0
        self.acc_peak_candidate_time_s = -999.0
        self.acc_peak_candidate_gyro = 0.0
        self.acc_peak_candidate_side_gyro = np.zeros(3)
        self.acc_peak_value = 0.0
        self.acc_peak_rise_s = 0.0
        self.acc_peak_delay_s = 0.0
        self.acc_impact_time_s = -999.0
        self.gyro_corrected = 0.0
        self.gyro_side_score = 0.0
        self.gyro_window: deque[tuple[float, np.ndarray]] = deque()
        self.motion_gyro_window: deque[tuple[float, float]] = deque()
        self.recent_gyro: deque[tuple[float, float]] = deque()
        self.acc_history: deque[float] = deque(maxlen=1000)
        self.gyro_history: deque[float] = deque(maxlen=1000)
        self.last_adapt_t = 0.0
        self.last_hs_t = -999.0
        self.last_side: str | None = None
        self.last_side_t = -999.0
        self.last_confident_side_t = {"left": -999.0, "right": -999.0}
        self.count = 0
        self.left_count = 0
        self.right_count = 0

    @staticmethod
    def _percentile(values: np.ndarray | list[float], q: float) -> float:
        return float(np.percentile(values, q))

    @staticmethod
    def _smooth(values: np.ndarray, samples: int) -> np.ndarray:
        samples = max(1, samples)
        result = np.empty_like(values)
        total = 0.0
        for i, value in enumerate(values):
            total += float(value)
            if i >= samples:
                total -= float(values[i - samples])
            result[i] = total / min(i + 1, samples)
        return result

    def _find_candidates(
        self,
        times: np.ndarray,
        residuals: np.ndarray,
        refractory_s: float | None = None,
    ) -> list[int]:
        candidates: list[int] = []
        last_t = -999.0
        refractory_s = self.refractory_s if refractory_s is None else refractory_s
        for i in range(1, len(times)):
            if (
                residuals[i - 1] < self.acc_threshold <= residuals[i]
                and times[i] - last_t >= refractory_s
            ):
                candidates.append(i)
                last_t = float(times[i])
        return candidates

    @staticmethod
    def _estimate_gait_cycle(times: np.ndarray, gyro_signal: np.ndarray) -> float | None:
        if len(times) < 100:
            return None
        sample_rate = 1.0 / max(1e-6, float(np.median(np.diff(times))))
        smooth_samples = max(1, round(sample_rate * 0.05))
        signal = AdaptiveImuHeelStrikeDetector._smooth(gyro_signal, smooth_samples)
        signal = signal - np.mean(signal)
        low = max(1, round(0.9 * sample_rate))
        high = min(len(signal) - 2, round(2.5 * sample_rate))
        if high <= low:
            return None

        correlation = np.zeros(high + 2)
        for lag in range(low - 1, high + 2):
            first = signal[:-lag]
            second = signal[lag:]
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator > 1e-12:
                correlation[lag] = float(np.dot(first, second) / denominator)
        local_peaks = [
            lag for lag in range(low, high + 1)
            if correlation[lag] > correlation[lag - 1]
            and correlation[lag] >= correlation[lag + 1]
        ]
        if not local_peaks:
            return None
        maximum_correlation = max(correlation[lag] for lag in local_peaks)
        if maximum_correlation < 0.1:
            return None
        peak = next(
            lag for lag in local_peaks
            if correlation[lag] >= max(0.1, 0.5 * maximum_correlation)
        )
        left, center, right = correlation[peak - 1:peak + 2]
        denominator = left - 2.0 * center + right
        offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        return (peak + float(np.clip(offset, -0.5, 0.5))) / sample_rate

    def _learn_belt_threshold(self, values: np.ndarray) -> float | None:
        low = float(np.min(values))
        high = float(np.max(values))
        for _ in range(12):
            low_group = values[np.abs(values - low) <= np.abs(values - high)]
            high_group = values[np.abs(values - low) > np.abs(values - high)]
            if len(low_group) == 0 or len(high_group) == 0:
                return None
            low, high = float(np.mean(low_group)), float(np.mean(high_group))
            if low > high:
                low, high = high, low
        if high - low < 5.0:
            return None
        separation = high - low
        if self.belt_lower:
            self.belt_reset_threshold = low + 0.80 * separation
            return low + 0.65 * separation
        self.belt_reset_threshold = low + 0.20 * separation
        return low + 0.35 * separation

    def _calibrate(self) -> None:
        times = np.array([sample[0] for sample in self.samples])
        gyros = np.array([sample[1] for sample in self.samples])
        accels = np.array([sample[2] for sample in self.samples])
        belts = np.array([sample[3] for sample in self.samples])

        gravity = np.median(accels, axis=0)
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm > 1e-6:
            self.gravity = gravity / gravity_norm

        self.gyro_bias = np.median(gyros, axis=0)
        corrected_gyro = gyros - self.gyro_bias
        _, eigenvectors = np.linalg.eigh(np.cov(corrected_gyro, rowvar=False))
        self.gyro_axis = eigenvectors[:, -1]

        acc_mag = np.linalg.norm(accels, axis=1)
        baseline = float(acc_mag[0])
        residuals = np.empty_like(acc_mag)
        for i, value in enumerate(acc_mag):
            baseline += self.alpha * (float(value) - baseline)
            residuals[i] = abs(float(value) - baseline)
        self.acc_baseline = baseline
        self.acc_threshold = max(0.05, self._percentile(residuals, self.acc_percentile))

        sample_rate = len(times) / max(0.1, float(times[-1] - times[0]))
        gyro_signal = corrected_gyro @ self.gyro_axis
        gyro_signal = self._smooth(gyro_signal, round(sample_rate * 0.05))
        gyro_vectors = np.column_stack(
            [self._smooth(corrected_gyro[:, axis], round(sample_rate * 0.12)) for axis in range(3)]
        )
        self.gyro_threshold = max(0.25, self._percentile(np.abs(gyro_signal), 45.0))

        gait_cycle_s = self._estimate_gait_cycle(times, gyro_signal)
        if gait_cycle_s is not None:
            self.step_period_s = gait_cycle_s / 2.0
        self.dynamic_refractory_s = min(
            0.70,
            max(self.refractory_s, 0.60 * self.step_period_s),
        )
        candidates = self._find_candidates(times, residuals, self.dynamic_refractory_s)
        if len(candidates) < 4 or self._percentile(np.abs(gyro_signal), 90.0) < 0.5:
            print("[CAL] 보행 신호가 부족합니다. 계속 걸으면서 캘리브레이션을 진행하세요.", flush=True)
            return

        candidate_belts = belts[candidates]
        learned_belt_threshold = self._learn_belt_threshold(candidate_belts)
        if learned_belt_threshold is None:
            print("[CAL] 좌우 belt 길이 차이가 부족합니다. 계속 걸으면서 캘리브레이션을 진행하세요.", flush=True)
            return
        self.belt_threshold = learned_belt_threshold
        belt_indices = [
            i for i in candidates
            if (belts[i] <= self.belt_threshold) == self.belt_lower
        ]
        anchor = [gyro_signal[i] for i in belt_indices]
        if anchor and float(np.median(anchor)) < 0.0:
            self.gyro_axis *= -1.0
            gyro_signal *= -1.0

        other_indices = [i for i in candidates if i not in belt_indices]
        if len(belt_indices) < 2 or len(other_indices) < 2:
            print("[CAL] 좌우 gyro 학습 표본이 부족합니다. 계속 걸어주세요.", flush=True)
            return
        belt_gyro = np.median(gyro_vectors[belt_indices], axis=0)
        other_gyro = np.median(gyro_vectors[other_indices], axis=0)
        self._set_side_model(belt_gyro, other_gyro)
        self.gait_cycle_intervals.extend([2.0 * self.step_period_s] * 5)

        self.gyro_corrected = float(gyro_signal[-1])
        self.prev_residual = float(residuals[-1])
        self.ready = True
        print(
            f"[CAL] 완료 acc_thr={self.acc_threshold:.3f} "
            f"gyro_thr={self.gyro_threshold:.3f} belt_thr={self.belt_threshold:.1f}mm "
            f"side_sep={2.0 * self.gyro_side_scale:.2f} step={self.step_period_s:.2f}s "
            f"belt_side={self.belt_side}",
            flush=True,
        )

    def belt_ok(self, belt_length: float) -> bool:
        if self.belt_lower:
            return belt_length <= self.belt_threshold
        return belt_length >= self.belt_threshold

    def _update_belt_rearm(self, belt_length: float) -> None:
        if self.belt_rearmed:
            return
        if self.belt_lower:
            self.belt_rearmed = belt_length > self.belt_reset_threshold
        else:
            self.belt_rearmed = belt_length < self.belt_reset_threshold

    def consume_belt_gate(self, belt_length: float) -> bool:
        if not self.belt_rearmed or not self.belt_ok(belt_length):
            return False
        self.belt_rearmed = False
        return True

    def _set_side_model(self, belt_gyro: np.ndarray, other_gyro: np.ndarray) -> None:
        delta = belt_gyro - other_gyro
        distance = float(np.linalg.norm(delta))
        self.gyro_side_center = (belt_gyro + other_gyro) / 2.0
        if distance > 1e-6:
            self.gyro_side_axis = delta / distance
        self.gyro_side_scale = max(0.2, distance / 2.0)

    def update(
        self,
        t: float,
        gyro: tuple[float, float, float],
        accel: tuple[float, float, float],
        belt_length: float,
    ) -> tuple[bool, str, float, float, bool]:
        gyro_vector = np.array(gyro)
        accel_vector = np.array(accel)
        acc_mag = float(np.linalg.norm(accel_vector))

        if not self.ready:
            if self.calibration_start_t is None:
                self.calibration_start_t = t
            self.samples.append((t, gyro_vector, accel_vector, belt_length))
            if (
                t - self.calibration_start_t >= self.calibration_s
                and len(self.samples) >= 100
                and t - self.last_calibration_attempt_t >= 1.0
            ):
                self.last_calibration_attempt_t = t
                self._calibrate()
            if self.acc_baseline is None:
                self.acc_baseline = acc_mag
            self.acc_baseline += self.alpha * (acc_mag - self.acc_baseline)
            return False, "calibrating", 0.0, abs(acc_mag - self.acc_baseline), False

        assert self.acc_baseline is not None
        self.acc_baseline += self.alpha * (acc_mag - self.acc_baseline)
        residual = abs(acc_mag - self.acc_baseline)
        self._update_belt_rearm(belt_length)

        corrected = gyro_vector - self.gyro_bias
        projected = float(np.dot(corrected, self.gyro_axis))
        self.motion_gyro_window.append((t, projected))
        while self.motion_gyro_window and t - self.motion_gyro_window[0][0] > 0.05:
            self.motion_gyro_window.popleft()
        self.gyro_corrected = sum(value for _, value in self.motion_gyro_window) / len(self.motion_gyro_window)
        self.gyro_window.append((t, corrected))
        while self.gyro_window and t - self.gyro_window[0][0] > 0.12:
            self.gyro_window.popleft()
        mean_gyro = np.mean([value for _, value in self.gyro_window], axis=0)
        self.recent_gyro.append((t, self.gyro_corrected))
        while self.recent_gyro and t - self.recent_gyro[0][0] > 0.18:
            self.recent_gyro.popleft()

        self.acc_history.append(residual)
        self.gyro_history.append(abs(self.gyro_corrected))
        if t - self.last_adapt_t >= 1.0 and len(self.acc_history) >= 200:
            learned_acc = max(0.05, self._percentile(list(self.acc_history), self.acc_percentile))
            learned_gyro = max(0.25, self._percentile(list(self.gyro_history), 45.0))
            self.acc_threshold = 0.9 * self.acc_threshold + 0.1 * learned_acc
            self.gyro_threshold = 0.9 * self.gyro_threshold + 0.1 * learned_gyro
            self.last_adapt_t = t

        rising_cross = self.prev_residual < self.acc_threshold <= residual
        self.prev_residual = residual
        phase_gyro = max(self.recent_gyro, key=lambda item: abs(item[1]))[1]
        self.dynamic_refractory_s = min(
            0.70,
            max(self.refractory_s, 0.60 * self.step_period_s),
        )
        detected = False
        detected_t = t
        detected_side_gyro = mean_gyro
        if self.use_acc_peak:
            if (
                not self.acc_peak_candidate
                and rising_cross
            ):
                self.acc_peak_candidate = True
                self.acc_peak_candidate_start_t = t
                self.acc_peak_candidate_threshold = self.acc_threshold
                self.acc_peak_candidate_value = residual
                self.acc_peak_candidate_time_s = t
                self.acc_peak_candidate_gyro = phase_gyro
                self.acc_peak_candidate_side_gyro = mean_gyro.copy()

            if self.acc_peak_candidate:
                if residual > self.acc_peak_candidate_value:
                    self.acc_peak_candidate_value = residual
                    self.acc_peak_candidate_time_s = t
                if abs(phase_gyro) > abs(self.acc_peak_candidate_gyro):
                    self.acc_peak_candidate_gyro = phase_gyro

                peak_drop = self.acc_peak_candidate_value - residual
                enough_amplitude = (
                    self.acc_peak_candidate_value
                    >= ACC_PEAK_AMPLITUDE_RATIO * self.acc_peak_candidate_threshold
                )
                falling_from_peak = peak_drop >= max(
                    0.015,
                    ACC_PEAK_DROP_RATIO * self.acc_peak_candidate_value,
                )
                candidate_expired = (
                    t - self.acc_peak_candidate_start_t >= ACC_PEAK_TIMEOUT_S
                )
                refractory_ok = (
                    self.acc_peak_candidate_time_s - self.last_hs_t
                    >= self.dynamic_refractory_s
                )
                if (
                    (falling_from_peak and enough_amplitude and refractory_ok)
                    or candidate_expired
                ):
                    gyro_ok = (
                        not self.use_gyro_gate
                        or abs(self.acc_peak_candidate_gyro) >= self.gyro_threshold
                    )
                    if enough_amplitude and gyro_ok and refractory_ok:
                        detected = True
                        detected_t = self.acc_peak_candidate_time_s
                        self.acc_peak_value = self.acc_peak_candidate_value
                        self.acc_peak_rise_s = detected_t - self.acc_peak_candidate_start_t
                        self.acc_peak_delay_s = t - detected_t
                        self.acc_impact_time_s = detected_t
                        detected_side_gyro = self.acc_peak_candidate_side_gyro
                    self.acc_peak_candidate = False
        else:
            detected = (
                rising_cross
                and (not self.use_gyro_gate or abs(phase_gyro) >= self.gyro_threshold)
                and t - self.last_hs_t >= self.dynamic_refractory_s
            )
            if detected:
                self.acc_peak_value = residual
                self.acc_peak_rise_s = 0.0
                self.acc_peak_delay_s = 0.0
                self.acc_impact_time_s = t
        belt_ok = self.belt_ok(belt_length)
        side_score = (
            np.dot(detected_side_gyro - self.gyro_side_center, self.gyro_side_axis)
            / self.gyro_side_scale
        )
        self.gyro_side_score = float(side_score)
        other_side = "right" if self.belt_side == "left" else "left"
        side = self.belt_side if belt_ok else other_side

        if detected:
            side = self.belt_side if self.consume_belt_gate(belt_length) else other_side
            gait_cycle_s = 2.0 * self.step_period_s
            cycle_interval = detected_t - self.last_confident_side_t[side]
            skipped_cycles = max(1, round(cycle_interval / gait_cycle_s))
            normalized_cycle = cycle_interval / skipped_cycles
            if 0.65 * gait_cycle_s <= normalized_cycle <= 1.35 * gait_cycle_s:
                self.gait_cycle_intervals.append(normalized_cycle)
                self.step_period_s = float(np.median(self.gait_cycle_intervals)) / 2.0
            self.last_confident_side_t[side] = detected_t
            self.last_hs_t = detected_t
            self.last_side = side
            self.last_side_t = detected_t
            self.count += 1
            if side == "left":
                self.left_count += 1
            else:
                self.right_count += 1
        return detected, side, self.gyro_corrected, residual, belt_ok


def gait_cycle_pulse_s(
    step_period_s: float,
    reference_pulse_s: float,
    min_pulse_s: float,
    max_pulse_s: float,
) -> float:
    if step_period_s <= 0.0:
        return reference_pulse_s
    pulse_s = reference_pulse_s * (2.0 * step_period_s) / NOMINAL_GAIT_CYCLE_S
    return max(min_pulse_s, min(max_pulse_s, pulse_s))


def should_apply_torque(
    imu_hs: bool,
    side: str,
    assist_side: str,
    now_s: float,
    last_assist_s: float,
    gait_cycle_s: float,
) -> bool:
    return (
        imu_hs
        and side == assist_side
        and now_s - last_assist_s >= ASSIST_REFRACTORY_RATIO * gait_cycle_s
    )


def auto_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("연결된 시리얼 포트가 없습니다.")
    print(f"[INFO] 포트 자동 선택: {ports[0]}")
    return ports[0]


def default_csv_path() -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    return CSV_DIR / f"imu_hs_torque_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def normalize_by_threshold(
    values: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray,
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    threshold_array = np.asarray(thresholds, dtype=float)
    return np.divide(
        values_array,
        threshold_array,
        out=np.zeros_like(values_array),
        where=np.abs(threshold_array) > 1e-9,
    )


def plot_csv(
    csv_path: Path,
    png_path: Path | None = None,
    show: bool = True,
    title: str | None = None,
) -> Path:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV에 데이터가 없습니다: {csv_path}")

    text_keys = {"heel_strike", "imu_heel_strike", "imu_side"}
    keys = rows[0].keys()
    data = {
        key: [float(row[key]) for row in rows]
        for key in keys
        if key not in text_keys
    }
    if "acc_mag" not in data:
        data["acc_mag"] = [
            math.sqrt(x * x + y * y + z * z)
            for x, y, z in zip(data["accel_x"], data["accel_y"], data["accel_z"])
        ]
    if "gyro_mag" not in data:
        data["gyro_mag"] = [
            math.sqrt(x * x + y * y + z * z)
            for x, y, z in zip(data["gyro_x"], data["gyro_y"], data["gyro_z"])
        ]
    if "acc_residual" not in data:
        baseline = data["acc_mag"][0]
        residuals = []
        for value in data["acc_mag"]:
            baseline += 0.02 * (value - baseline)
            residuals.append(abs(value - baseline))
        data["acc_residual"] = residuals
    if "acc_impact_peak" not in data:
        data["acc_impact_peak"] = [0.0] * len(rows)
    if "acc_peak_rise_s" not in data:
        data["acc_peak_rise_s"] = [0.0] * len(rows)
    if "acc_peak_confirm_delay_s" not in data:
        data["acc_peak_confirm_delay_s"] = [0.0] * len(rows)
    if "acc_impact_time_s" not in data:
        data["acc_impact_time_s"] = [-1.0] * len(rows)
    if "acc_peak_candidate" not in data:
        data["acc_peak_candidate"] = [0.0] * len(rows)
    if "threshold" not in data:
        data["threshold"] = [0.28] * len(rows)
    if "gyro_corrected" not in data:
        data["gyro_corrected"] = data["gyro_x"]
    if "gyro_threshold" not in data:
        data["gyro_threshold"] = [0.0] * len(rows)
    if "gyro_side_score" not in data:
        data["gyro_side_score"] = [0.0] * len(rows)
    if "dynamic_refractory_s" not in data:
        data["dynamic_refractory_s"] = [0.45] * len(rows)
    if "step_period_s" not in data:
        data["step_period_s"] = [0.6] * len(rows)
    if "gait_cycle_s" not in data:
        data["gait_cycle_s"] = [2.0 * value for value in data["step_period_s"]]
    if "cadence_spm" not in data:
        data["cadence_spm"] = [60.0 / value if value > 0.0 else 0.0 for value in data["step_period_s"]]
    if "gait_speed_ratio" not in data:
        data["gait_speed_ratio"] = [1.0] * len(rows)
    if "assist_pulse_s" not in data:
        data["assist_pulse_s"] = [0.18] * len(rows)
    if "belt_threshold" not in data:
        data["belt_threshold"] = [0.0] * len(rows)
    if "belt_ready" not in data:
        data["belt_ready"] = [1.0] * len(rows)
    if "torque_cmd_nm" not in data:
        data["torque_cmd_nm"] = [0.0] * len(rows)
    imu_hs = [
        (
            float(row.get("acc_impact_time_s", -1.0))
            if float(row.get("acc_impact_time_s", -1.0)) >= 0.0
            else float(row["host_time_s"]),
            float(row["host_time_s"]),
            row.get("imu_side", "unknown"),
        )
        for row in rows
        if row.get("imu_heel_strike", row.get("heel_strike", "0")) == "1"
    ]
    png_path = png_path or csv_path.with_suffix(".png")
    t = data["host_time_s"]
    acc_ratio = normalize_by_threshold(data["acc_residual"], data["threshold"])
    gyro_ratio = normalize_by_threshold(
        np.abs(data["gyro_corrected"]),
        data["gyro_threshold"],
    )

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 10))
    for key in ("gyro_x", "gyro_y", "gyro_z"):
        axes[0].plot(t, data[key], label=key)
    axes[0].plot(t, data["gyro_mag"], color="black", alpha=0.35, label="gyro_mag")
    axes[0].plot(t, data["gyro_corrected"], linewidth=2, label="gyro_corrected")
    axes[0].set_ylabel("gyro (deg/s)")
    axes[0].set_ylim(-10.0, 10.0)

    for key in ("accel_x", "accel_y", "accel_z"):
        axes[1].plot(t, data[key], label=key)
    axes[1].plot(t, data["acc_mag"], color="black", alpha=0.35, label="acc_mag")
    axes[1].set_ylabel("acc")
    axes[1].set_ylim(-0.65, 2.20)

    axes[2].plot(t, acc_ratio, label="acc residual / threshold")
    axes[2].axhline(1.0, color="tab:orange", linestyle="--", label="threshold = 1")
    axes[2].set_ylabel("ACC / threshold")
    axes[2].set_ylim(0.0, 4.25)
    hs_gyro_ax = axes[2].twinx()
    hs_gyro_ax.plot(t, gyro_ratio, color="tab:green", alpha=0.75, label="abs gyro / threshold")
    hs_gyro_ax.plot(t, data["gyro_side_score"], color="tab:purple", alpha=0.55, label="gyro side score")
    hs_gyro_ax.set_ylabel("gyro ratio / side score")
    hs_gyro_ax.set_ylim(-10.0, 10.0)
    impact_times = [impact_t for impact_t, _, _ in imu_hs]
    impact_peaks = [
        float(row.get("acc_impact_peak", 0.0))
        for row in rows
        if row.get("imu_heel_strike", row.get("heel_strike", "0")) == "1"
    ]
    impact_thresholds = [
        float(row.get("threshold", 0.0))
        for row in rows
        if row.get("imu_heel_strike", row.get("heel_strike", "0")) == "1"
    ]
    impact_ratios = normalize_by_threshold(impact_peaks, impact_thresholds)
    axes[2].scatter(impact_times, impact_ratios, marker="x", color="black", label="ACC impact / threshold")

    axes[3].plot(t, data["torque_cmd_nm"], label="torque_cmd_nm")
    axes[3].set_ylabel("torque (Nm)")
    axes[3].set_ylim(-0.05, 1.60)
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        for hs_t, command_t, side in imu_hs:
            color = "tab:blue" if side == "left" else "tab:red"
            ax.axvline(hs_t, color=color, alpha=0.25, linewidth=1)
            if ax is axes[3] and command_t > hs_t:
                ax.axvline(command_t, color=color, alpha=0.35, linewidth=1, linestyle=":")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[2]:
            gyro_handles, gyro_labels = hs_gyro_ax.get_legend_handles_labels()
            handles += gyro_handles
            labels += gyro_labels
        if handles:
            if ax is axes[2]:
                legend = hs_gyro_ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    facecolor="white",
                    edgecolor="0.8",
                    framealpha=1.0,
                    labelcolor="black",
                )
            else:
                legend = ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    facecolor="white",
                    edgecolor="0.8",
                    framealpha=1.0,
                    labelcolor="black",
                )
            legend.set_zorder(100)

    fig.suptitle(title or str(csv_path))
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"[INFO] plot 저장: {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def run(args: argparse.Namespace) -> Path:
    csv_path = args.csv or default_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    detector = AdaptiveImuHeelStrikeDetector(
        calibration_s=args.calibration_s,
        acc_threshold=args.threshold,
        gyro_threshold=args.gyro_threshold,
        belt_threshold=args.min_belt_length,
        belt_side=args.belt_side,
        use_gyro_gate=not args.acc_only_hs,
        use_acc_peak=not args.threshold_cross_hs,
        acc_percentile=args.acc_percentile,
        refractory_s=args.refractory,
        alpha=args.baseline_alpha,
    )
    assist_side = args.assist_side or args.belt_side
    torque_until = -999.0
    last_assist_t = -999.0

    with csv_path.open("w", newline="") as f, PadController(
        args.port or auto_port(),
        baudrate=args.baud,
        read_timeout_s=0.01,
        ack_timeout_s=2.0,
        telemetry_stale_s=0.25,
    ) as ctrl:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        if not ctrl.start_session():
            raise RuntimeError("PAD 세션 시작 실패")
        t0 = time.monotonic()

        def send_torque(torque_nm: float) -> None:
            ctrl._link.send_padctrl_torque(torque_nm)

        print(
            f"[INFO] {args.calibration_s:.1f}s 동안 보행 캘리브레이션 후 시작: "
            f"assist_side={assist_side} torque={args.torque:.3f}Nm csv={csv_path}"
        )
        while args.duration is None or time.monotonic() - t0 < args.duration:
            data = ctrl.poll()
            if data is None:
                time.sleep(0.002)
                continue

            now = time.monotonic() - t0
            acc_mag = math.sqrt(data.accel_x**2 + data.accel_y**2 + data.accel_z**2)
            gyro_mag = math.sqrt(data.gyro_x**2 + data.gyro_y**2 + data.gyro_z**2)
            imu_hs, side, gyro_corrected, residual, belt_ok = detector.update(
                now,
                (data.gyro_x, data.gyro_y, data.gyro_z),
                (data.accel_x, data.accel_y, data.accel_z),
                data.belt_length,
            )
            gait_cycle_s = 2.0 * detector.step_period_s
            cadence_spm = 60.0 / detector.step_period_s
            gait_speed_ratio = NOMINAL_GAIT_CYCLE_S / gait_cycle_s
            assist_pulse_s = args.pulse_s
            if detector.ready:
                assist_pulse_s = gait_cycle_pulse_s(
                    detector.step_period_s,
                    args.pulse_s,
                    args.min_pulse_s,
                    args.max_pulse_s,
                )
            belt_ready = detector.belt_rearmed
            hs_for_torque = should_apply_torque(
                imu_hs,
                side,
                assist_side,
                now,
                last_assist_t,
                gait_cycle_s,
            )
            if imu_hs:
                print(
                    f"[HS ] #{detector.count:3d} {side.upper():5s} t={now:6.3f}s "
                    f"impact={detector.acc_peak_value:.3f}/{detector.acc_threshold:.3f} "
                    f"peak_delay={1000.0 * detector.acc_peak_delay_s:.0f}ms "
                    f"gyro={gyro_corrected:+.2f}/{detector.gyro_threshold:.2f} "
                    f"belt={data.belt_length:.1f}/{detector.belt_threshold:.1f}mm "
                    f"cycle={gait_cycle_s:.2f}s cadence={cadence_spm:.1f}spm "
                    f"refractory={detector.dynamic_refractory_s:.3f}s "
                    f"pulse={assist_pulse_s:.3f}s "
                    f"assist={'ON' if hs_for_torque else 'OFF'}",
                    flush=True,
                )
            if hs_for_torque:
                last_assist_t = now
                torque_until = now + assist_pulse_s

            torque = args.torque if now < torque_until else 0.0
            send_torque(torque)
            writer.writerow(
                {
                    "host_time_s": now,
                    "seq": data.seq,
                    "device_time_s": data.time_s,
                    "roll": data.roll,
                    "pitch": data.pitch,
                    "yaw": data.yaw,
                    "gyro_x": data.gyro_x,
                    "gyro_y": data.gyro_y,
                    "gyro_z": data.gyro_z,
                    "accel_x": data.accel_x,
                    "accel_y": data.accel_y,
                    "accel_z": data.accel_z,
                    "belt_length": data.belt_length,
                    "belt_velocity": data.belt_velocity,
                    "belt_acceleration": data.belt_acceleration,
                    "motor_pos": data.motor_pos,
                    "motor_vel": data.motor_vel,
                    "motor_iq_meas": data.motor_iq_meas,
                    "motor_iq_set": data.motor_iq_set,
                    "acc_mag": acc_mag,
                    "gyro_mag": gyro_mag,
                    "gyro_corrected": gyro_corrected,
                    "gyro_side_score": detector.gyro_side_score,
                    "gyro_side_separation": 2.0 * detector.gyro_side_scale,
                    "dynamic_refractory_s": detector.dynamic_refractory_s,
                    "step_period_s": detector.step_period_s,
                    "gait_cycle_s": gait_cycle_s,
                    "cadence_spm": cadence_spm,
                    "gait_speed_ratio": gait_speed_ratio,
                    "assist_pulse_s": assist_pulse_s,
                    "acc_residual": residual,
                    "acc_impact_peak": detector.acc_peak_value if imu_hs else 0.0,
                    "acc_peak_rise_s": detector.acc_peak_rise_s if imu_hs else 0.0,
                    "acc_peak_confirm_delay_s": detector.acc_peak_delay_s if imu_hs else 0.0,
                    "acc_impact_time_s": detector.acc_impact_time_s if imu_hs else -1.0,
                    "acc_peak_candidate": 1 if detector.acc_peak_candidate else 0,
                    "threshold": detector.acc_threshold,
                    "acc_threshold_percentile": detector.acc_percentile,
                    "gyro_threshold": detector.gyro_threshold,
                    "belt_threshold": detector.belt_threshold,
                    "belt_ok": 1 if belt_ok else 0,
                    "belt_ready": 1 if belt_ready else 0,
                    "calibrated": 1 if detector.ready else 0,
                    "imu_side": side,
                    "imu_heel_strike": 1 if imu_hs else 0,
                    "heel_strike": 1 if hs_for_torque else 0,
                    "hs_count": detector.count,
                    "left_hs_count": detector.left_count,
                    "right_hs_count": detector.right_count,
                    "torque_cmd_nm": torque,
                }
            )
            f.flush()

            if not args.quiet:
                print(
                    f"[TEL] belt={data.belt_length:7.1f}mm "
                    f"vel={data.belt_velocity:+7.1f}mm/s "
                    f"side={side:11s} hs={detector.count:3d} "
                    f"acc={residual:.3f}/{detector.acc_threshold:.3f} "
                    f"gyro_corr={gyro_corrected:+.2f}/{detector.gyro_threshold:.2f} "
                    f"side_score={detector.gyro_side_score:+.2f} "
                    f"cycle={gait_cycle_s:.2f}s cadence={cadence_spm:.1f}spm "
                    f"refractory={detector.dynamic_refractory_s:.3f}s "
                    f"pulse={assist_pulse_s:.3f}s "
                    f"torque={torque:.3f}Nm "
                    f"motor_pos={data.motor_pos:+.2f} "
                    f"acc=({data.accel_x:+.3f},{data.accel_y:+.3f},{data.accel_z:+.3f}) "
                    f"gyro=({data.gyro_x:+.2f},{data.gyro_y:+.2f},{data.gyro_z:+.2f})",
                    flush=True,
                )

        send_torque(0.0)
        ctrl.stop_session(timeout_s=1.0)

    print(f"[INFO] CSV 저장: {csv_path}")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="belt로 좌우 모델을 보정한 IMU가 HS를 판단하면 해당 발에 토크를 인가합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", help="시리얼 포트")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-d", "--duration", type=float, default=60.0)
    parser.add_argument("-c", "--csv", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--plot-only", type=Path, metavar="CSV")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--torque", type=float, default=1.5, help="HS 후 인가할 토크(Nm)")
    parser.add_argument("--pulse-s", type=float, default=0.18, help="gait cycle 1.2초에서의 토크 유지 시간(s)")
    parser.add_argument("--min-pulse-s", type=float, default=0.10, help="속도 적응 토크 유지 시간 하한(s)")
    parser.add_argument("--max-pulse-s", type=float, default=0.30, help="속도 적응 토크 유지 시간 상한(s)")
    parser.add_argument("--calibration-s", type=float, default=6.0, help="시작 후 걸으면서 자동 보정할 시간(s)")
    parser.add_argument("--threshold", type=float, default=0.20, help="자동 보정 실패 시 acc 임계값")
    parser.add_argument(
        "--acc-percentile",
        type=float,
        default=87.0,
        help="ACC residual 자동 임계값에 사용할 백분위수",
    )
    parser.add_argument("--gyro-threshold", type=float, default=1.0, help="자동 보정 실패 시 gyro 임계값")
    parser.add_argument("--refractory", type=float, default=0.45, help="동적 HS 중복 방지 시간의 하한(s)")
    parser.add_argument("--min-belt-length", type=float, default=-160.0, help="자동 보정 실패 시 belt 기준(mm). 부호로 허용 방향 결정")
    parser.add_argument("--belt-side", choices=("left", "right"), default="left", help="belt가 가장 늘어난 쪽의 발")
    parser.add_argument("--assist-side", choices=("left", "right"), default=None, help="토크를 줄 발. 생략하면 belt-side")
    parser.add_argument(
        "--acc-only-hs",
        action="store_true",
        help="HS 검출은 ACC 충격 피크만 사용하고 gyro는 좌우 판정에만 사용",
    )
    parser.add_argument(
        "--threshold-cross-hs",
        action="store_true",
        help="ACC 충격 피크 대신 기존 임계값 상승 통과 시점을 HS로 사용",
    )
    parser.add_argument("--baseline-alpha", type=float, default=0.02, help="acc magnitude baseline EMA 계수")
    args = parser.parse_args()

    if args.min_pulse_s <= 0.0 or args.max_pulse_s < args.min_pulse_s:
        parser.error("pulse 범위는 0 < min-pulse-s <= max-pulse-s 여야 합니다.")
    if not 50.0 <= args.acc_percentile < 100.0:
        parser.error("acc-percentile은 50 이상 100 미만이어야 합니다.")

    if args.list_ports:
        ports = list_serial_ports()
        print("\n".join(ports) if ports else "(사용 가능한 포트 없음)")
        return 0

    if args.plot_only:
        plot_csv(args.plot_only, args.png, show=not args.no_show)
        return 0

    csv_path = run(args)
    plot_csv(csv_path, args.png, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
