#!/usr/bin/env python3
"""Calculation-only checks for adaptive IMU heel-strike detection."""

from contextlib import redirect_stdout
import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from imu_hs_torque import (
    AdaptiveImuHeelStrikeDetector,
    gait_cycle_pulse_s,
    normalize_by_threshold,
    should_apply_torque,
)


CSV_DIR = Path(__file__).resolve().parent / "csv"


def event_f1(reference: list[float], detected: list[float], tolerance_s: float = 0.25) -> float:
    available = set(range(len(detected)))
    matches = 0
    for event in reference:
        candidates = [i for i in available if abs(detected[i] - event) <= tolerance_s]
        if candidates:
            best = min(candidates, key=lambda i: abs(detected[i] - event))
            available.remove(best)
            matches += 1
    precision = matches / len(detected) if detected else 0.0
    recall = matches / len(reference) if reference else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def replay_recorded_csv(path: Path) -> tuple[float, float]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    detector = AdaptiveImuHeelStrikeDetector(
        calibration_s=6.0,
        acc_threshold=0.2,
        gyro_threshold=1.0,
        belt_threshold=-160.0,
        belt_side="left",
    )
    t0 = float(rows[0]["host_time_s"])
    ready_t = None
    detected = []
    assisted = []
    with redirect_stdout(io.StringIO()):
        for row in rows:
            t = float(row["host_time_s"]) - t0
            event, side, _, _, belt_ok = detector.update(
                t,
                tuple(float(row[key]) for key in ("gyro_x", "gyro_y", "gyro_z")),
                tuple(float(row[key]) for key in ("accel_x", "accel_y", "accel_z")),
                float(row["belt_length"]),
            )
            if detector.ready and ready_t is None:
                ready_t = t
            if event:
                detected.append(t)
                if side == "left" and belt_ok:
                    assisted.append(t)

    assert ready_t is not None
    reference = []
    gate_reference = []
    previous = int(float(rows[0]["hs_count"]))
    for row in rows[1:]:
        current = int(float(row["hs_count"]))
        t = float(row["host_time_s"]) - t0
        if current > previous and t >= ready_t:
            reference.append(t)
            if detector.belt_ok(float(row["belt_length"])):
                gate_reference.append(t)
        previous = current
    return event_f1(reference, detected), event_f1(gate_reference, assisted)


def replay_synthetic(
    rotation: np.ndarray,
    events: list[tuple[float, str]] | None = None,
    duration_s: float = 14.0,
    acc_percentile: float = 82.0,
) -> AdaptiveImuHeelStrikeDetector:
    detector = AdaptiveImuHeelStrikeDetector(
        calibration_s=6.0,
        acc_threshold=0.2,
        gyro_threshold=1.0,
        belt_threshold=-160.0,
        belt_side="left",
        acc_percentile=acc_percentile,
    )
    events = events or [(0.7 + 0.6 * i, "left" if i % 2 == 0 else "right") for i in range(22)]
    for sample in range(round(duration_s / 0.01)):
        t = sample * 0.01
        gyro_signal = 0.0
        impact = 0.0
        belt = -110.0
        for event_t, side in events:
            dt = t - event_t
            sign = 1.0 if side == "left" else -1.0
            gyro_signal += sign * 5.0 * math.exp(-(dt / 0.07) ** 2)
            impact += 0.55 * math.exp(-(dt / 0.025) ** 2)
            if side == "left":
                belt -= 70.0 * math.exp(-(dt / 0.09) ** 2)
            else:
                belt += 10.0 * math.exp(-(dt / 0.09) ** 2)
        gyro = rotation @ np.array([gyro_signal, 0.2 * math.sin(2.0 * t), 0.0])
        accel = rotation @ np.array([0.0, 0.0, 1.0 + impact])
        detector.update(t, tuple(gyro), tuple(accel), belt)
    return detector


def replay_speed_csv(path: Path, reference_cycle_s: float) -> dict[str, float | list[float]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    detector = AdaptiveImuHeelStrikeDetector(
        calibration_s=6.0,
        acc_threshold=0.2,
        gyro_threshold=1.0,
        belt_threshold=-160.0,
        belt_side="left",
        use_gyro_gate=False,
        use_acc_peak=False,
        acc_percentile=87.0,
    )
    t0 = float(rows[0]["host_time_s"])
    events: list[tuple[float, str, bool]] = []
    assists: list[float] = []
    last_assist_t = -999.0
    cycles: list[float] = []
    pulses: list[float] = []
    with redirect_stdout(io.StringIO()):
        for row in rows:
            t = float(row["host_time_s"]) - t0
            event, side, _, _, belt_ok = detector.update(
                t,
                tuple(float(row[key]) for key in ("gyro_x", "gyro_y", "gyro_z")),
                tuple(float(row[key]) for key in ("accel_x", "accel_y", "accel_z")),
                float(row["belt_length"]),
            )
            if detector.ready and t >= 8.0:
                cycles.append(2.0 * detector.step_period_s)
                pulses.append(gait_cycle_pulse_s(detector.step_period_s, 0.18, 0.10, 0.30))
            if event and t >= 8.0:
                events.append((t, side, belt_ok))
                if should_apply_torque(
                    event,
                    side,
                    "left",
                    t,
                    last_assist_t,
                    2.0 * detector.step_period_s,
                ):
                    assists.append(t)
                    last_assist_t = t

    duration = float(rows[-1]["host_time_s"]) - t0 - 8.0
    expected_events = 2.0 * duration / reference_cycle_s
    sides = [side for _, side, _ in events]
    alternation = sum(a != b for a, b in zip(sides, sides[1:])) / (len(sides) - 1)
    belt_agreement = sum((side == "left") == belt_ok for _, side, belt_ok in events) / len(events)
    return {
        "cycle_error": abs(float(np.median(cycles)) / reference_cycle_s - 1.0),
        "event_coverage": len(events) / expected_events,
        "alternation": alternation,
        "belt_agreement": belt_agreement,
        "torque_coverage": len(assists) / (duration / reference_cycle_s),
        "minimum_torque_interval": min(np.diff(assists)),
        "pulse": float(np.median(pulses)),
    }


class AdaptiveDetectorTest(unittest.TestCase):
    def test_signal_normalization_uses_threshold_as_one(self) -> None:
        normalized = normalize_by_threshold([0.1, 0.2, 0.3], [0.2, 0.2, 0.0])

        np.testing.assert_allclose(normalized, [0.5, 1.0, 0.0])

    def test_runtime_side_follows_raw_belt_state_when_gyro_disagrees(self) -> None:
        def detect_side(gyro_x: float, belt_length: float) -> str:
            detector = AdaptiveImuHeelStrikeDetector(
                calibration_s=6.0,
                acc_threshold=0.2,
                gyro_threshold=1.0,
                belt_threshold=-160.0,
                belt_side="left",
                use_gyro_gate=False,
                use_acc_peak=False,
            )
            detector.ready = True
            detector.acc_baseline = 1.0
            detector.prev_residual = 0.0
            event, side, *_ = detector.update(
                1.0,
                (gyro_x, 0.0, 0.0),
                (0.0, 0.0, 1.5),
                belt_length,
            )
            self.assertTrue(event)
            return side

        self.assertEqual(detect_side(-5.0, -170.0), "left")
        self.assertEqual(detect_side(+5.0, -100.0), "right")

    def test_torque_trigger_uses_imu_event_and_side_only(self) -> None:
        self.assertTrue(should_apply_torque(True, "left", "left", 2.0, -999.0, 1.2))
        self.assertFalse(should_apply_torque(True, "right", "left", 2.0, -999.0, 1.2))
        self.assertFalse(should_apply_torque(False, "left", "left", 2.0, -999.0, 1.2))

    def test_torque_trigger_rejects_duplicate_within_gait_cycle(self) -> None:
        self.assertFalse(should_apply_torque(True, "left", "left", 2.7, 2.0, 1.2))
        self.assertTrue(should_apply_torque(True, "left", "left", 2.8, 2.0, 1.2))

    def test_cli_default_acc_percentile_is_87(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CSV_DIR.parent / "imu_hs_torque.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("(default: 87.0)", result.stdout)

    def test_acc_percentile_changes_learned_threshold(self) -> None:
        percentile_82 = replay_synthetic(np.eye(3), acc_percentile=82.0)
        percentile_90 = replay_synthetic(np.eye(3), acc_percentile=90.0)

        self.assertGreater(percentile_90.acc_threshold, percentile_82.acc_threshold)

    def test_acc_only_mode_does_not_require_gyro_gate(self) -> None:
        common = {
            "calibration_s": 6.0,
            "acc_threshold": 0.2,
            "gyro_threshold": 1.0,
            "belt_threshold": -160.0,
            "belt_side": "left",
        }
        gyro_gate = AdaptiveImuHeelStrikeDetector(**common)
        acc_only = AdaptiveImuHeelStrikeDetector(**common, use_gyro_gate=False)
        for detector in (gyro_gate, acc_only):
            detector.ready = True
            detector.acc_baseline = 1.0
            detector.prev_residual = 0.0

        gyro_gate.update(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5), -170.0)
        acc_only.update(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5), -170.0)
        gyro_event = gyro_gate.update(1.02, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -170.0)[0]
        acc_event = acc_only.update(1.02, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -170.0)[0]

        self.assertFalse(gyro_event)
        self.assertTrue(acc_event)
        self.assertAlmostEqual(acc_only.acc_peak_delay_s, 0.02)
        self.assertAlmostEqual(acc_only.acc_peak_value, 0.49, places=2)
        self.assertAlmostEqual(acc_only.acc_impact_time_s, 1.0)

    def test_torque_pulse_follows_gait_cycle(self) -> None:
        self.assertAlmostEqual(gait_cycle_pulse_s(0.6, 0.18, 0.10, 0.30), 0.18)
        self.assertAlmostEqual(gait_cycle_pulse_s(0.4, 0.18, 0.10, 0.30), 0.12)
        self.assertAlmostEqual(gait_cycle_pulse_s(0.9, 0.18, 0.10, 0.30), 0.27)
        self.assertAlmostEqual(gait_cycle_pulse_s(0.2, 0.18, 0.10, 0.30), 0.10)
        self.assertAlmostEqual(gait_cycle_pulse_s(1.2, 0.18, 0.10, 0.30), 0.30)

    def test_recorded_walking_regression(self) -> None:
        files = (
            "imu_hs_torque_20260812_180329.csv",
            "imu_hs_torque_20260812_181643.csv",
            "imu_hs_torque_20260812_184822.csv",
        )
        scores = [replay_recorded_csv(CSV_DIR / name) for name in files]
        self.assertGreaterEqual(sum(score[0] for score in scores) / len(scores), 0.95)
        self.assertGreaterEqual(sum(score[1] for score in scores) / len(scores), 0.85)

    def test_speed_series_adapts_cycle_without_duplicate_torque(self) -> None:
        experiments = (
            ("imu_hs_torque_20260826_105902.csv", 1.9192390582),
            ("imu_hs_torque_20260826_110130.csv", 1.5714305189),
            ("imu_hs_torque_20260826_110324.csv", 1.3259419919),
            ("imu_hs_torque_20260826_110500.csv", 1.2370573297),
        )
        results = [replay_speed_csv(CSV_DIR / name, cycle) for name, cycle in experiments]

        self.assertTrue(all(result["cycle_error"] <= 0.05 for result in results))
        self.assertTrue(all(0.90 <= result["event_coverage"] <= 1.08 for result in results))
        self.assertTrue(all(result["alternation"] >= 0.90 for result in results))
        self.assertTrue(all(result["belt_agreement"] >= 0.80 for result in results))
        self.assertGreaterEqual(sum(result["torque_coverage"] for result in results) / len(results), 0.93)
        self.assertTrue(all(
            result["minimum_torque_interval"] >= 0.65 * cycle
            for result, (_, cycle) in zip(results, experiments)
        ))
        self.assertEqual(
            [result["pulse"] for result in results],
            sorted((result["pulse"] for result in results), reverse=True),
        )

    def test_no_walking_keeps_torque_detector_uncalibrated(self) -> None:
        detector = AdaptiveImuHeelStrikeDetector(
            calibration_s=6.0,
            acc_threshold=0.2,
            gyro_threshold=1.0,
            belt_threshold=-160.0,
            belt_side="left",
        )
        for sample in range(700):
            detector.update(sample * 0.01, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -120.0)
        self.assertFalse(detector.ready)

    def test_mounting_rotation_keeps_left_right_detection(self) -> None:
        angle = math.radians(73.0)
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [math.cos(angle), 0.0, -math.sin(angle)],
                [math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        normal = replay_synthetic(np.eye(3))
        rotated = replay_synthetic(rotation)

        self.assertTrue(normal.ready and rotated.ready)
        self.assertGreaterEqual(normal.left_count, 4)
        self.assertGreaterEqual(normal.right_count, 4)
        self.assertEqual((normal.left_count, normal.right_count), (rotated.left_count, rotated.right_count))
        self.assertAlmostEqual(normal.belt_threshold, rotated.belt_threshold, places=6)

    def test_cycle_tracks_speed_change_during_session(self) -> None:
        event_times = []
        t = 0.7
        while t < 8.0:
            event_times.append(t)
            t += 0.8
        while t < 22.0:
            event_times.append(t)
            t += 0.55
        events = [
            (event_t, "left" if i % 2 == 0 else "right")
            for i, event_t in enumerate(event_times)
        ]

        detector = replay_synthetic(np.eye(3), events, duration_s=23.0)

        self.assertTrue(detector.ready)
        self.assertAlmostEqual(2.0 * detector.step_period_s, 1.10, places=2)


if __name__ == "__main__":
    unittest.main()
