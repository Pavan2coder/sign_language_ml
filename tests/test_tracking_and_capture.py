"""
test_tracking_and_capture.py
============================
Regression test suite for enhanced hand-tracking accuracy, temporal motion capture,
coordinate consistency, and ML model protection.

Validates:
1. MediaPipe landmark extraction (21 landmarks x 2 hands).
2. One-hand (Left or Right) and two-hand gesture slot assignment.
3. Spatial horizontal disambiguation for duplicate handedness (swap prevention).
4. Temporal tracking continuity and tracking loss handling (no invented landmarks).
5. Non-finite (NaN/Inf) coordinate guards and 21-landmark anatomical ordering.
6. Monotonic frame timestamps, frame intervals, and duration tracking.
7. Optional 1€ (One-Euro) smoothing: preserves absent-hand zeros and fast motion.
8. Raw vs. Display landmark separation (raw saved for training, display filtered).
9. Colourful human-hand anatomical visualization (5 fingers, palm, wrist, outline).
10. Rigorous recording validation: flags dropouts, insufficient presence, short clips.
11. Preservation of ISL500 ML model interface (126 features, (150, 126) tensor).
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.config import (
    EXPECTED_LANDMARK_SHAPE,
    FEATURE_DIM,
    MAX_SEQ_LEN,
)
from src.record_custom import (
    extract_frame_landmarks,
    extract_frame_landmarks_with_confidence,
    OneEuroFilter,
    TemporalHandTracker,
    draw_styled_hand_landmarks,
    validate_recording,
    save_recording,
    load_custom_sample,
    FINGER_COLORS,
    ANATOMICAL_SEGMENTS,
    JOINT_COLOR_MAP,
)
from src.preprocessing import preprocess_sample
from src.hybrid_predict import HybridSignPredictor


# ── Mock Helpers for MediaPipe Data Structures ───────────────────────────────

class MockLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class MockHandLandmarks:
    def __init__(self, landmarks_list):
        self.landmark = landmarks_list


class MockClassification:
    def __init__(self, label: str, score: float = 0.95):
        self.label = label
        self.score = float(score)


class MockHandedness:
    def __init__(self, label: str, score: float = 0.95):
        self.classification = [MockClassification(label, score)]


class MockMediaPipeResults:
    def __init__(self, hand_landmarks=None, handedness=None):
        self.multi_hand_landmarks = hand_landmarks or []
        self.multi_handedness = handedness or []


def create_synthetic_hand(wrist_x: float, wrist_y: float, wrist_z: float = 0.0) -> MockHandLandmarks:
    """Generate 21 landmarks branching from a wrist coordinate."""
    lms = []
    for i in range(21):
        lms.append(MockLandmark(
            x=wrist_x + i * 0.005,
            y=wrist_y + i * 0.005,
            z=wrist_z + i * 0.002,
        ))
    return MockHandLandmarks(lms)


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestHandTrackingAndCoordinateConsistency(unittest.TestCase):
    """Test hand landmark detection, coordinate assignment, and handedness consistency."""

    def test_no_hands_detected(self):
        """When no hands are in frame, output must be strictly all zeros (2, 21, 3)."""
        results = MockMediaPipeResults([], [])
        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 0)
        self.assertEqual(lms.shape, EXPECTED_LANDMARK_SHAPE)
        self.assertEqual(lms.dtype, np.float32)
        self.assertTrue(np.all(lms == 0.0))

    def test_single_left_hand_assignment(self):
        """Single detected Left hand must occupy Slot 0; Slot 1 must remain all zeros."""
        hand = create_synthetic_hand(wrist_x=0.3, wrist_y=0.5)
        handedness = MockHandedness("Left", score=0.96)
        results = MockMediaPipeResults([hand], [handedness])

        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 1)
        self.assertEqual(lms.shape, (2, 21, 3))

        # Slot 0 (Left Hand) must be populated
        self.assertFalse(np.all(lms[0] == 0.0))
        self.assertAlmostEqual(float(lms[0, 0, 0]), 0.3, places=4)
        self.assertAlmostEqual(float(lms[0, 0, 1]), 0.5, places=4)

        # Slot 1 (Right Hand) must be strictly all zeros
        self.assertTrue(np.all(lms[1] == 0.0))

    def test_single_right_hand_assignment(self):
        """Single detected Right hand must occupy Slot 1; Slot 0 must remain all zeros."""
        hand = create_synthetic_hand(wrist_x=0.7, wrist_y=0.5)
        handedness = MockHandedness("Right", score=0.94)
        results = MockMediaPipeResults([hand], [handedness])

        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 1)
        self.assertEqual(lms.shape, (2, 21, 3))

        # Slot 0 (Left Hand) must be strictly all zeros
        self.assertTrue(np.all(lms[0] == 0.0))

        # Slot 1 (Right Hand) must be populated
        self.assertFalse(np.all(lms[1] == 0.0))
        self.assertAlmostEqual(float(lms[1, 0, 0]), 0.7, places=4)
        self.assertAlmostEqual(float(lms[1, 0, 1]), 0.5, places=4)

    def test_two_hands_distinct_labels(self):
        """Two hands with distinct labels must map directly: Left -> Slot 0, Right -> Slot 1."""
        left_hand = create_synthetic_hand(wrist_x=0.25, wrist_y=0.45)
        right_hand = create_synthetic_hand(wrist_x=0.75, wrist_y=0.45)

        # Test regardless of iteration order
        results = MockMediaPipeResults(
            [right_hand, left_hand],
            [MockHandedness("Right"), MockHandedness("Left")],
        )

        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 2)
        # Slot 0 is Left hand (wrist_x ~ 0.25)
        self.assertAlmostEqual(float(lms[0, 0, 0]), 0.25, places=4)
        # Slot 1 is Right hand (wrist_x ~ 0.75)
        self.assertAlmostEqual(float(lms[1, 0, 0]), 0.75, places=4)

    def test_two_hands_duplicate_labels_spatial_disambiguation(self):
        """
        When MediaPipe erroneously labels BOTH hands as 'Right' (or both as 'Left'),
        spatial horizontal X disambiguation in mirrored selfie mode MUST map the left-side
        hand to Slot 0 (Left) and the right-side hand to Slot 1 (Right), preventing hand-swap jitter.
        """
        hand_on_left = create_synthetic_hand(wrist_x=0.22, wrist_y=0.50)
        hand_on_right = create_synthetic_hand(wrist_x=0.78, wrist_y=0.50)

        # Both misclassified as 'Right' by MediaPipe
        results = MockMediaPipeResults(
            [hand_on_right, hand_on_left],  # Notice right hand is first in list
            [MockHandedness("Right", 0.70), MockHandedness("Right", 0.75)],
        )

        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 2)
        # Slot 0 must receive the hand on the left (x=0.22)
        self.assertAlmostEqual(float(lms[0, 0, 0]), 0.22, places=4)
        # Slot 1 must receive the hand on the right (x=0.78)
        self.assertAlmostEqual(float(lms[1, 0, 0]), 0.78, places=4)

    def test_nan_inf_coordinate_guard(self):
        """Coordinates containing NaN or Inf must be safely rejected without crashing."""
        lms_bad = [MockLandmark(np.nan, 0.5, 0.0) for _ in range(21)]
        bad_hand = MockHandLandmarks(lms_bad)
        results = MockMediaPipeResults([bad_hand], [MockHandedness("Right")])

        lms, count = extract_frame_landmarks(results)
        self.assertEqual(count, 0)
        self.assertTrue(np.all(lms == 0.0))
        self.assertTrue(np.all(np.isfinite(lms)))

    def test_landmark_anatomical_ordering(self):
        """All 21 landmarks must be captured in strict anatomical order (0 to 20)."""
        hand = create_synthetic_hand(wrist_x=0.5, wrist_y=0.5)
        results = MockMediaPipeResults([hand], [MockHandedness("Right")])
        lms, count = extract_frame_landmarks(results)

        for i in range(21):
            expected_x = 0.5 + i * 0.005
            self.assertAlmostEqual(float(lms[1, i, 0]), expected_x, places=4)


class TestTemporalMotionCaptureAndDropout(unittest.TestCase):
    """Test continuous temporal capture, tracking dropout detection, and no landmark invention."""

    def test_no_landmark_invention_on_temporary_loss(self):
        """When hand tracking drops out, landmarks must be 0.0; never hallucinated or interpolated."""
        tracker = TemporalHandTracker(enable_smoothing=False)

        # Frame 1: Hand present
        res1 = MockMediaPipeResults([create_synthetic_hand(0.6, 0.4)], [MockHandedness("Right")])
        raw1, disp1, cnt1, health1, confs1 = tracker.process(res1, timestamp=0.0)
        self.assertEqual(cnt1, 1)
        self.assertFalse(np.all(raw1[1] == 0.0))

        # Frame 2: Hand drops out
        res2 = MockMediaPipeResults([], [])
        raw2, disp2, cnt2, health2, confs2 = tracker.process(res2, timestamp=0.033)
        self.assertEqual(cnt2, 0)
        # Slot 1 must be strictly 0.0 — NO hallucinated coordinates
        self.assertTrue(np.all(raw2 == 0.0))
        self.assertTrue(np.all(disp2 == 0.0))
        self.assertEqual(tracker.consecutive_drops, 1)

    def test_tracking_health_transitions(self):
        """Tracking health transitions from STABLE to UNSTABLE and LOST upon dropout."""
        tracker = TemporalHandTracker(enable_smoothing=False)
        res_present = MockMediaPipeResults([create_synthetic_hand(0.6, 0.4)], [MockHandedness("Right", 0.95)])
        res_empty = MockMediaPipeResults([], [])

        tracker.process(res_present, 0.0)
        self.assertEqual(tracker.tracking_health, "STABLE")

        # Few dropped frames -> UNSTABLE
        for i in range(5):
            tracker.process(res_empty, 0.033 * (i + 1))
        self.assertEqual(tracker.tracking_health, "UNSTABLE")

        # Extended dropped frames (> 10) -> LOST
        for i in range(12):
            tracker.process(res_empty, 0.033 * (i + 6))
        self.assertEqual(tracker.tracking_health, "LOST")

    def test_one_euro_filter_preserves_absent_hands_zero(self):
        """When 1€ smoothing is enabled, absent hands must strictly stay all-zero."""
        tracker = TemporalHandTracker(enable_smoothing=True)
        res = MockMediaPipeResults([create_synthetic_hand(0.65, 0.45)], [MockHandedness("Right")])

        # Right hand active, Left hand absent
        raw, disp, cnt, _, _ = tracker.process(res, timestamp=0.0)
        self.assertEqual(cnt, 1)
        # Left hand (Slot 0) must remain strictly 0.0 in both raw and display
        self.assertTrue(np.all(raw[0] == 0.0))
        self.assertTrue(np.all(disp[0] == 0.0))
        # Right hand (Slot 1) must be valid float values
        self.assertFalse(np.all(raw[1] == 0.0))
        self.assertFalse(np.all(disp[1] == 0.0))
        self.assertTrue(np.all(np.isfinite(raw)))
        self.assertTrue(np.all(np.isfinite(disp)))

    def test_one_euro_filter_responsiveness_on_fast_motion(self):
        """Fast hand motion must not be heavily dampened or distorted by 1€ filter."""
        filt = OneEuroFilter(min_cutoff=1.0, beta=5.0)
        # Simulate fast leap: x jumps from 0.2 to 0.7 in 33ms (speed = 15 unit/s)
        p0 = np.array([0.2, 0.5, 0.0], dtype=np.float32)
        p1 = np.array([0.7, 0.5, 0.0], dtype=np.float32)

        out0 = filt.filter(p0, t=0.0)
        out1 = filt.filter(p1, t=0.033)

        # Adaptive cutoff responds rapidly to velocity: output must track jump
        self.assertGreater(out1[0], 0.55)
        # Next frame continues tracking target closely
        out2 = filt.filter(p1, t=0.066)
        self.assertGreater(out2[0], 0.65)

    def test_filter_reset_on_tracking_loss(self):
        """When a hand disappears and reappears at a distant coordinate, filter resets cleanly."""
        tracker = TemporalHandTracker(enable_smoothing=True, min_cutoff=1.0, beta=5.0)

        # Hand active at position A (x=0.2)
        res_a = MockMediaPipeResults([create_synthetic_hand(0.2, 0.5)], [MockHandedness("Right")])
        tracker.process(res_a, timestamp=0.0)

        # Hand drops out for 5 frames
        res_empty = MockMediaPipeResults([], [])
        for i in range(1, 6):
            tracker.process(res_empty, timestamp=i * 0.033)

        # Hand reappears at distant position B (x=0.85)
        res_b = MockMediaPipeResults([create_synthetic_hand(0.85, 0.5)], [MockHandedness("Right")])
        raw_b, disp_b, cnt_b, _, _ = tracker.process(res_b, timestamp=0.200)

        # Filter was reset: display must immediately snap to position B (0.85), NOT interpolate from 0.2
        self.assertAlmostEqual(float(disp_b[1, 0, 0]), 0.85, places=3)


class TestColourfulAnatomicalVisualization(unittest.TestCase):
    """Test colour palette mapping, anatomical topology, and overlay rendering."""

    def test_finger_colors_definition(self):
        """Ensure all required finger and palm colors exist as valid 3-tuple BGR colors."""
        required_keys = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist", "outline"]
        for k in required_keys:
            self.assertIn(k, FINGER_COLORS)
            color = FINGER_COLORS[k]
            self.assertEqual(len(color), 3)
            for ch in color:
                self.assertIsInstance(ch, int)
                self.assertTrue(0 <= ch <= 255)

    def test_anatomical_connections_count(self):
        """Ensure all 21 MediaPipe hand connections are present and valid."""
        self.assertEqual(len(ANATOMICAL_SEGMENTS), 21)
        for p1, p2, col_key in ANATOMICAL_SEGMENTS:
            self.assertTrue(0 <= p1 <= 20)
            self.assertTrue(0 <= p2 <= 20)
            self.assertIn(col_key, FINGER_COLORS)

    def test_all_landmarks_have_color_mapping(self):
        """Ensure all 21 landmarks (0 to 20) have designated joint colors."""
        for i in range(21):
            self.assertIn(i, JOINT_COLOR_MAP)
            self.assertIn(JOINT_COLOR_MAP[i], FINGER_COLORS)

    def test_draw_styled_hand_landmarks_execution(self):
        """Draw styled landmarks on synthetic image to verify rendering without error."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        landmarks = np.zeros((2, 21, 3), dtype=np.float32)

        # Populate Right hand with synthetic coordinates
        for i in range(21):
            landmarks[1, i] = [0.5 + i * 0.01, 0.5 + i * 0.01, 0.0]

        draw_styled_hand_landmarks(frame, landmarks, confidences={"left": 0.0, "right": 0.95})

        # Frame should now contain non-zero drawn pixels
        self.assertTrue(np.any(frame != 0))


class TestValidationAndStorageIntegrity(unittest.TestCase):
    """Test recording validation safeguards, timestamp serialization, and metadata."""

    def test_excessive_dropout_rejection(self):
        """A recording with excessive consecutive dropped frames mid-gesture must be flagged."""
        frames = []
        for i in range(40):
            f = np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32)
            # Active in frames 0..9 and 36..39; dropped in frames 10..35 (26 consecutive drops)
            if i < 10 or i >= 36:
                f[1, :, :] = 0.5
            frames.append(f)

        is_val, reason, stats = validate_recording(frames, min_frames=5, min_active_ratio=0.25, max_consecutive_drops=20)
        self.assertFalse(is_val)
        self.assertIn("Excessive mid-gesture tracking dropout", reason)
        self.assertGreater(stats["max_consecutive_drops"], 20)

    def test_save_and_load_with_timestamps(self):
        """Ensure per-frame timestamps, duration, and FPS are accurately saved in .npz."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            frames = [np.ones(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) * 0.4 for _ in range(30)]
            timestamps = [i * 0.0333 for i in range(30)]  # 30 FPS over ~1.0s

            saved_path = save_recording(
                output_dir=tmp_dir,
                label="HelpMe",
                signer_id="signer01",
                rep=1,
                frames=frames,
                fps=30.0,
                quality_status="PASSED",
                timestamps=timestamps,
            )
            self.assertTrue(saved_path.exists())

            loaded_lms, meta = load_custom_sample(saved_path)
            self.assertEqual(loaded_lms.shape, (30, 2, 21, 3))
            self.assertEqual(meta["frame_count"], 30)
            self.assertIn("timestamps", meta)
            self.assertEqual(len(meta["timestamps"]), 30)
            self.assertIn("duration_seconds", meta)
            self.assertAlmostEqual(meta["duration_seconds"], 29 * 0.0333, places=2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMLModelProtection(unittest.TestCase):
    """Verify that recordings from the enhanced recorder are 100% compatible with ISL500 model."""

    def test_preprocessing_and_feature_dim_preservation(self):
        """Recorded sample must produce exactly (150, 126) features when preprocessed."""
        raw = np.zeros((35, 2, 21, 3), dtype=np.float32)
        for t in range(35):
            raw[t, 1, 0, :] = [0.5, 0.5, 0.0]  # Right wrist
            for j in range(1, 21):
                raw[t, 1, j, :] = [0.5 + j * 0.01, 0.5 + j * 0.01, 0.0]

        features = preprocess_sample(raw, max_seq_len=MAX_SEQ_LEN)
        self.assertEqual(features.shape, (150, 126))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_hybrid_predictor_with_recorded_sample(self):
        """Recorded sequence can be passed directly to HybridSignPredictor without error."""
        predictor = HybridSignPredictor()
        raw = np.zeros((25, 2, 21, 3), dtype=np.float32)
        raw[:, 1, :, :] = 0.45

        res = predictor.predict(raw, top_k=3, mode="isl500")
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertEqual(len(res["top_k"]), 3)
        self.assertEqual(res["recognizer"], "isl500")


if __name__ == "__main__":
    unittest.main()
