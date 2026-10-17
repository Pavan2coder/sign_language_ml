"""
record_custom.py
================
Interactive webcam recording tool for custom sign-language vocabulary.

Captures raw MediaPipe hand landmark sequences of shape (frames, 2, 21, 3)
and saves them as compressed NumPy (.npz) archives with structured metadata.

Features & Improvements
-----------------------
1. Colourful Human-Hand-Aligned Skeletal Visualization:
   - Full 21-landmark topology according to official MediaPipe hand anatomy.
   - Distinct, visually attractive colour palette per finger:
       * Thumb         : Vibrant Orange (BGR: 0, 140, 255)
       * Index finger  : Neon Cyan      (BGR: 255, 235, 0)
       * Middle finger : Emerald Green  (BGR: 100, 255, 0)
       * Ring finger   : Royal Purple   (BGR: 255, 50, 180)
       * Little finger : Electric Blue  (BGR: 255, 120, 30)
       * Palm and wrist: Warm Amber     (BGR: 50, 190, 255) & Crisp White
   - High-contrast dark shadow outline beneath all skeletal bones and circular joints
     for crystal-clear visibility against any clothing or lighting background.
   - Circular joint markers with white highlight cores on fingertips.
   - Large prominent dual-ring wrist marker with real-time confidence badge.
2. Low-Latency Adaptive Motion Tracking:
   - Fast, responsive 1€ (One-Euro) adaptive filter: eliminates stationary webcam
     sensor flutter while adapting cutoff dynamically during rapid gesture swipes.
   - Preserves stable identity for each hand across frames (no hand-slot swapping).
   - Spatial horizontal X disambiguation prevents MediaPipe duplicate label flips.
   - Immediate filter reset on tracking loss: eliminates stale trails or dragging lines.
3. Training Data Integrity (Raw vs. Display Separation):
   - By default, saves pure raw, unsmoothed MediaPipe landmarks to .npz files for model training.
   - Adaptive display smoothing is kept strictly isolated for the on-screen visual overlay
     unless explicitly configured with --save-smoothed.
   - Preserves exact (frames, 2, 21, 3) float32 shape (126 features/frame).
   - Never fabricates missing landmarks; absent hands strictly remain 0.0.
4. Camera Pipeline & Controls:
   - DirectShow (CAP_DSHOW) backend on Windows with buffer size 1 for zero capture lag.
   - Fallback to default backend if DirectShow is unavailable.
   - Full keyboard controls: [SPACE] to record, [S] to save, [R] to retry, [N]/[P] to navigate.
"""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Suppress noisy C++ abseil logs from MediaPipe landmark calculator
os.environ.setdefault("GLOG_minloglevel", "2")

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CUSTOM_RAW_DIR,
    CUSTOM_RECORDS_CSV,
    CUSTOM_TARGET_LABELS,
    MAX_SEQ_LEN,
    EXPECTED_LANDMARK_SHAPE,
)

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] MediaPipe is required. Install with: pip install mediapipe")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
)
logger = logging.getLogger("record_custom")


# ── Colour Palette & Anatomical Hand Topology ─────────────────────────────────

# Distinct vibrant colors per anatomical segment (BGR format for OpenCV)
FINGER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (0, 140, 255),       # Vibrant Orange
    "index": (255, 235, 0),       # Neon Cyan
    "middle": (100, 255, 0),      # Emerald Green
    "ring": (255, 50, 180),       # Royal Purple / Magenta
    "pinky": (255, 120, 30),      # Electric Blue
    "palm": (50, 190, 255),       # Warm Amber
    "wrist": (255, 255, 255),     # Crisp White
    "outline": (15, 15, 20),      # High-contrast dark shadow outline
}

# Complete anatomical bone connections for MediaPipe's 21 landmarks
# (p1, p2, color_key)
ANATOMICAL_SEGMENTS: List[Tuple[int, int, str]] = [
    # Palm arch & wrist base (Warm Amber)
    (0, 1, "palm"),   # Wrist to Thumb CMC
    (0, 5, "palm"),   # Wrist to Index MCP
    (0, 17, "palm"),  # Wrist to Pinky MCP
    (5, 9, "palm"),   # Index MCP to Middle MCP
    (9, 13, "palm"),  # Middle MCP to Ring MCP
    (13, 17, "palm"), # Ring MCP to Pinky MCP
    # Thumb: Landmarks 1-4 (Vibrant Orange)
    (1, 2, "thumb"),  # Thumb CMC to MCP
    (2, 3, "thumb"),  # Thumb MCP to IP
    (3, 4, "thumb"),  # Thumb IP to Tip
    # Index finger: Landmarks 5-8 (Neon Cyan)
    (5, 6, "index"),  # Index MCP to PIP
    (6, 7, "index"),  # Index PIP to DIP
    (7, 8, "index"),  # Index DIP to Tip
    # Middle finger: Landmarks 9-12 (Emerald Green)
    (9, 10, "middle"), # Middle MCP to PIP
    (10, 11, "middle"),# Middle PIP to DIP
    (11, 12, "middle"),# Middle DIP to Tip
    # Ring finger: Landmarks 13-16 (Royal Purple)
    (13, 14, "ring"),  # Ring MCP to PIP
    (14, 15, "ring"),  # Ring PIP to DIP
    (15, 16, "ring"),  # Ring DIP to Tip
    # Little finger (pinky): Landmarks 17-20 (Electric Blue)
    (17, 18, "pinky"), # Pinky MCP to PIP
    (18, 19, "pinky"), # Pinky PIP to DIP
    (19, 20, "pinky"), # Pinky DIP to Tip
]

# Mapping each joint index 0..20 to its designated color key
JOINT_COLOR_MAP: Dict[int, str] = {
    0: "wrist",
    1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
    5: "index", 6: "index", 7: "index", 8: "index",
    9: "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring", 14: "ring", 15: "ring", 16: "ring",
    17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky",
}


# ── Low-Latency Adaptive 1€ (One-Euro) Motion Filter ──────────────────────────

class OneEuroFilter:
    """
    1€ Filter (Casiez et al., CHI 2012) for adaptive smoothing of human hand motion.

    Adapts cutoff frequency based on movement speed:
    - At low speeds (tremor / resting), applies low cutoff (~1.0 Hz)
      to eliminate webcam sensor noise and landmark flutter.
    - At high speeds (rapid signing / finger flicks), increases cutoff dynamically
      to eliminate lag, trails, and overshoot, following the hand immediately.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        # Estimate derivative (speed)
        dx = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff: increases linearly with speed to eliminate phase lag
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: Any) -> Any:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


# ── Landmark Extraction & Hand Tracking ───────────────────────────────────────

def extract_frame_landmarks_with_confidence(
    results: Any,
    prev_landmarks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """
    Extract 21 landmarks for up to 2 hands from MediaPipe detection results,
    along with detection confidence scores for Left and Right hands.

    Preserves the two-hand coordinate structure expected by the ISL model:
        frame_landmarks[0]: Left Hand  (21 landmarks, 3 coordinates: x, y, z)
        frame_landmarks[1]: Right Hand (21 landmarks, 3 coordinates: x, y, z)

    Consistency & Disambiguation Rules:
    1. Mirrored Selfie Alignment: MediaPipe operates on the horizontally-flipped frame.
       'Left' classification maps to user's Left Hand (Slot 0, left side of screen).
       'Right' classification maps to user's Right Hand (Slot 1, right side of screen).
    2. Spatial Horizontal Disambiguation: If MediaPipe assigns duplicate labels to both
       hands (e.g., both labeled 'Right'), the candidate with smaller wrist X coordinate
       is assigned to Slot 0 (left of screen) and larger X to Slot 1 (right of screen).
       This prevents random slot swapping across frames.
    3. Temporal Continuity: When only 1 hand is detected and MediaPipe's single-hand label
       is uncertain, proximity to the previous active hand prevents single-frame slot flips.
    4. Coordinate Integrity: All coordinates are verified finite (no NaN/Inf).
    5. Structural Invariant: Exactly 21 landmarks are captured in MediaPipe anatomical order.

    Returns:
        (frame_landmarks, num_hands_detected, confidences_dict)
    """
    frame_landmarks = np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32)
    confidences: Dict[str, float] = {"left": 0.0, "right": 0.0}

    if not results or not results.multi_hand_landmarks or not results.multi_handedness:
        return frame_landmarks, 0, confidences

    # Parse and validate candidates
    candidates = []
    for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        if not handedness.classification:
            continue
        classification = handedness.classification[0]
        label = classification.label  # "Left" or "Right"
        score = float(classification.score)

        coords = np.zeros((21, 3), dtype=np.float32)
        valid = True
        for i, lm in enumerate(hand_lms.landmark):
            x = float(lm.x)
            y = float(lm.y)
            z = float(lm.z)
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                valid = False
                break
            coords[i] = [x, y, z]

        if not valid:
            continue

        wrist_x = coords[0, 0]
        wrist_y = coords[0, 1]
        mean_x = float(np.mean(coords[:, 0]))

        candidates.append({
            "coords": coords,
            "label": label,
            "score": score,
            "wrist_x": wrist_x,
            "wrist_y": wrist_y,
            "mean_x": mean_x,
            "raw_lms": hand_lms,
        })

    num_detected = len(candidates)
    if num_detected == 0:
        return frame_landmarks, 0, confidences

    # Case A: Exactly 1 hand detected
    if num_detected == 1:
        cand = candidates[0]
        target_slot = 0 if cand["label"] == "Left" else 1

        # Temporal continuity check: if previous frame had exactly one active hand
        # and current wrist is very close to that previous hand position, preserve slot
        if prev_landmarks is not None and prev_landmarks.shape == EXPECTED_LANDMARK_SHAPE:
            prev_left_active = bool(np.any(prev_landmarks[0] != 0.0))
            prev_right_active = bool(np.any(prev_landmarks[1] != 0.0))

            if prev_left_active != prev_right_active:
                prev_slot = 0 if prev_left_active else 1
                prev_wrist = prev_landmarks[prev_slot, 0, :2]
                curr_wrist = cand["coords"][0, :2]
                dist = float(np.linalg.norm(curr_wrist - prev_wrist))

                if dist < 0.18 and cand["score"] < 0.85:
                    target_slot = prev_slot

        frame_landmarks[target_slot] = cand["coords"]
        if target_slot == 0:
            confidences["left"] = cand["score"]
        else:
            confidences["right"] = cand["score"]
        return frame_landmarks, 1, confidences

    # Case B: 2 or more hands detected (take top 2 candidates)
    cand_a = candidates[0]
    cand_b = candidates[1]

    if cand_a["label"] != cand_b["label"]:
        slot_a = 0 if cand_a["label"] == "Left" else 1
        slot_b = 1 - slot_a
        frame_landmarks[slot_a] = cand_a["coords"]
        frame_landmarks[slot_b] = cand_b["coords"]
        conf_map = {slot_a: cand_a["score"], slot_b: cand_b["score"]}
        confidences["left"] = conf_map[0]
        confidences["right"] = conf_map[1]
    else:
        # DUPLICATE labels: Disambiguate spatially in mirrored selfie mode
        if cand_a["wrist_x"] <= cand_b["wrist_x"]:
            frame_landmarks[0] = cand_a["coords"]
            frame_landmarks[1] = cand_b["coords"]
            confidences["left"] = cand_a["score"]
            confidences["right"] = cand_b["score"]
        else:
            frame_landmarks[0] = cand_b["coords"]
            frame_landmarks[1] = cand_a["coords"]
            confidences["left"] = cand_b["score"]
            confidences["right"] = cand_a["score"]

    return frame_landmarks, 2, confidences


def extract_frame_landmarks(
    results: Any,
    prev_landmarks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Standard backward-compatible wrapper returning (frame_landmarks, num_detected).
    """
    raw_lms, count, _ = extract_frame_landmarks_with_confidence(results, prev_landmarks)
    return raw_lms, count


class TemporalHandTracker:
    """
    Stateful temporal manager for hand tracking continuity, dropout detection,
    and low-latency adaptive display smoothing.
    """

    def __init__(
        self,
        enable_smoothing: bool = False,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
    ):
        self.enable_smoothing = enable_smoothing
        self.filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Slot 0 (Left)
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Slot 1 (Right)
        ]
        self.last_landmarks: Optional[np.ndarray] = None
        self.consecutive_drops = 0
        self.max_consecutive_drops = 0
        self.total_frames = 0
        self.active_frames = 0
        self.left_frames = 0
        self.right_frames = 0
        self.two_hand_frames = 0
        self.tracking_health = "LOST"
        self.confidences: Dict[str, float] = {"left": 0.0, "right": 0.0}

    def reset(self) -> None:
        for f in self.filters:
            f.reset()
        self.last_landmarks = None
        self.consecutive_drops = 0
        self.max_consecutive_drops = 0
        self.total_frames = 0
        self.active_frames = 0
        self.left_frames = 0
        self.right_frames = 0
        self.two_hand_frames = 0
        self.tracking_health = "LOST"
        self.confidences = {"left": 0.0, "right": 0.0}

    def process(
        self,
        results: Any,
        timestamp: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, str, Dict[str, float]]:
        """
        Process detection results with temporal tracking and display smoothing.

        Returns:
            (raw_landmarks, display_landmarks, count, tracking_health, confidences)
            - raw_landmarks: (2, 21, 3) float32, pure unsmoothed coordinates (saved for training)
            - display_landmarks: (2, 21, 3) float32, adaptive filtered for visual overlay
            - count: number of detected hands (0, 1, or 2)
            - tracking_health: 'STABLE', 'UNSTABLE', or 'LOST'
            - confidences: {'left': float, 'right': float}
        """
        raw_lms, count, confs = extract_frame_landmarks_with_confidence(
            results, prev_landmarks=self.last_landmarks
        )
        self.total_frames += 1
        self.confidences = confs

        has_left = bool(np.any(raw_lms[0] != 0.0))
        has_right = bool(np.any(raw_lms[1] != 0.0))

        if count > 0:
            self.active_frames += 1
            self.consecutive_drops = 0
            if has_left:
                self.left_frames += 1
            if has_right:
                self.right_frames += 1
            if has_left and has_right:
                self.two_hand_frames += 1

            min_conf = min([c for c in confs.values() if c > 0.0] or [1.0])
            self.tracking_health = "STABLE" if min_conf >= 0.60 else "UNSTABLE"
        else:
            self.consecutive_drops += 1
            self.max_consecutive_drops = max(self.max_consecutive_drops, self.consecutive_drops)
            self.tracking_health = "UNSTABLE" if self.consecutive_drops <= 10 else "LOST"

        # Apply low-latency adaptive filter for display landmarks
        display_lms = raw_lms.copy()
        for slot in (0, 1):
            if np.any(raw_lms[slot] != 0.0):
                if self.enable_smoothing:
                    display_lms[slot] = self.filters[slot].filter(raw_lms[slot], timestamp)
            else:
                # On tracking loss, reset filter immediately to prevent trailing lines
                self.filters[slot].reset()
                display_lms[slot] = 0.0

        self.last_landmarks = raw_lms.copy()
        return raw_lms, display_lms, count, self.tracking_health, self.confidences

    def get_summary_stats(self) -> Dict[str, Any]:
        active_ratio = self.active_frames / max(self.total_frames, 1)
        return {
            "total_frames": self.total_frames,
            "active_frames": self.active_frames,
            "active_ratio": round(active_ratio, 3),
            "left_active_frames": self.left_frames,
            "right_active_frames": self.right_frames,
            "two_hand_frames": self.two_hand_frames,
            "max_consecutive_drops": self.max_consecutive_drops,
            "tracking_health": self.tracking_health,
            "smoothing_enabled": self.enable_smoothing,
        }


# ── Quality Validation & Storage ──────────────────────────────────────────────

def validate_recording(
    frames: List[np.ndarray],
    min_frames: int = 5,
    min_active_ratio: float = 0.25,
    max_consecutive_drops: int = 25,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate quality of recorded landmark sequence.

    Checks:
    1. Sufficient clip length (len >= min_frames).
    2. Proper 4-D landmark shape matching (frames, 2, 21, 3).
    3. Strict absence of NaN or Inf coordinate values.
    4. Sufficient hand presence throughout the gesture (active_ratio >= min_active_ratio).
    5. Absence of excessive mid-gesture tracking dropout.

    Returns:
        (is_valid, reason_str, stats_dict)
    """
    stats = {
        "frame_count": len(frames),
        "active_frames": 0,
        "active_ratio": 0.0,
        "left_active_frames": 0,
        "right_active_frames": 0,
        "two_hand_frames": 0,
        "max_consecutive_drops": 0,
        "is_finite": True,
    }

    if len(frames) < min_frames:
        return False, f"Too few frames ({len(frames)} < {min_frames})", stats

    arr = np.array(frames, dtype=np.float32)

    if arr.ndim != 4 or arr.shape[1:] != EXPECTED_LANDMARK_SHAPE:
        return False, f"Invalid shape: {arr.shape}, expected (F, {EXPECTED_LANDMARK_SHAPE})", stats

    if not np.all(np.isfinite(arr)):
        stats["is_finite"] = False
        return False, "Contains NaN or Inf coordinates", stats

    # Analyze active hands per frame
    left_present = np.any(arr[:, 0, :, :] != 0.0, axis=(1, 2))
    right_present = np.any(arr[:, 1, :, :] != 0.0, axis=(1, 2))
    any_present = left_present | right_present
    both_present = left_present & right_present

    active_frames = int(np.sum(any_present))
    active_ratio = active_frames / max(len(frames), 1)

    # Compute maximum consecutive drop frames
    max_drops = 0
    curr_drops = 0
    first_active_seen = False
    for is_active in any_present:
        if is_active:
            first_active_seen = True
            curr_drops = 0
        elif first_active_seen:
            curr_drops += 1
            if curr_drops > max_drops:
                max_drops = curr_drops

    stats["active_frames"] = active_frames
    stats["active_ratio"] = round(active_ratio, 3)
    stats["left_active_frames"] = int(np.sum(left_present))
    stats["right_active_frames"] = int(np.sum(right_present))
    stats["two_hand_frames"] = int(np.sum(both_present))
    stats["max_consecutive_drops"] = max_drops

    if active_ratio < min_active_ratio:
        return (
            False,
            f"Insufficient hand presence ({active_ratio*100:.1f}% < {min_active_ratio*100:.0f}%)",
            stats,
        )

    if max_drops > max_consecutive_drops and active_ratio >= min_active_ratio:
        return (
            False,
            f"Excessive mid-gesture tracking dropout ({max_drops} frames dropped)",
            stats,
        )

    return True, "PASSED", stats


def save_recording(
    output_dir: Path,
    label: str,
    signer_id: str,
    rep: int,
    frames: List[np.ndarray],
    fps: float,
    quality_status: str = "PASSED",
    timestamps: Optional[List[float]] = None,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save landmark sequence as compressed NumPy archive and update registry CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c for c in label if c.isalnum() or c in ("_", "-"))
    safe_signer = "".join(c for c in signer_id if c.isalnum() or c in ("_", "-"))

    filename = f"{safe_label}__{safe_signer}__rep{rep:03d}_{timestamp_str}.npz"
    file_path = output_dir / filename

    arr = np.array(frames, dtype=np.float32)

    # Compute duration and average FPS if timestamps provided
    duration = 0.0
    computed_fps = fps
    if timestamps and len(timestamps) > 1:
        duration = float(timestamps[-1] - timestamps[0])
        if duration > 0.05:
            computed_fps = float((len(timestamps) - 1) / duration)

    meta: Dict[str, Any] = {
        "label": label,
        "signer": signer_id,
        "rep": rep,
        "frame_count": len(frames),
        "fps": round(computed_fps, 2),
        "quality_status": quality_status,
        "timestamp": datetime.now().isoformat(),
        "shape": list(arr.shape),
        "duration_seconds": round(duration, 3),
    }

    if extra_stats:
        for k in ("active_ratio", "left_active_frames", "right_active_frames", "two_hand_frames", "smoothing_enabled", "saved_smoothed"):
            if k in extra_stats:
                meta[k] = extra_stats[k]

    save_kwargs: Dict[str, Any] = {
        "landmarks": arr,
        "metadata_json": json.dumps(meta),
    }
    if timestamps is not None:
        save_kwargs["timestamps"] = np.array(timestamps, dtype=np.float32)

    np.savez_compressed(file_path, **save_kwargs)

    # Append to CSV registry (maintains exact standard schema)
    csv_path = output_dir / CUSTOM_RECORDS_CSV
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["file_path", "label", "signer", "rep", "frame_count", "fps", "quality_status", "timestamp"])
        writer.writerow([
            str(file_path),
            label,
            signer_id,
            rep,
            len(frames),
            round(computed_fps, 2),
            quality_status,
            meta["timestamp"],
        ])

    print(f"\n[+] [SUCCESS] Saved recording {file_path.name} ({len(frames)} frames, quality: {quality_status})")
    return file_path


def load_custom_sample(npz_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a recorded custom sample .npz file.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        landmarks = data["landmarks"].astype(np.float32)
        meta_str = str(data["metadata_json"])
        meta = json.loads(meta_str)
        if "timestamps" in data:
            meta["timestamps"] = data["timestamps"].tolist()
    return landmarks, meta


def print_dataset_status(data_dir: Optional[Path] = None, target_reps: int = 10) -> None:
    """
    Print a summary table of all recorded custom vocabulary samples.
    """
    target_dir = Path(data_dir) if data_dir else CUSTOM_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(list(target_dir.glob("*.npz")))

    print("\n" + "=" * 70)
    print("  Custom Vocabulary Recording Status Report")
    print("=" * 70)
    print(f"  Directory: {target_dir}")
    print(f"  Total .npz Archive Files Found: {len(npz_files)}")
    print("-" * 70)

    if not npz_files:
        print("  No custom recordings found yet.")
        print(f"  Target Vocabulary ({len(CUSTOM_TARGET_LABELS)} classes):")
        for idx, lbl in enumerate(CUSTOM_TARGET_LABELS, 1):
            print(f"    {idx}. {lbl:<22} : 0 / {target_reps} reps  [MISSING]")
        print("\n  Start recording with:")
        print("    python src/record_custom.py --label Bye --signer signer01 --reps 10")
        print("=" * 70 + "\n")
        return

    # Parse metadata from files
    class_data: Dict[str, List[Dict[str, Any]]] = {}
    signer_counts: Dict[str, int] = {}

    for p in npz_files:
        try:
            with np.load(p, allow_pickle=False) as d:
                meta = json.loads(str(d["metadata_json"]))
                lbl = meta.get("label", "Unknown")
                sgn = meta.get("signer", "Unknown")
                class_data.setdefault(lbl, []).append(meta)
                signer_counts[sgn] = signer_counts.get(sgn, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to read {p.name}: {e}")

    all_labels = sorted(list(set(CUSTOM_TARGET_LABELS) | set(class_data.keys())))

    print(f"  {'Sign Label':<24} {'Count':<8} {'Target':<8} {'Signers':<18} {'Status'}")
    print("  " + "-" * 66)

    ready_count = 0
    for lbl in all_labels:
        samples = class_data.get(lbl, [])
        cnt = len(samples)
        signers = sorted(list(set(s.get("signer", "Unknown") for s in samples)))
        signers_str = ", ".join(signers) if signers else "None"
        if len(signers_str) > 16:
            signers_str = signers_str[:13] + "..."

        if cnt >= target_reps:
            status = "[COMPLETE]"
            ready_count += 1
        elif cnt >= 3:
            status = f"[READY ({cnt}/{target_reps})]"
            ready_count += 1
        elif cnt > 0:
            status = f"[LOW DATA ({cnt}/{target_reps})]"
        else:
            status = "[MISSING]"

        print(f"  {lbl:<24} {cnt:<8} {target_reps:<8} {signers_str:<18} {status}")

    print("  " + "-" * 66)
    print(f"  Signers active : {dict(signer_counts)}")
    print(f"  Classes ready  : {ready_count}/{len(all_labels)}")
    print("=" * 70 + "\n")


# ── Landmark Visualization ───────────────────────────────────────────────────

def draw_styled_hand_landmarks(
    frame: np.ndarray,
    landmarks_2x21x3: np.ndarray,
    confidences: Optional[Dict[str, float]] = None,
) -> None:
    """
    Render a colourful, human-hand-aligned skeletal overlay with distinct finger colors,
    high-contrast dark outlines, circular joint markers, large wrist markers, and
    accurate per-hand label badges with confidence.

    Finger Color Palette:
    - Thumb         : Vibrant Orange (BGR: 0, 140, 255)
    - Index finger  : Neon Cyan      (BGR: 255, 235, 0)
    - Middle finger : Emerald Green  (BGR: 100, 255, 0)
    - Ring finger   : Royal Purple   (BGR: 255, 50, 180)
    - Little finger : Electric Blue  (BGR: 255, 120, 30)
    - Palm & Wrist  : Warm Amber     (BGR: 50, 190, 255) & Crisp White
    """
    h, w, _ = frame.shape
    outline_color = FINGER_COLORS["outline"]

    for slot in (0, 1):
        hand = landmarks_2x21x3[slot]
        if not np.any(hand != 0.0):
            continue

        # Convert normalized coordinates to pixel points
        pts = []
        for i in range(21):
            px = int(np.clip(hand[i, 0] * w, 0, w - 1))
            py = int(np.clip(hand[i, 1] * h, 0, h - 1))
            pts.append((px, py))

        # 1. Bone connections: First pass outline for high contrast against background
        for p1, p2, _ in ANATOMICAL_SEGMENTS:
            cv2.line(frame, pts[p1], pts[p2], outline_color, 5, cv2.LINE_AA)

        # 2. Bone connections: Second pass vibrant anatomical finger color
        for p1, p2, col_key in ANATOMICAL_SEGMENTS:
            bone_color = FINGER_COLORS[col_key]
            cv2.line(frame, pts[p1], pts[p2], bone_color, 2, cv2.LINE_AA)

        # 3. Joint nodes: Circular markers
        for i in range(21):
            pt = pts[i]
            col_key = JOINT_COLOR_MAP[i]
            node_color = FINGER_COLORS[col_key]

            if i == 0:
                # Wrist: Large prominent marker with amber rim and white core
                cv2.circle(frame, pt, 10, outline_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, FINGER_COLORS["palm"], -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, FINGER_COLORS["wrist"], -1, cv2.LINE_AA)
            elif i in (4, 8, 12, 16, 20):
                # Fingertips: Highlighted marker with white center dot
                cv2.circle(frame, pt, 6, outline_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, node_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Intermediate joints (MCP, PIP, DIP)
                cv2.circle(frame, pt, 5, outline_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, node_color, -1, cv2.LINE_AA)

        # 4. Hand slot and confidence badge near wrist
        wrist_pt = pts[0]
        hand_name = "Left" if slot == 0 else "Right"
        accent_color = FINGER_COLORS["index"] if slot == 0 else FINGER_COLORS["thumb"]
        conf_val = confidences.get("left" if slot == 0 else "right", 0.0) if confidences else 0.0

        if conf_val > 0.0:
            badge_text = f"{hand_name[0]}-Hand [{int(conf_val*100)}%]"
        else:
            badge_text = f"{hand_name[0]}-Hand [Slot {slot}]"

        # Badge pill coordinates
        badge_x = int(np.clip(wrist_pt[0] - 40, 10, w - 120))
        badge_y = int(np.clip(wrist_pt[1] - 16, 25, h - 20))

        # Draw dark badge background with accent border
        cv2.rectangle(frame, (badge_x - 4, badge_y - 14), (badge_x + 92, badge_y + 4), (18, 18, 22), -1)
        cv2.rectangle(frame, (badge_x - 4, badge_y - 14), (badge_x + 92, badge_y + 4), accent_color, 1)
        cv2.putText(
            frame,
            badge_text,
            (badge_x, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            accent_color,
            1,
            cv2.LINE_AA,
        )


# ── Interactive Recording Session ─────────────────────────────────────────────

def run_recorder(
    label: str,
    signer_id: str,
    target_reps: int = 10,
    output_dir: Optional[Path] = None,
    camera_index: int = 0,
    max_frames: int = MAX_SEQ_LEN,
    countdown_seconds: int = 3,
    auto_save: bool = False,
    detection_confidence: float = 0.70,
    tracking_confidence: float = 0.60,
    enable_smoothing: bool = False,
    save_smoothed: bool = False,
    capture_width: int = 1280,
    capture_height: int = 720,
    target_fps: int = 30,
) -> None:
    """
    Run interactive recording session with enhanced hand tracking and quality validation.
    """
    out_dir = Path(output_dir) if output_dir else CUSTOM_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ISL Custom Vocabulary Landmark Recorder (Enhanced)")
    print("=" * 70)
    print(f"  Target Sign     : {label}")
    print(f"  Signer ID       : {signer_id}")
    print(f"  Target Reps     : {target_reps}")
    print(f"  Output Dir      : {out_dir}")
    print(f"  Max Frames/Clip : {max_frames} (approx. {max_frames/target_fps:.1f}s at {target_fps} FPS)")
    print(f"  Auto-Save Mode  : {'ENABLED' if auto_save else 'DISABLED (manual review)'}")
    print(f"  Display Smooth  : {'1€ Adaptive Filter (ON)' if enable_smoothing else 'OFF (Raw MediaPipe)'}")
    print(f"  Training Save   : {'SMOOTHED (Caution)' if save_smoothed else 'RAW UNSMOOTHED (Standard)'}")
    print(f"  Thresholds      : Detection={detection_confidence:.2f}, Tracking={tracking_confidence:.2f}")
    print("-" * 70)
    print("  KEYBOARD CONTROLS (Click on the OpenCV Camera Window first!):")
    print("    [SPACE] : START 3s countdown -> Record -> Finish sign -> SAVE & ADVANCE")
    print("    [S]     : Save current recording during review")
    print("    [R]     : Discard current recording and retry")
    print("    [N]     : Switch to next label in target vocabulary")
    print("    [P]     : Switch to previous label in target vocabulary")
    print("    [1-7]   : Quick-jump to target label [1=Bye, 2=Sorry, 3=HelpMe...]")
    print("    [Q/ESC] : Quit recorder")
    print("=" * 70)
    print("\n[!] Ensure the camera window has focus so keypresses are detected!\n")

    # Initialize Camera (Use DirectShow on Windows for fast startup and buffer control)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera {camera_index}.")
        print("  Please check that your camera is connected and not in use by another app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] Camera initialized: {actual_w}x{actual_h} @ {actual_fps_prop:.1f} FPS (Target: {target_fps} FPS)")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    tracker = TemporalHandTracker(enable_smoothing=enable_smoothing, min_cutoff=1.0, beta=5.0)

    current_label = label
    current_rep = 1
    state = "READY"  # READY, COUNTDOWN, RECORDING, REVIEW
    countdown_start_time = 0.0
    rec_start_time = 0.0

    recorded_frames: List[np.ndarray] = []
    recorded_timestamps: List[float] = []

    status_msg = f"Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to start."
    last_validation_result: Tuple[bool, str, Dict[str, Any]] = (True, "PASSED", {})
    last_tracker_stats: Dict[str, Any] = {}

    prev_time = time.perf_counter()
    fps_smooth = float(target_fps)
    consecutive_failed_reads = 0

    window_name = "ISL Custom Landmark Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    print(f"[*] [Rep {current_rep}/{target_reps}] Ready to record '{current_label}'. Press [SPACE] on camera window to start...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failed_reads += 1
                if consecutive_failed_reads > 60:
                    print("\n[ERROR] Camera stream interrupted. Exiting session cleanly.")
                    break
                time.sleep(0.01)
                continue

            consecutive_failed_reads = 0
            curr_perf = time.perf_counter()
            dt = max(curr_perf - prev_time, 1e-6)
            prev_time = curr_perf
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # Flip horizontally for mirrored selfie mode
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe hands processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_detector.process(rgb)
            rgb.flags.writeable = True

            # Extract & track landmarks (returns both raw landmarks and display landmarks)
            raw_lms, display_lms, num_hands, tracking_health, confidences = tracker.process(results, curr_perf)

            # Draw colourful skeletal connections on display frame
            draw_styled_hand_landmarks(frame, display_lms, confidences=confidences)

            has_left = bool(np.any(raw_lms[0] != 0.0))
            has_right = bool(np.any(raw_lms[1] != 0.0))

            # ── State Machine Logic ──
            if state == "READY":
                if current_rep > target_reps:
                    status_msg = f"Completed all {target_reps} reps for '{current_label}'! [N]=Next sign, [Q]=Quit."
                else:
                    status_msg = f"Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to begin 3s countdown."

            elif state == "COUNTDOWN":
                elapsed = time.perf_counter() - countdown_start_time
                remaining = countdown_seconds - elapsed
                if remaining <= 0:
                    state = "RECORDING"
                    recorded_frames = []
                    recorded_timestamps = []
                    tracker.reset()
                    rec_start_time = time.perf_counter()
                    status_msg = f"RECORDING NOW! Sign '{current_label}'... (Press SPACE or wait {max_frames} frames)"
                    print(f"\n[● RECORDING] Recording Rep {current_rep}/{target_reps} for '{current_label}'... Perform the sign!")
                else:
                    count_int = int(np.ceil(remaining))
                    status_msg = f"GET READY in {count_int} seconds..."
                    # Large centered countdown overlay
                    cv2.putText(
                        frame,
                        f"{count_int}",
                        (w // 2 - 35, h // 2 + 35),
                        cv2.FONT_HERSHEY_DUPLEX,
                        3.8,
                        (0, 230, 255),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"GET READY FOR '{current_label.upper()}'",
                        (w // 2 - 270, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            elif state == "RECORDING":
                rel_time = time.perf_counter() - rec_start_time
                # Store raw unsmoothed landmarks for training by default
                sample_to_store = display_lms if save_smoothed else raw_lms
                recorded_frames.append(sample_to_store)
                recorded_timestamps.append(rel_time)
                frame_count = len(recorded_frames)
                status_msg = f"RECORDING: {frame_count}/{max_frames} frames. Press [SPACE] when finished."

                # Animated red recording indicator
                is_blink = int(time.perf_counter() * 3) % 2 == 0
                dot_color = (0, 0, 255) if is_blink else (0, 0, 180)
                cv2.circle(frame, (45, 42), 15, dot_color, -1)
                cv2.putText(
                    frame,
                    f"REC [{frame_count}/{max_frames}]",
                    (70, 48),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.72,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Warning if tracking drops during active recording
                if tracking_health != "STABLE":
                    cv2.putText(
                        frame,
                        f"[!] TRACKING {tracking_health}: KEEP HANDS IN FRAME",
                        (w // 2 - 240, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 80, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if frame_count >= max_frames:
                    state = "REVIEW"
                    last_tracker_stats = tracker.get_summary_stats()
                    last_tracker_stats["saved_smoothed"] = save_smoothed
                    last_validation_result = validate_recording(recorded_frames)
                    is_val, reason, stats = last_validation_result
                    print(f"\n[REVIEW] Clip reached {max_frames} frames. Quality: {reason}")
                    print(f"         Active: {stats.get('active_frames')}/{len(recorded_frames)} frames ({stats.get('active_ratio',0)*100:.1f}%)")

                    if auto_save and is_val:
                        save_recording(
                            out_dir,
                            current_label,
                            signer_id,
                            current_rep,
                            recorded_frames,
                            fps_smooth,
                            reason,
                            timestamps=recorded_timestamps,
                            extra_stats=last_tracker_stats,
                        )
                        current_rep += 1
                        recorded_frames = []
                        recorded_timestamps = []
                        state = "READY"
                        print(f"[*] Auto-saved. Ready for rep {current_rep}/{target_reps}.")

            elif state == "REVIEW":
                is_val, reason, stats = last_validation_result
                if is_val:
                    status_msg = f"Review ({len(recorded_frames)} frames): Quality PASSED. Press [SPACE] or [S] to SAVE & advance."
                else:
                    status_msg = f"Review ({len(recorded_frames)} frames): WARNING ({reason}). Press [R] to RETRY or [S] to force save."

            # ── Render HUD Overlay ──
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 88), (20, 20, 25), -1)
            cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

            # Top bar information
            state_color = (0, 255, 120) if state == "READY" else ((0, 230, 255) if state == "COUNTDOWN" else ((0, 0, 255) if state == "RECORDING" else (255, 215, 0)))
            cv2.putText(
                frame,
                f"Sign: {current_label}  |  Signer: {signer_id}  |  Rep: {current_rep}/{target_reps}  |  State: {state}",
                (20, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.82,
                state_color,
                2,
                cv2.LINE_AA,
            )

            # Hand presence breakdown text
            hand_str = "None"
            if has_left and has_right:
                hand_str = "Both Hands (L+R)"
            elif has_right:
                hand_str = "Right Hand Only"
            elif has_left:
                hand_str = "Left Hand Only"

            health_color = (0, 255, 120) if tracking_health == "STABLE" else ((0, 200, 255) if tracking_health == "UNSTABLE" else (0, 70, 240))
            cv2.putText(
                frame,
                f"Hands: {num_hands}/2 [{hand_str}]  |  Tracking: {tracking_health}  |  FPS: {fps_smooth:.1f}  |  Smooth: {'ON' if enable_smoothing else 'OFF'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Bottom status helper bar
            cv2.putText(
                frame,
                status_msg,
                (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255) if state != "REVIEW" or last_validation_result[0] else (0, 160, 255),
                1,
                cv2.LINE_AA,
            )

            # Review mode card
            if state == "REVIEW":
                is_val, reason, stats = last_validation_result
                box_color = (0, 180, 60) if is_val else (0, 80, 220)
                card_x = w - 410
                cv2.rectangle(frame, (card_x, 100), (w - 20, 280), (30, 30, 35), -1)
                cv2.rectangle(frame, (card_x, 100), (w - 20, 280), box_color, 2)
                cv2.putText(frame, "RECORDING REVIEW", (card_x + 15, 130), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 215, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Frames Captured : {len(recorded_frames)}", (card_x + 15, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Hand Presence   : {stats.get('active_ratio',0)*100:.1f}%", (card_x + 15, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"L/R Breakdown   : L={stats.get('left_active_frames',0)}f, R={stats.get('right_active_frames',0)}f", (card_x + 15, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Quality Status  : {reason}", (card_x + 15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2 if not is_val else 1, cv2.LINE_AA)
                cv2.putText(frame, "[SPACE/S]=SAVE  [R]=RETRY  [N]=NEXT", (card_x + 15, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[*] Window closed by user. Quitting recorder session.")
                break

            # Key controls
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                print("\n[*] [QUIT] User requested exit. Session ended.")
                break

            elif key == ord(" "):  # SPACEBAR
                if state == "READY":
                    state = "COUNTDOWN"
                    countdown_start_time = time.perf_counter()
                    print(f"\n[*] [Rep {current_rep}/{target_reps}] Starting 3s countdown for '{current_label}'...")

                elif state == "RECORDING":
                    state = "REVIEW"
                    last_tracker_stats = tracker.get_summary_stats()
                    last_tracker_stats["saved_smoothed"] = save_smoothed
                    last_validation_result = validate_recording(recorded_frames)
                    is_val, reason, stats = last_validation_result
                    print(f"\n[REVIEW] Finished recording early ({len(recorded_frames)} frames). Quality: {reason}")
                    print(f"         Hand presence: {stats.get('active_frames')}/{len(recorded_frames)} frames ({stats.get('active_ratio',0)*100:.1f}%)")

                    if auto_save and is_val:
                        save_recording(
                            out_dir,
                            current_label,
                            signer_id,
                            current_rep,
                            recorded_frames,
                            fps_smooth,
                            reason,
                            timestamps=recorded_timestamps,
                            extra_stats=last_tracker_stats,
                        )
                        current_rep += 1
                        recorded_frames = []
                        recorded_timestamps = []
                        state = "READY"
                        print(f"[*] Auto-saved. Ready for rep {current_rep}/{target_reps}.")

                elif state == "REVIEW":
                    is_val, reason, _ = last_validation_result
                    save_recording(
                        output_dir=out_dir,
                        label=current_label,
                        signer_id=signer_id,
                        rep=current_rep,
                        frames=recorded_frames,
                        fps=fps_smooth,
                        quality_status=reason,
                        timestamps=recorded_timestamps,
                        extra_stats=last_tracker_stats,
                    )
                    current_rep += 1
                    recorded_frames = []
                    recorded_timestamps = []
                    state = "READY"
                    if current_rep <= target_reps:
                        print(f"[*] Rep {current_rep - 1} saved. Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to record.")
                    else:
                        print(f"\n[★ COMPLETED] All {target_reps} reps saved for '{current_label}'! Press [N] for next sign or [Q] to quit.")

            elif key in (ord("s"), ord("S"), 13, 10):  # S or ENTER
                if state == "REVIEW":
                    is_val, reason, _ = last_validation_result
                    save_recording(
                        output_dir=out_dir,
                        label=current_label,
                        signer_id=signer_id,
                        rep=current_rep,
                        frames=recorded_frames,
                        fps=fps_smooth,
                        quality_status=reason,
                        timestamps=recorded_timestamps,
                        extra_stats=last_tracker_stats,
                    )
                    current_rep += 1
                    recorded_frames = []
                    recorded_timestamps = []
                    state = "READY"
                    if current_rep <= target_reps:
                        print(f"[*] Rep {current_rep - 1} saved. Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to record.")
                    else:
                        print(f"\n[★ COMPLETED] All {target_reps} reps saved for '{current_label}'! Press [N] for next sign or [Q] to quit.")

            elif key in (ord("r"), ord("R")):  # Retry
                if state == "REVIEW":
                    print(f"\n[-] [DISCARDED] Discarded Rep {current_rep}. Ready to retry. Press [SPACE] to record again.")
                    state = "READY"
                    recorded_frames = []
                    recorded_timestamps = []
                    tracker.reset()

            elif key in (ord("n"), ord("N")):  # Next sign label
                if current_label in CUSTOM_TARGET_LABELS:
                    idx = (CUSTOM_TARGET_LABELS.index(current_label) + 1) % len(CUSTOM_TARGET_LABELS)
                    current_label = CUSTOM_TARGET_LABELS[idx]
                current_rep = 1
                state = "READY"
                recorded_frames = []
                recorded_timestamps = []
                tracker.reset()
                print(f"\n[*] Switched to label: '{current_label}'. Ready for Rep 1/{target_reps}. Press [SPACE] to record.")

            elif key in (ord("p"), ord("P")):  # Previous sign label
                if current_label in CUSTOM_TARGET_LABELS:
                    idx = (CUSTOM_TARGET_LABELS.index(current_label) - 1) % len(CUSTOM_TARGET_LABELS)
                    current_label = CUSTOM_TARGET_LABELS[idx]
                current_rep = 1
                state = "READY"
                recorded_frames = []
                recorded_timestamps = []
                tracker.reset()
                print(f"\n[*] Switched to label: '{current_label}'. Ready for Rep 1/{target_reps}. Press [SPACE] to record.")

            elif ord("1") <= key <= ord("7"):
                idx = key - ord("1")
                if idx < len(CUSTOM_TARGET_LABELS):
                    current_label = CUSTOM_TARGET_LABELS[idx]
                    current_rep = 1
                    state = "READY"
                    recorded_frames = []
                    recorded_timestamps = []
                    tracker.reset()
                    print(f"\n[*] Switched to label [{idx+1}]: '{current_label}'. Ready for Rep 1/{target_reps}. Press [SPACE] to record.")

    finally:
        hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[*] Camera and recorder resources closed cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(description="ISL Custom Vocabulary Landmark Recorder")
    parser.add_argument(
        "--label",
        type=str,
        default="Bye",
        help="Initial sign label to record (default: Bye)",
    )
    parser.add_argument(
        "--signer",
        type=str,
        default="signer01",
        help="Signer identifier (default: signer01)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Target number of repetitions per sign (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/custom_raw)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length per clip (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown duration in seconds (default: 3)",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Automatically save valid recordings without manual review confirmation",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.70,
        help="MediaPipe minimum detection confidence (default: 0.70)",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.60,
        help="MediaPipe minimum tracking confidence (default: 0.60)",
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Enable low-latency 1€ adaptive motion smoothing for visual overlay (default: False/raw)",
    )
    parser.add_argument(
        "--save-smoothed",
        action="store_true",
        help="Save smoothed landmarks instead of raw landmarks in .npz (default: False/raw)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target camera capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target camera capture height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera frame rate (default: 30)",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="Print list of target custom labels and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print summary table of recorded custom vocabulary samples and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_labels:
        print("\nTarget Custom Vocabulary:")
        for idx, lbl in enumerate(CUSTOM_TARGET_LABELS, 1):
            print(f"  {idx}. {lbl}")
        print()
        sys.exit(0)

    if args.status:
        print_dataset_status(data_dir=args.output_dir, target_reps=args.reps)
        sys.exit(0)

    run_recorder(
        label=args.label,
        signer_id=args.signer,
        target_reps=args.reps,
        output_dir=args.output_dir,
        camera_index=args.camera,
        max_frames=args.max_frames,
        countdown_seconds=args.countdown,
        auto_save=args.auto_save,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        enable_smoothing=args.smoothing,
        save_smoothed=args.save_smoothed,
        capture_width=args.width,
        capture_height=args.height,
        target_fps=args.fps,
    )
