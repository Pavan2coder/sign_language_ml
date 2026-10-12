"""
record_custom.py
================
Interactive webcam recording tool for custom sign-language vocabulary.

Captures raw MediaPipe hand landmark sequences of shape (frames, 2, 21, 3)
and saves them as compressed NumPy (.npz) archives with structured metadata.

Features
--------
- Step-by-step guided recording: READY -> COUNTDOWN -> RECORDING -> REVIEW -> SAVE.
- Primary workflow key: [SPACE] to start countdown, [SPACE] to finish sign, [SPACE] to save and advance.
- Quality validation: checks sequence length, finite values, and hand presence.
- Dual visual HUD and terminal console status logging.
- Optional --auto-save mode for streamlined hands-free recording loops.

Usage
-----
# Interactive recording session:
python src/record_custom.py --label Bye --signer signer01 --reps 10

# Hands-free auto-save mode (saves valid sequences automatically):
python src/record_custom.py --label Bye --signer signer01 --reps 10 --auto-save

# List available target labels:
python src/record_custom.py --list-labels
"""

import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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


# ── Landmark Extraction ───────────────────────────────────────────────────────

def extract_frame_landmarks(results: Any) -> Tuple[np.ndarray, int]:
    """
    Extract 21 landmarks for up to 2 hands from MediaPipe detection results.

    Preserves the two-hand coordinate structure expected by the ISL model:
        frame_landmarks[0]: Left Hand  (21 landmarks, 3 coordinates)
        frame_landmarks[1]: Right Hand (21 landmarks, 3 coordinates)

    Matches the exact mirror selfie convention in demo_webcam.py.
    """
    frame_landmarks = np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32)
    detected_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        used_slots = set()

        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            slot = 0 if label == "Left" else 1

            if slot in used_slots:
                slot = 1 - slot

            used_slots.add(slot)
            detected_count += 1

            for i, lm in enumerate(hand_lms.landmark):
                frame_landmarks[slot, i, 0] = lm.x
                frame_landmarks[slot, i, 1] = lm.y
                frame_landmarks[slot, i, 2] = lm.z

    return frame_landmarks, detected_count


# ── Validation & Storage ──────────────────────────────────────────────────────

def validate_recording(
    frames: List[np.ndarray],
    min_frames: int = 5,
    min_active_ratio: float = 0.25,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate quality of recorded landmark sequence.

    Returns:
        (is_valid, reason_str, stats_dict)
    """
    stats = {
        "frame_count": len(frames),
        "active_frames": 0,
        "active_ratio": 0.0,
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

    # Count frames with at least one active hand (non-zero)
    active_frames = int(np.sum(np.any(arr != 0.0, axis=(1, 2, 3))))
    active_ratio = active_frames / max(len(frames), 1)

    stats["active_frames"] = active_frames
    stats["active_ratio"] = round(active_ratio, 3)

    if active_ratio < min_active_ratio:
        return (
            False,
            f"Insufficient hand presence ({active_ratio*100:.1f}% < {min_active_ratio*100:.0f}%)",
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
    meta = {
        "label": label,
        "signer": signer_id,
        "rep": rep,
        "frame_count": len(frames),
        "fps": round(fps, 2),
        "quality_status": quality_status,
        "timestamp": datetime.now().isoformat(),
        "shape": list(arr.shape),
    }

    # Save compressed npz
    np.savez_compressed(
        file_path,
        landmarks=arr,
        metadata_json=json.dumps(meta),
    )

    # Append to CSV registry
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
            round(fps, 2),
            quality_status,
            meta["timestamp"],
        ])

    print(f"\n[+] [SUCCESS] Saved recording {file_path.name} ({len(frames)} frames, quality: {quality_status})")
    return file_path


def load_custom_sample(npz_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a recorded custom sample .npz file.

    Returns:
        (landmarks_array, metadata_dict)
    """
    with np.load(npz_path, allow_pickle=False) as data:
        landmarks = data["landmarks"].astype(np.float32)
        meta_str = str(data["metadata_json"])
        meta = json.loads(meta_str)
    return landmarks, meta


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
) -> None:
    """
    Run interactive recording session.
    """
    out_dir = Path(output_dir) if output_dir else CUSTOM_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("  ISL Custom Vocabulary Landmark Recorder")
    print("=" * 68)
    print(f"  Target Sign     : {label}")
    print(f"  Signer ID       : {signer_id}")
    print(f"  Target Reps     : {target_reps}")
    print(f"  Output Dir      : {out_dir}")
    print(f"  Max Frames/Clip : {max_frames} (approx. 5.0 seconds at 30 FPS)")
    print(f"  Auto-Save Mode  : {'ENABLED' if auto_save else 'DISABLED (manual review)'}")
    print("-" * 68)
    print("  KEYBOARD CONTROLS (Click on the OpenCV Camera Window first!):")
    print("    [SPACE] : START 3s countdown -> Record -> Finish sign -> SAVE & ADVANCE")
    print("    [S]     : Save current recording during review")
    print("    [R]     : Discard current recording and retry")
    print("    [N]     : Switch to next label in target vocabulary")
    print("    [1-7]   : Quick-jump to target label [1=Bye, 2=Sorry, 3=HelpMe...]")
    print("    [Q/ESC] : Quit recorder")
    print("=" * 68)
    print("\n[!] IMPORTANT: Ensure the camera window has focus so keypresses are detected!\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera {camera_index}.")
        print("  Please check that your camera is connected and not open in another app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    current_label = label
    current_rep = 1
    state = "READY"  # READY, COUNTDOWN, RECORDING, REVIEW
    countdown_start_time = 0.0
    review_start_time = 0.0
    recorded_frames: List[np.ndarray] = []
    status_msg = f"Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to start."
    last_validation_result: Tuple[bool, str, Dict[str, Any]] = (True, "PASSED", {})
    prev_time = time.time()
    fps = 30.0
    consecutive_failed_reads = 0

    window_name = "ISL Custom Landmark Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"[*] [Rep {current_rep}/{target_reps}] Ready to record '{current_label}'. Press [SPACE] on camera window to start...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failed_reads += 1
                if consecutive_failed_reads > 60:
                    print("\n[ERROR] Camera stream interrupted. Exiting session cleanly.")
                    break
                time.sleep(0.02)
                continue

            consecutive_failed_reads = 0
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Flip horizontally for mirrored selfie mode
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe hands processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_detector.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            frame_lms, num_hands = extract_frame_landmarks(results)

            # ── State Machine Logic ──
            if state == "READY":
                if current_rep > target_reps:
                    status_msg = f"Completed all {target_reps} reps for '{current_label}'! [N]=Next sign, [Q]=Quit."
                else:
                    status_msg = f"Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to begin 3s countdown."

            elif state == "COUNTDOWN":
                elapsed = time.time() - countdown_start_time
                remaining = countdown_seconds - elapsed
                if remaining <= 0:
                    state = "RECORDING"
                    recorded_frames = []
                    status_msg = f"RECORDING NOW! Sign '{current_label}'... (Press SPACE or wait {max_frames} frames)"
                    print(f"\n[● RECORDING] Recording Rep {current_rep}/{target_reps} for '{current_label}'... Perform the sign!")
                else:
                    count_int = int(np.ceil(remaining))
                    status_msg = f"GET READY in {count_int} seconds..."
                    # Large centered countdown overlay
                    cv2.putText(
                        frame,
                        f"{count_int}",
                        (w // 2 - 40, h // 2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        4.0,
                        (0, 230, 255),
                        6,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"GET READY FOR '{current_label.upper()}'",
                        (w // 2 - 280, h // 2 - 80),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            elif state == "RECORDING":
                recorded_frames.append(frame_lms)
                frame_count = len(recorded_frames)
                status_msg = f"RECORDING: {frame_count}/{max_frames} frames. Press [SPACE] to finish early."

                # Animated red recording indicator
                is_blink = int(time.time() * 3) % 2 == 0
                dot_color = (0, 0, 255) if is_blink else (0, 0, 180)
                cv2.circle(frame, (45, 42), 16, dot_color, -1)
                cv2.putText(
                    frame,
                    f"REC [{frame_count}/{max_frames}]",
                    (70, 48),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                if frame_count >= max_frames:
                    # Clip duration reached
                    state = "REVIEW"
                    review_start_time = time.time()
                    last_validation_result = validate_recording(recorded_frames)
                    is_val, reason, stats = last_validation_result
                    print(f"\n[REVIEW] Clip reached {max_frames} frames. Quality: {reason}")
                    print(f"         Active hands in {stats.get('active_frames')}/{len(recorded_frames)} frames ({stats.get('active_ratio',0)*100:.1f}%)")
                    if auto_save and is_val:
                        save_recording(out_dir, current_label, signer_id, current_rep, recorded_frames, fps, reason)
                        current_rep += 1
                        recorded_frames = []
                        state = "READY"
                        print(f"[*] Auto-saved. Ready for rep {current_rep}/{target_reps}.")

            elif state == "REVIEW":
                is_val, reason, stats = last_validation_result
                if is_val:
                    status_msg = f"Review ({len(recorded_frames)} frames): Quality PASSED. Press [SPACE] or [S] to SAVE & advance."
                else:
                    status_msg = f"Review ({len(recorded_frames)} frames): WARNING ({reason}). Press [R] to RETRY or [S] to force save."

            # ── Render HUD ──
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 25), -1)
            cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

            # Top bar information
            cv2.putText(
                frame,
                f"Sign: {current_label}  |  Signer: {signer_id}  |  Rep: {current_rep}/{target_reps}  |  State: {state}",
                (20, 38),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Hands Detected: {num_hands}/2  |  FPS: {fps:.1f}  |  Auto-Save: {'ON' if auto_save else 'OFF'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

            # Bottom status & helper bar
            cv2.putText(
                frame,
                status_msg,
                (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if state != "REVIEW" or last_validation_result[0] else (0, 160, 255),
                1,
                cv2.LINE_AA,
            )

            # Review mode card
            if state == "REVIEW":
                is_val, reason, stats = last_validation_result
                box_color = (0, 180, 60) if is_val else (0, 80, 220)
                card_x = w - 380
                cv2.rectangle(frame, (card_x, 100), (w - 20, 260), (30, 30, 35), -1)
                cv2.rectangle(frame, (card_x, 100), (w - 20, 260), box_color, 2)
                cv2.putText(frame, "RECORDING REVIEW", (card_x + 15, 130), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 215, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Frames Captured : {len(recorded_frames)}", (card_x + 15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Hand Presence   : {stats.get('active_ratio',0)*100:.1f}%", (card_x + 15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Quality Status  : {reason}", (card_x + 15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2 if not is_val else 1, cv2.LINE_AA)
                cv2.putText(frame, "[SPACE/S]=SAVE  [R]=RETRY", (card_x + 15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            # Check if user clicked window close 'X'
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
                    countdown_start_time = time.time()
                    print(f"\n[*] [Rep {current_rep}/{target_reps}] Starting 3s countdown for '{current_label}'...")

                elif state == "RECORDING":
                    # User finished sign early
                    state = "REVIEW"
                    review_start_time = time.time()
                    last_validation_result = validate_recording(recorded_frames)
                    is_val, reason, stats = last_validation_result
                    print(f"\n[REVIEW] Finished recording early ({len(recorded_frames)} frames). Quality: {reason}")
                    print(f"         Hand presence: {stats.get('active_frames')}/{len(recorded_frames)} frames ({stats.get('active_ratio',0)*100:.1f}%)")
                    if auto_save and is_val:
                        save_recording(out_dir, current_label, signer_id, current_rep, recorded_frames, fps, reason)
                        current_rep += 1
                        recorded_frames = []
                        state = "READY"
                        print(f"[*] Auto-saved. Ready for rep {current_rep}/{target_reps}.")

                elif state == "REVIEW":
                    # In review, SPACE confirms and saves!
                    is_val, reason, _ = last_validation_result
                    save_recording(
                        output_dir=out_dir,
                        label=current_label,
                        signer_id=signer_id,
                        rep=current_rep,
                        frames=recorded_frames,
                        fps=fps,
                        quality_status=reason,
                    )
                    current_rep += 1
                    recorded_frames = []
                    state = "READY"
                    if current_rep <= target_reps:
                        print(f"[*] Rep {current_rep - 1} saved. Ready for Rep {current_rep}/{target_reps}. Press [SPACE] to record.")
                    else:
                        print(f"\n[★ COMPLETED] All {target_reps} reps saved for '{current_label}'! Press [N] for next sign or [Q] to quit.")

            elif key in (ord("s"), ord("S"), 13, 10):  # S or ENTER to save
                if state == "REVIEW":
                    is_val, reason, _ = last_validation_result
                    save_recording(
                        output_dir=out_dir,
                        label=current_label,
                        signer_id=signer_id,
                        rep=current_rep,
                        frames=recorded_frames,
                        fps=fps,
                        quality_status=reason,
                    )
                    current_rep += 1
                    recorded_frames = []
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

            elif key in (ord("n"), ord("N")):  # Next label
                if current_label in CUSTOM_TARGET_LABELS:
                    idx = (CUSTOM_TARGET_LABELS.index(current_label) + 1) % len(CUSTOM_TARGET_LABELS)
                    current_label = CUSTOM_TARGET_LABELS[idx]
                current_rep = 1
                state = "READY"
                recorded_frames = []
                print(f"\n[*] Switched to label: '{current_label}'. Ready for Rep 1/{target_reps}. Press [SPACE] to record.")

            elif ord("1") <= key <= ord("7"):
                idx = key - ord("1")
                if idx < len(CUSTOM_TARGET_LABELS):
                    current_label = CUSTOM_TARGET_LABELS[idx]
                    current_rep = 1
                    state = "READY"
                    recorded_frames = []
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
        "--list-labels",
        action="store_true",
        help="Print list of target custom labels and exit",
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

    run_recorder(
        label=args.label,
        signer_id=args.signer,
        target_reps=args.reps,
        output_dir=args.output_dir,
        camera_index=args.camera,
        max_frames=args.max_frames,
        countdown_seconds=args.countdown,
        auto_save=args.auto_save,
    )
