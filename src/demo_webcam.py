"""
demo_webcam.py
==============
Real-time webcam MediaPipe landmark capture and sign prediction demo
for the ISL500 Sign Language Recognition Model.

Features
--------
- Real-time hand landmark extraction (21 landmarks x 2 hands) using MediaPipe
- Mirror-mode selfie camera display with skeletal landmark visualization
- Preserves exact two-hand coordinate structure (Hand 0: Left, Hand 1: Right)
- Rolling sequence buffer (up to 150 frames) matching model sequence length
- Dual inference mode:
    1. Direct local SignPredictor (GPU/CPU, default, ultra-low latency)
    2. FastAPI HTTP backend via --use-api
- Modern on-screen HUD displaying predicted gesture, confidence, and Top-5 candidates
- Keyboard shortcuts:
    [Q] / [ESC] : Quit demo
    [SPACE]     : Predict immediately on current buffer
    [C]         : Clear frame buffer
    [M]         : Toggle Continuous vs Manual Record mode

Usage
-----
# Default direct inference on GPU:
python src/demo_webcam.py

# Select custom camera index:
python src/demo_webcam.py --camera 1

# Route inference through local FastAPI backend:
python src/demo_webcam.py --use-api --api-url http://127.0.0.1:8000/predict
"""

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MAX_SEQ_LEN, NUM_CLASSES
from src.predict import SignPredictor
from src.hybrid_predict import HybridSignPredictor

try:
    import mediapipe as mp
except ImportError:
    print(
        "[ERROR] MediaPipe is not installed. Please install it with:\n"
        "    pip install mediapipe\n"
    )
    sys.exit(1)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
)
logger = logging.getLogger("demo_webcam")


# ── Landmark Extraction Helper ────────────────────────────────────────────────

def extract_frame_landmarks(
    results: Any,
) -> Tuple[np.ndarray, int]:
    """
    Extract 21 landmarks for up to 2 hands from MediaPipe detection results.

    Preserves the two-hand structure expected by the trained model:
        frame_landmarks[0]: Left Hand  (21 landmarks, 3 coordinates)
        frame_landmarks[1]: Right Hand (21 landmarks, 3 coordinates)

    If a hand is absent, its entire 21x3 array remains all-zero (0.0).
    If both hands are absent, the entire 2x21x3 array is all-zero.

    Args:
        results: MediaPipe Hands detection output.

    Returns:
        (frame_landmarks, num_hands_detected)
        where frame_landmarks is np.ndarray of shape (2, 21, 3), float32.
    """
    frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    detected_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        used_slots = set()

        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"

            # In mirrored selfie view:
            # MediaPipe's "Left" corresponds to the user's Left hand -> slot 0
            # MediaPipe's "Right" corresponds to the user's Right hand -> slot 1
            slot = 0 if label == "Left" else 1

            # Prevent slot collision if both hands are detected with the same label
            if slot in used_slots:
                slot = 1 - slot

            used_slots.add(slot)
            detected_count += 1

            for i, lm in enumerate(hand_lms.landmark):
                frame_landmarks[slot, i, 0] = lm.x
                frame_landmarks[slot, i, 1] = lm.y
                frame_landmarks[slot, i, 2] = lm.z

    return frame_landmarks, detected_count


# ── Inference Dispatcher ──────────────────────────────────────────────────────

def run_prediction(
    sequence: np.ndarray,
    predictor: Optional[Any],
    use_api: bool,
    api_url: str,
    top_k: int = 5,
    recognizer_mode: str = "isl500",
) -> Optional[Dict[str, Any]]:
    """
    Dispatch landmark sequence to either local Predictor or FastAPI backend.
    """
    # Defensive check: sequence must have at least 1 non-empty frame
    has_active_hand = False
    for t in range(len(sequence)):
        if np.any(sequence[t] != 0):
            has_active_hand = True
            break

    if not has_active_hand:
        return None

    if use_api:
        if not HAS_REQUESTS:
            logger.error("The 'requests' package is required for --use-api. Install with: pip install requests")
            return None
        try:
            payload = {
                "landmarks": sequence.tolist(),
                "top_k": top_k,
                "recognizer": recognizer_mode,
            }
            resp = requests.post(api_url, json=payload, timeout=2.5)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"API returned status {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"API connection error ({api_url}): {e}")
            return None
    else:
        if predictor is None:
            return None
        try:
            if hasattr(predictor, "has_custom_model"):
                return predictor.predict(sequence, top_k=top_k, mode=recognizer_mode)
            return predictor.predict(sequence, top_k=top_k)
        except Exception as e:
            logger.warning(f"Local inference error: {e}")
            return None


# ── UI Drawing Helper ─────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    prediction: Optional[Dict[str, Any]],
    buffer_len: int,
    max_len: int,
    detected_hands: int,
    is_manual_mode: bool,
    is_recording: bool,
    fps: float,
    status_msg: str,
    recognizer_mode: str = "isl500",
) -> np.ndarray:
    """
    Render a clean, semi-transparent HUD overlay with predictions and status.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top status bar (semi-transparent dark header)
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 25), -1)

    # 2. Top-5 Prediction Card on Right Side
    card_w = 260
    card_x = w - card_w - 15
    cv2.rectangle(overlay, (card_x, 15), (w - 15, 230), (25, 25, 30), -1)

    # 3. Bottom controls bar
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)

    # Blend overlay with original frame (alpha = 0.72)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # ── Top Bar Text ──
    if prediction:
        pred_label = prediction.get("label", "Unknown")
        pred_conf = prediction.get("confidence", 0.0) * 100
        conf_color = (0, 255, 120) if pred_conf >= 75 else ((0, 220, 255) if pred_conf >= 40 else (100, 180, 255))

        cv2.putText(
            frame,
            f"Sign: {pred_label}",
            (15, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            1.15,
            conf_color,
            2,
            cv2.LINE_AA,
        )
        pred_source = prediction.get("recognizer", recognizer_mode)
        cv2.putText(
            frame,
            f"Confidence: {pred_conf:.1f}%  |  [{pred_source.upper()}]  |  Active: {prediction.get('effective_seq_len', '?')} frames",
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"ISL Sign Recognition [{recognizer_mode.upper()}]",
            (15, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_msg,
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # ── Top-5 Prediction List ──
    cv2.putText(
        frame,
        "TOP-5 PREDICTIONS",
        (card_x + 12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 215, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (card_x + 12, 45), (w - 27, 45), (70, 70, 80), 1)

    if prediction and "top_k" in prediction:
        top_k = prediction["top_k"]
        for idx, item in enumerate(top_k[:5]):
            cand_label = item.get("label", "")
            cand_conf = item.get("confidence", 0.0) * 100
            y_pos = 70 + idx * 30

            # Rank number
            cv2.putText(
                frame,
                f"{idx+1}.",
                (card_x + 12, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            # Label
            label_disp = cand_label[:14]
            cv2.putText(
                frame,
                label_disp,
                (card_x + 35, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255) if idx == 0 else (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            # Percentage
            cv2.putText(
                frame,
                f"{cand_conf:>5.1f}%",
                (w - 75, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 120) if idx == 0 else (170, 170, 170),
                1,
                cv2.LINE_AA,
            )
    else:
        for idx in range(5):
            y_pos = 70 + idx * 30
            cv2.putText(
                frame,
                f"{idx+1}.  --",
                (card_x + 12, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (120, 120, 130),
                1,
                cv2.LINE_AA,
            )

    # ── Buffer & Mode Indicators (Left corner above bottom bar) ──
    mode_text = "MODE: MANUAL" if is_manual_mode else "MODE: CONTINUOUS"
    rec_text = " [REC]" if (is_manual_mode and is_recording) else ""
    rec_color = (0, 0, 255) if (is_manual_mode and is_recording) else (200, 200, 200)

    cv2.putText(
        frame,
        f"Hands: {detected_hands}/2  |  Buffer: {buffer_len}/{max_len}  |  {mode_text}{rec_text}  |  FPS: {fps:.1f}",
        (15, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        rec_color,
        1,
        cv2.LINE_AA,
    )

    # ── Bottom Control Helper Bar ──
    controls_text = "[Q/ESC] Quit  [SPACE] Predict  [C] Clear  [M] Mode  [T] Recognizer"
    cv2.putText(
        frame,
        controls_text,
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


# ── Main Application Loop ─────────────────────────────────────────────────────

def run_webcam_demo(
    camera_index: int = 0,
    use_api: bool = False,
    api_url: str = "http://127.0.0.1:8000/predict",
    checkpoint_path: Optional[str] = None,
    max_seq_len: int = MAX_SEQ_LEN,
    predict_interval: int = 5,
    min_active_frames: int = 15,
    recognizer: str = "isl500",
) -> None:
    """
    Run the live webcam recognition loop.
    """
    print("\n" + "=" * 65)
    print("  ISL Real-Time Webcam Sign Recognition Demo")
    print("=" * 65)

    # 1. Initialize inference engine
    predictor: Optional[Any] = None
    recognizer_mode = recognizer
    if use_api:
        print(f"[*] Inference backend : FastAPI ({api_url})")
        if not HAS_REQUESTS:
            print("[ERROR] 'requests' is required for --use-api. Install with: pip install requests")
            return
        # Pre-flight check on API health
        health_url = api_url.replace("/predict", "/health")
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                print(f"[*] API connection verified: {r.json().get('status')} ({r.json().get('device_name')})")
            else:
                print(f"[!] Warning: API returned status {r.status_code}")
        except Exception as e:
            print(f"[!] Warning: Could not connect to API at {health_url}: {e}")
            print("    Make sure the FastAPI server is running:")
            print("    uvicorn src.api:app --host 127.0.0.1 --port 8000\n")
    else:
        print(f"[*] Inference backend : Local PyTorch Engine (Mode: {recognizer_mode.upper()})")
        try:
            predictor = HybridSignPredictor(isl500_ckpt=checkpoint_path)
            print(f"[*] Predictor loaded successfully on device: {predictor.device}")
        except Exception as e:
            print(f"[*] Fallback to SignPredictor: {e}")
            try:
                predictor = SignPredictor(checkpoint_path=checkpoint_path)
            except Exception as e2:
                print(f"[ERROR] Failed to initialize predictor: {e2}")
                return

    # 2. Open VideoCapture
    print(f"[*] Initializing camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            f"\n[ERROR] Could not open video device (camera index {camera_index}).\n"
            "Troubleshooting:\n"
            "  1. Verify your webcam is connected and not in use by another app.\n"
            "  2. Check Windows Camera Privacy Settings:\n"
            "     Settings -> Privacy & security -> Camera -> Allow desktop apps to access camera.\n"
            "  3. Try another camera index:  python src/demo_webcam.py --camera 1\n"
        )
        return

    # Set camera resolution (e.g. 1280x720 if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Sequence buffer (stores 2, 21, 3 frames)
    landmark_buffer: deque = deque(maxlen=max_seq_len)

    # State tracking
    is_manual_mode = False
    is_recording = False
    last_prediction: Optional[Dict[str, Any]] = None
    frame_counter = 0
    consecutive_empty_frames = 0
    status_msg = "Raise hands to begin recognition..."

    prev_time = time.time()
    fps = 30.0

    print("\n" + "-" * 65)
    print("  Webcam demo running! Press 'Q' or ESC in the window to quit.")
    print("  Controls:")
    print("    [Q] / [ESC] : Quit")
    print("    [SPACE]     : Predict on current buffer")
    print("    [C]         : Clear buffer")
    print("    [M]         : Toggle Continuous / Manual Record mode")
    print("-" * 65 + "\n")

    window_name = "ISL500 Sign Language Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab webcam frame. Retrying...")
                time.sleep(0.05)
                continue

            # Compute FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Flip horizontally for selfie mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands_detector.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Draw MediaPipe hand landmarks on frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Extract 2-hand landmarks: shape (2, 21, 3)
            frame_landmarks, num_detected = extract_frame_landmarks(results)
            frame_counter += 1

            # Track hand presence
            if num_detected > 0:
                consecutive_empty_frames = 0
                status_msg = f"{num_detected} Hand(s) Detected - Buffering sign..."
            else:
                consecutive_empty_frames += 1
                if consecutive_empty_frames > 25 and len(landmark_buffer) > 0:
                    status_msg = "No hands detected. Buffer ready."

            # Update buffer based on mode
            should_buffer = True
            if is_manual_mode and not is_recording:
                should_buffer = False

            if should_buffer:
                landmark_buffer.append(frame_landmarks)

            # Check if continuous prediction should trigger
            if not is_manual_mode:
                if (
                    len(landmark_buffer) >= min_active_frames
                    and frame_counter % predict_interval == 0
                    and consecutive_empty_frames < 20
                ):
                    buf_array = np.array(landmark_buffer, dtype=np.float32)
                    pred = run_prediction(
                        sequence=buf_array,
                        predictor=predictor,
                        use_api=use_api,
                        api_url=api_url,
                        recognizer_mode=recognizer_mode,
                    )
                    if pred is not None:
                        last_prediction = pred

            # Render HUD
            display_frame = draw_hud(
                frame=frame,
                prediction=last_prediction,
                buffer_len=len(landmark_buffer),
                max_len=max_seq_len,
                detected_hands=num_detected,
                is_manual_mode=is_manual_mode,
                is_recording=is_recording,
                fps=fps,
                status_msg=status_msg,
                recognizer_mode=recognizer_mode,
            )

            cv2.imshow(window_name, display_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                print("[*] Exit key pressed. Closing demo...")
                break

            elif key in (ord("t"), ord("T")):
                modes = ["isl500", "custom", "hybrid"]
                curr_idx = modes.index(recognizer_mode) if recognizer_mode in modes else 0
                recognizer_mode = modes[(curr_idx + 1) % len(modes)]
                status_msg = f"Switched Recognizer to: {recognizer_mode.upper()}"
                print(f"[*] Recognizer switched to: {recognizer_mode.upper()}")

            elif key in (ord("c"), ord("C")):
                landmark_buffer.clear()
                last_prediction = None
                status_msg = "Buffer cleared."
                print("[*] Buffer cleared.")

            elif key in (ord("m"), ord("M")):
                is_manual_mode = not is_manual_mode
                is_recording = False
                mode_str = "MANUAL RECORD" if is_manual_mode else "CONTINUOUS"
                status_msg = f"Switched to {mode_str} mode."
                print(f"[*] Switched to {mode_str} mode.")

            elif key == ord(" "):  # SPACEBAR
                if is_manual_mode:
                    if not is_recording:
                        # Start recording
                        landmark_buffer.clear()
                        is_recording = True
                        status_msg = "Recording sign... Press SPACE when done."
                        print("[*] Recording started...")
                    else:
                        # Stop recording and predict
                        is_recording = False
                        status_msg = "Processing gesture..."
                        print(f"[*] Recording finished ({len(landmark_buffer)} frames). Running inference...")
                        if len(landmark_buffer) >= 5:
                            buf_array = np.array(landmark_buffer, dtype=np.float32)
                            pred = run_prediction(
                                sequence=buf_array,
                                predictor=predictor,
                                use_api=use_api,
                                api_url=api_url,
                                recognizer_mode=recognizer_mode,
                            )
                            if pred:
                                last_prediction = pred
                                print(f"    -> Predicted: {pred['label']} ({pred['confidence']*100:.2f}%)")
                        else:
                            status_msg = "Too few frames recorded."
                else:
                    # In continuous mode: force trigger prediction now
                    if len(landmark_buffer) >= 5:
                        buf_array = np.array(landmark_buffer, dtype=np.float32)
                        pred = run_prediction(
                            sequence=buf_array,
                            predictor=predictor,
                            use_api=use_api,
                            api_url=api_url,
                            recognizer_mode=recognizer_mode,
                        )
                        if pred:
                            last_prediction = pred
                            print(f"[*] Manual Trigger -> {pred['label']} ({pred['confidence']*100:.2f}%)")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        hands_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Camera and resources released cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(description="ISL500 Real-Time Webcam Sign Recognition Demo")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Route predictions to FastAPI backend instead of direct PyTorch model",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="FastAPI predict endpoint URL (default: http://127.0.0.1:8000/predict)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (default: models/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        dest="max_seq_len",
        help=f"Rolling buffer maximum length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        dest="predict_interval",
        help="Inference frame interval in continuous mode (default: 5 frames)",
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default="isl500",
        choices=["isl500", "custom", "hybrid"],
        help="Recognition mode (default: isl500)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam_demo(
        camera_index=args.camera,
        use_api=args.use_api,
        api_url=args.api_url,
        checkpoint_path=args.checkpoint,
        max_seq_len=args.max_seq_len,
        predict_interval=args.predict_interval,
        recognizer=args.recognizer,
    )
