"""
predict.py
==========
Standalone inference module for the ISL500 Sign Language Recognition Model.

Predicts sign labels from raw MediaPipe hand landmark sequences
of shape (frames, 2, 21, 3).

Usage
-----
# Command-line test on one test sample:
python src/predict.py

# Specify sample index:
python src/predict.py --sample-index 5

# Specify explicit .h5 file path:
python src/predict.py --path "D:/ISL-DATA/Landmarks/MediaPipe/ISL_DATA_USER001/Absent__session117__clip014.h5"

# Python API:
from src.predict import SignPredictor
predictor = SignPredictor()
result = predictor.predict(raw_landmarks)  # numpy array (F, 2, 21, 3)
print(result["label"], result["confidence"])
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR, BEST_CKPT_NAME,
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    EXPECTED_LANDMARK_SHAPE,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
)
from src.data_loader        import load_h5_file, load_mapping, get_label_from_path
from src.preprocessing      import preprocess_sample, pad_or_truncate
from src.feature_extraction import compute_seq_lengths
from src.train              import GRUClassifier, get_device


class SignPredictor:
    """
    Offline inference engine for the ISL500 sign language classifier.

    Loads the trained GRUClassifier checkpoint and provides methods
    to classify raw MediaPipe hand landmark sequences.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if checkpoint_path is None:
            self.checkpoint_path = CHECKPOINT_DIR / BEST_CKPT_NAME
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}\n"
                "Please train or provide a valid model checkpoint."
            )

        self.device = device if device is not None else get_device()
        self._load_model()

    def _load_model(self) -> None:
        """Load checkpoint and construct the exact GRUClassifier architecture."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.label_to_id: Dict[str, int] = ckpt["label_to_id"]
        self.id_to_label: Dict[int, str] = {int(v): k for k, v in self.label_to_id.items()}
        self.num_classes: int = len(self.label_to_id)
        cfg: Dict[str, Any] = ckpt.get("config", {})

        hidden_size = cfg.get("gru_hidden", GRU_HIDDEN_SIZE)
        num_layers  = cfg.get("gru_layers", GRU_NUM_LAYERS)
        fc_hidden   = cfg.get("fc_hidden", FC_HIDDEN_SIZE)
        dropout     = cfg.get("gru_dropout", GRU_DROPOUT)
        input_size  = cfg.get("feature_dim", FEATURE_DIM)

        self.model = GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=dropout,
            fc_hidden=fc_hidden,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def validate_and_preprocess(self, raw_landmarks: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Validate input format and apply the training preprocessing pipeline.

        Args:
            raw_landmarks : raw MediaPipe array of shape (frames, 2, 21, 3)
                            or pre-flattened features (frames, 126).

        Returns:
            np.ndarray of shape (MAX_SEQ_LEN, FEATURE_DIM) = (150, 126), float32.

        Raises:
            ValueError : on invalid dimensions, empty frames, or non-finite values.
        """
        if isinstance(raw_landmarks, torch.Tensor):
            landmarks = raw_landmarks.detach().cpu().numpy()
        elif isinstance(raw_landmarks, np.ndarray):
            landmarks = raw_landmarks
        else:
            landmarks = np.asarray(raw_landmarks, dtype=np.float32)

        if not np.all(np.isfinite(landmarks)):
            raise ValueError("Input landmark array contains NaN or Inf values.")

        if landmarks.ndim == 4:
            if landmarks.shape[0] == 0:
                raise ValueError("Input landmark sequence contains 0 frames.")
            if landmarks.shape[1:] != EXPECTED_LANDMARK_SHAPE:
                raise ValueError(
                    f"Invalid per-frame landmark shape {landmarks.shape[1:]}. "
                    f"Expected {EXPECTED_LANDMARK_SHAPE} (2 hands, 21 landmarks, 3 coordinates)."
                )
            return preprocess_sample(landmarks, max_seq_len=MAX_SEQ_LEN)

        elif landmarks.ndim == 2:
            if landmarks.shape[0] == 0:
                raise ValueError("Input sequence contains 0 frames.")
            if landmarks.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"Invalid feature dimension {landmarks.shape[1]}. "
                    f"Expected {FEATURE_DIM} (2 hands * 21 landmarks * 3 coordinates = 126)."
                )
            return pad_or_truncate(landmarks, max_seq_len=MAX_SEQ_LEN)

        else:
            raise ValueError(
                f"Invalid landmark array shape {landmarks.shape}. "
                f"Expected 4D array (frames, 2, 21, 3) or 2D array (frames, 126)."
            )

    def predict(
        self,
        raw_landmarks: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Classify a single MediaPipe landmark sequence.

        Args:
            raw_landmarks : np.ndarray or torch.Tensor of shape (frames, 2, 21, 3) or (frames, 126).
            top_k         : number of top predictions to return (default: 5).

        Returns:
            dict with:
                label             : str, top-1 predicted label name
                confidence        : float, top-1 probability in [0.0, 1.0]
                class_id          : int, top-1 class ID
                top_k             : list of dicts with label, confidence, class_id
                raw_frame_count   : int, number of raw input frames
                effective_seq_len : int, active non-padded sequence length
        """
        raw_frame_count = len(raw_landmarks) if hasattr(raw_landmarks, "__len__") else 0
        processed = self.validate_and_preprocess(raw_landmarks)

        tensor_x = torch.from_numpy(processed).unsqueeze(0).to(self.device)  # shape (1, 150, 126)
        lengths = compute_seq_lengths(tensor_x)

        with torch.inference_mode():
            logits = self.model(tensor_x, lengths)
            probs  = torch.softmax(logits, dim=-1)

        k = max(1, min(top_k, self.num_classes))
        top_probs, top_indices = probs.topk(k, dim=-1)

        top_probs_list   = top_probs[0].cpu().tolist()
        top_indices_list = top_indices[0].cpu().tolist()

        top_k_list = [
            {
                "label": self.id_to_label[idx],
                "confidence": prob,
                "class_id": idx,
            }
            for prob, idx in zip(top_probs_list, top_indices_list)
        ]

        return {
            "label": top_k_list[0]["label"],
            "confidence": top_k_list[0]["confidence"],
            "class_id": top_k_list[0]["class_id"],
            "top_k": top_k_list,
            "raw_frame_count": raw_frame_count,
            "effective_seq_len": int(lengths[0].item()),
        }

    def predict_file(
        self,
        h5_path: Union[str, Path],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Classify landmarks loaded directly from an HDF5 file.

        Args:
            h5_path : path to .h5 file containing the 'intermediate' dataset.
            top_k   : number of top predictions to return (default: 5).

        Returns:
            dict containing prediction results.
        """
        raw = load_h5_file(Path(h5_path))
        return self.predict(raw, top_k=top_k)


def predict(
    raw_landmarks: Union[np.ndarray, torch.Tensor],
    checkpoint_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience functional API to classify a landmark sequence.

    Args:
        raw_landmarks   : raw MediaPipe array of shape (frames, 2, 21, 3).
        checkpoint_path : optional checkpoint path (default: best_model.pt).
        top_k           : number of top predictions (default: 5).

    Returns:
        prediction result dictionary.
    """
    predictor = SignPredictor(checkpoint_path=checkpoint_path)
    return predictor.predict(raw_landmarks, top_k=top_k)


def parse_args():
    parser = argparse.ArgumentParser(description="ISL500 Model Inference Test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (default: models/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of test sample from test mapping (default: 0)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to a specific .h5 file to predict",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 65)
    print("  ISL500 Sign Language Model -- Standalone Inference Test")
    print("=" * 65)

    predictor = SignPredictor(checkpoint_path=args.checkpoint)

    if args.path:
        sample_path = Path(args.path)
        if not sample_path.exists():
            raise FileNotFoundError(f"Specified sample file does not exist: {sample_path}")
        true_label = get_label_from_path(sample_path)
    else:
        print("[*] Loading test mapping...")
        test_df = load_mapping("test")
        if args.sample_index >= len(test_df) or args.sample_index < 0:
            raise IndexError(f"sample-index {args.sample_index} out of range (0 to {len(test_df)-1})")
        row = test_df.iloc[args.sample_index]
        sample_path = Path(row["local_path"])
        true_label = row["label"]

    print(f"  Sample Path   : {sample_path}")
    print(f"  True Label    : {true_label}")

    result = predictor.predict_file(sample_path, top_k=5)

    is_match = (result["label"] == true_label)
    match_str = "[MATCH]" if is_match else "[MISMATCH]"

    # Check rank in top-k
    rank_str = "None"
    for i, item in enumerate(result["top_k"], 1):
        if item["label"] == true_label:
            rank_str = f"Rank {i}"
            break

    print(f"  Raw Frames    : {result['raw_frame_count']}")
    print(f"  Active Frames : {result['effective_seq_len']}")
    print(f"  Predicted     : {result['label']} (Confidence: {result['confidence']*100:.2f}%)")
    print(f"  Outcome       : {match_str} (Ground truth in top-5: {rank_str})")
    print("-" * 65)
    print("  Top-5 Predictions:")
    for rank, item in enumerate(result["top_k"], 1):
        indicator = " <-- TRUE" if item["label"] == true_label else ""
        print(f"    {rank}. {item['label']:<20} {item['confidence']*100:>6.2f}%  (id: {item['class_id']:>3}){indicator}")
    print("=" * 65 + "\n")
