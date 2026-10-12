"""
hybrid_predict.py
=================
Dual-Recognizer Inference Engine for Indian Sign Language Recognition.

Integrates the trained ISL500 baseline model with an independently trained
custom-vocabulary classifier (for isolated signs and continuous phrases).

Supported Routing Strategies
----------------------------
1. 'isl500' : Direct prediction via the trained 500-class ISL model.
2. 'custom' : Direct prediction via the custom-vocabulary classifier.
3. 'hybrid' : Principled arbitration using calibrated threshold gating
              and margin verification, rather than naive probability comparison.

Usage
-----
from src.hybrid_predict import HybridSignPredictor

predictor = HybridSignPredictor()
# Default hybrid mode:
result = predictor.predict(raw_landmarks, mode="hybrid")
print(result["label"], result["confidence"], result["recognizer"])
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR,
    BEST_CKPT_NAME,
    CUSTOM_CKPT_NAME,
    CUSTOM_LABEL_MAP_NAME,
    MAX_SEQ_LEN,
    FEATURE_DIM,
)
from src.predict import SignPredictor
from src.preprocessing import preprocess_sample, pad_or_truncate
from src.feature_extraction import compute_seq_lengths
from src.train import GRUClassifier, get_device

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
)
logger = logging.getLogger("hybrid_predict")


class CustomSignPredictor:
    """
    Inference engine for the separate custom-vocabulary classifier.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / CUSTOM_CKPT_NAME
        )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Custom checkpoint not found at: {self.checkpoint_path}\n"
                "Please train the custom model first using:  python src/train_custom.py"
            )

        self.device = device if device is not None else get_device()
        self._load_model()

    def _load_model(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.label_to_id: Dict[str, int] = ckpt["label_to_id"]
        self.id_to_label: Dict[int, str] = {int(v): k for k, v in self.label_to_id.items()}
        self.num_classes: int = len(self.label_to_id)
        cfg: Dict[str, Any] = ckpt.get("config", {})

        hidden_size = cfg.get("gru_hidden", 128)
        num_layers  = cfg.get("gru_layers", 2)
        fc_hidden   = cfg.get("fc_hidden", hidden_size)
        input_size  = cfg.get("feature_dim", FEATURE_DIM)

        self.model = GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=0.3,
            fc_hidden=fc_hidden,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(
        self,
        raw_landmarks: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Inference on custom vocabulary classifier.
        """
        if isinstance(raw_landmarks, torch.Tensor):
            arr = raw_landmarks.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_landmarks, dtype=np.float32)

        raw_frames = len(arr) if hasattr(arr, "__len__") else 0

        # Preprocessing
        if arr.ndim == 4:
            processed = preprocess_sample(arr, max_seq_len=MAX_SEQ_LEN)
        elif arr.ndim == 2:
            processed = pad_or_truncate(arr, max_seq_len=MAX_SEQ_LEN)
        else:
            raise ValueError(f"Invalid landmark shape: {arr.shape}")

        tensor_x = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        lengths = compute_seq_lengths(tensor_x)

        with torch.inference_mode():
            logits = self.model(tensor_x, lengths)
            probs = torch.softmax(logits, dim=-1)

        k = max(1, min(top_k, self.num_classes))
        top_probs, top_indices = probs.topk(k, dim=-1)

        top_probs_list = top_probs[0].cpu().tolist()
        top_indices_list = top_indices[0].cpu().tolist()

        top_k_list = [
            {
                "label": self.id_to_label[idx],
                "confidence": prob,
                "class_id": idx,
            }
            for prob, idx in zip(top_probs_list, top_indices_list)
        ]

        margin = 0.0
        if len(top_probs_list) >= 2:
            margin = top_probs_list[0] - top_probs_list[1]
        elif len(top_probs_list) == 1:
            margin = top_probs_list[0]

        return {
            "label": top_k_list[0]["label"],
            "confidence": top_k_list[0]["confidence"],
            "class_id": top_k_list[0]["class_id"],
            "top_k": top_k_list,
            "margin": margin,
            "raw_frame_count": raw_frames,
            "effective_seq_len": int(lengths[0].item()),
        }


class HybridSignPredictor:
    """
    Unified Dual-Recognizer Engine.

    Routes between ISL500 model and custom-vocabulary model
    without modifying existing checkpoint weights or mappings.
    """

    def __init__(
        self,
        isl500_ckpt: Optional[Union[str, Path]] = None,
        custom_ckpt: Optional[Union[str, Path]] = None,
        custom_threshold: float = 0.70,
        custom_margin_min: float = 0.20,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.custom_threshold = custom_threshold
        self.custom_margin_min = custom_margin_min

        # 1. Load original ISL500 predictor
        self.isl500_predictor = SignPredictor(checkpoint_path=isl500_ckpt, device=self.device)
        logger.info(f"Loaded ISL500 predictor: {self.isl500_predictor.num_classes} classes")

        # 2. Optionally load custom predictor if checkpoint exists
        self.custom_predictor: Optional[CustomSignPredictor] = None
        custom_path = Path(custom_ckpt) if custom_ckpt else CHECKPOINT_DIR / CUSTOM_CKPT_NAME

        if custom_path.exists():
            try:
                self.custom_predictor = CustomSignPredictor(checkpoint_path=custom_path, device=self.device)
                logger.info(f"Loaded Custom predictor: {self.custom_predictor.num_classes} classes")
            except Exception as e:
                logger.warning(f"Failed to load custom checkpoint at {custom_path}: {e}")
                self.custom_predictor = None
        else:
            logger.info("Custom model checkpoint not found; operating in ISL500-only fallback mode.")

    @property
    def has_custom_model(self) -> bool:
        return self.custom_predictor is not None

    def predict(
        self,
        raw_landmarks: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
        mode: str = "hybrid",  # "isl500", "custom", or "hybrid"
    ) -> Dict[str, Any]:
        """
        Classify landmark sequence using specified routing mode.

        Args:
            raw_landmarks : Array of shape (frames, 2, 21, 3) or (frames, 126).
            top_k         : Number of candidates to return.
            mode          : 'isl500', 'custom', or 'hybrid'.

        Returns:
            Dict containing label, confidence, recognizer, top_k, and diagnostics.
        """
        mode_clean = mode.strip().lower()

        if mode_clean == "isl500":
            res = self.isl500_predictor.predict(raw_landmarks, top_k=top_k)
            res["recognizer"] = "isl500"
            return res

        elif mode_clean == "custom":
            if self.custom_predictor is None:
                raise RuntimeError(
                    "Custom recognizer requested, but no custom model checkpoint is loaded.\n"
                    "Train custom vocabulary first using:  python src/train_custom.py"
                )
            res = self.custom_predictor.predict(raw_landmarks, top_k=top_k)
            res["recognizer"] = "custom"
            return res

        elif mode_clean == "hybrid":
            # If custom model is not available, default gracefully to isl500
            if self.custom_predictor is None:
                res = self.isl500_predictor.predict(raw_landmarks, top_k=top_k)
                res["recognizer"] = "isl500"
                res["hybrid_note"] = "Custom model unavailable; fell back to isl500"
                return res

            # Both models available: run parallel inference
            res_isl = self.isl500_predictor.predict(raw_landmarks, top_k=top_k)
            res_custom = self.custom_predictor.predict(raw_landmarks, top_k=top_k)

            # Principled Arbitration:
            # Custom vocabulary contains dedicated phrases and specific target signs.
            # We trigger the custom prediction if:
            #   1. Custom confidence exceeds calibrated threshold tau (e.g. 0.70)
            #   2. Custom margin (top1 - top2) exceeds minimum margin (e.g. 0.20)
            custom_conf = res_custom["confidence"]
            custom_margin = res_custom.get("margin", 0.0)

            should_route_custom = (
                custom_conf >= self.custom_threshold
                and custom_margin >= self.custom_margin_min
            )

            if should_route_custom:
                chosen = dict(res_custom)
                chosen["recognizer"] = "custom"
            else:
                chosen = dict(res_isl)
                chosen["recognizer"] = "isl500"

            chosen["hybrid_details"] = {
                "routing_decision": "custom" if should_route_custom else "isl500",
                "custom_confidence": custom_conf,
                "custom_margin": custom_margin,
                "isl500_confidence": res_isl["confidence"],
                "isl500_top_label": res_isl["label"],
                "custom_top_label": res_custom["label"],
            }
            return chosen

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'isl500', 'custom', or 'hybrid'.")

    def predict_file(
        self,
        file_path: Union[str, Path],
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Run inference directly on a stored .npz (custom) or .h5 (ISL500) landmark file.

        Args:
            file_path: Path to the landmark file (.npz or .h5/.hdf5).
            top_k: Number of candidate classes to return.
            mode: Routing mode ('isl500', 'custom', or 'hybrid').

        Returns:
            Dict containing prediction results augmented with file metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Landmark file not found: {path}")

        meta: Dict[str, Any] = {
            "source_file": str(path.resolve()),
            "filename": path.name,
            "file_format": path.suffix.lower(),
        }

        if path.suffix.lower() == ".npz":
            try:
                from src.record_custom import load_custom_sample
                landmarks, custom_meta = load_custom_sample(path)
                meta["custom_metadata"] = custom_meta
                if "label" in custom_meta:
                    meta["ground_truth_label"] = custom_meta["label"]
            except Exception:
                with np.load(path, allow_pickle=True) as npz:
                    key = "landmarks" if "landmarks" in npz else ("data" if "data" in npz else npz.files[0])
                    landmarks = npz[key].astype(np.float32)
                    if "label" in npz:
                        meta["ground_truth_label"] = str(npz["label"])
        elif path.suffix.lower() in (".h5", ".hdf5"):
            from src.data_loader import load_h5_file
            landmarks = load_h5_file(path)
            try:
                from src.data_loader import get_label_from_path
                meta["ground_truth_label"] = get_label_from_path(path)
            except Exception:
                pass
        else:
            raise ValueError(
                f"Unsupported landmark file format '{path.suffix}'. "
                "Supported formats: .npz, .h5, .hdf5"
            )

        res = self.predict(landmarks, top_k=top_k, mode=mode)
        res["file_metadata"] = meta
        return res


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid ISL Sign Recognition Test")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to .h5 or .npz landmark file to predict",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["isl500", "custom", "hybrid"],
        help="Recognition mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K candidate count (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictor = HybridSignPredictor()
    print("\n" + "=" * 65)
    print("  ISL Hybrid Recognizer Ready")
    print(f"  Mode           : {args.mode}")
    print(f"  ISL500 Classes : {predictor.isl500_predictor.num_classes}")
    print(f"  Custom Model   : {'Loaded' if predictor.has_custom_model else 'Not found (fallback active)'}")
    print("=" * 65)

    if args.path:
        res = predictor.predict_file(args.path, top_k=args.top_k, mode=args.mode)
        gt = res.get("file_metadata", {}).get("ground_truth_label")
        gt_info = f" (Ground Truth: {gt})" if gt else ""
        print(f"  Sample: {res['file_metadata']['filename']}{gt_info}")
        print(f"\n  Result: {res['label']} ({res['confidence']*100:.2f}%) via [{res['recognizer']}]")
        print("  Top candidates:")
        for idx, item in enumerate(res["top_k"], 1):
            print(f"    {idx}. {item['label']:<20} {item['confidence']*100:>6.2f}%")
        print()
