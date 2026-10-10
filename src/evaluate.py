"""
evaluate.py
===========
STEP 6 of the ISL Sign Language Recognition ML Pipeline:
Final evaluation on the untouched test split.

Usage
-----
python src/evaluate.py                        # evaluate best_model.pt
python src/evaluate.py --checkpoint path.pt   # specific checkpoint
python src/evaluate.py --smoke                # 100-sample sanity check

The test split is NEVER used during training or model selection.
This script is the only place it is read.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR, BEST_CKPT_NAME, REPORT_DIR,
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
    DEFAULT_BATCH_SIZE,
)
from src.data_loader        import load_mapping
from src.feature_extraction import ISLDataset, compute_seq_lengths
from src.train              import GRUClassifier, get_device


def evaluate(ckpt_path=None, batch_size=DEFAULT_BATCH_SIZE, smoke=False):
    """
    Load best checkpoint and evaluate on the untouched test split.

    Steps:
      1. Resolve checkpoint path (default: best_model.pt)
      2. Load checkpoint; extract label_to_id mapping
      3. Build GRUClassifier and load weights
      4. Load test mapping (never used during training)
      5. Run inference — no gradients, no parameter updates
      6. Compute accuracy, classification report, confusion matrix
      7. Save outputs to reports/

    Args:
        ckpt_path  : Path to a .pt file; defaults to best_model.pt.
        batch_size : evaluation batch size.
        smoke      : if True, use only 100 test samples.

    Returns:
        dict with test_loss, accuracy, report_str, confusion_matrix.
    """
    if ckpt_path is None:
        ckpt_path = CHECKPOINT_DIR / BEST_CKPT_NAME

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run training first:  python src/train.py"
        )

    device = get_device()

    # Load checkpoint
    print(f"[1] Loading checkpoint: {ckpt_path}")
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)
    print(f"    Epoch: {ckpt.get('epoch','?')}  "
          f"Best val loss: {ckpt.get('best_val_loss', float('nan')):.4f}  "
          f"Classes: {num_classes}")

    # Build model and load weights
    print("[2] Building model...")
    model = GRUClassifier(
        input_size=FEATURE_DIM, hidden_size=GRU_HIDDEN_SIZE,
        num_layers=GRU_NUM_LAYERS, num_classes=num_classes,
        dropout=GRU_DROPOUT, fc_hidden=FC_HIDDEN_SIZE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load test split (only happens here — never during training)
    print("[3] Loading test mapping...")
    test_df = load_mapping("test")
    print(f"    Test samples: {len(test_df):,}")

    if smoke:
        test_df = test_df.sample(min(100, len(test_df)), random_state=42).reset_index(drop=True)
        print(f"    [smoke] Using {len(test_df)} samples")

    unknown = set(test_df["label"].unique()) - set(label_to_id.keys())
    if unknown:
        print(f"    WARNING: {len(unknown)} test labels not in encoder.")
        test_df = test_df[test_df["label"].isin(label_to_id)].reset_index(drop=True)
