"""
train.py
========
STEP 5 of the ISL Sign Language Recognition ML Pipeline:
GRU-based sequence classifier with full training loop.

Usage
-----
python src/train.py --check
python src/train.py --smoke
python src/train.py
python src/train.py --resume
python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    MODEL_DIR, REPORT_DIR, CHECKPOINT_DIR,
    LATEST_CKPT_NAME, BEST_CKPT_NAME, LABEL_MAP_NAME, HISTORY_CSV_NAME,
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    RANDOM_SEED, VAL_FRACTION,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    PATIENCE,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
)
from src.data_loader        import load_mapping
from src.preprocessing      import build_label_encoder
from src.feature_extraction import make_dataloaders, compute_seq_lengths

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device selection ──────────────────────────────────────────────────────────

def get_device(require_cuda: bool = False) -> torch.device:
    """
    Select computation device and print GPU details.

    Args:
        require_cuda : raise RuntimeError if CUDA is unavailable.

    Returns:
        torch.device — cuda:0 or cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props  = torch.cuda.get_device_properties(0)
        print(f"[device] GPU        : {props.name}")
        print(f"[device] VRAM       : {props.total_memory / 1024**3:.1f} GB")
        print(f"[device] CUDA SM    : {props.major}.{props.minor}")
        print(f"[device] PyTorch    : {torch.__version__}")
    else:
        if require_cuda:
            raise RuntimeError(
                "CUDA not available but --require-cuda was set.\n"
                "Install:  pip install torch "
                "--index-url https://download.pytorch.org/whl/cu128"
            )
        device = torch.device("cpu")
        print("[device] CUDA not available — using CPU (training will be slow).")
    return device
