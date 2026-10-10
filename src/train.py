"""
train.py
========
STEP 5 — GRU-based sequence classifier with full training loop.

Usage:  python src/train.py [--check | --smoke | --resume]
        python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""

import argparse
import csv
import json
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
    PATIENCE, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
)
from src.data_loader        import load_mapping
from src.preprocessing      import build_label_encoder
from src.feature_extraction import make_dataloaders, compute_seq_lengths

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        p = torch.cuda.get_device_properties(0)
        print(f"[device] GPU: {p.name}  VRAM: {p.total_memory/1024**3:.1f} GB  SM: {p.major}.{p.minor}")
    else:
        if require_cuda:
            raise RuntimeError("CUDA not available. Install torch with cu128 wheel.")
        device = torch.device("cpu")
        print("[device] CUDA unavailable — using CPU.")
    return device


class GRUClassifier(nn.Module):
    """Stacked GRU classifier. Input (B, 150, 126) -> logits (B, 500)."""

    def __init__(
        self,
        input_size:  int   = FEATURE_DIM,
        hidden_size: int   = GRU_HIDDEN_SIZE,
        num_layers:  int   = GRU_NUM_LAYERS,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = GRU_DROPOUT,
        fc_hidden:   int   = FC_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        gru_drop = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=gru_drop,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fc_hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.LongTensor] = None) -> torch.Tensor:
        if lengths is not None:
            lc = lengths.clamp(min=1, max=x.size(1)).cpu()
            sl, si = lc.sort(descending=True)
            packed = nn.utils.rnn.pack_padded_sequence(
                x[si], sl, batch_first=True, enforce_sorted=True)
            _, h_n = self.gru(packed)
            _, ui  = si.sort()
            h_last = h_n[-1][ui]
        else:
            _, h_n = self.gru(x)
            h_last = h_n[-1]
        return self.head(h_last)


# ── Stratified train/validation split ────────────────────────────────────────

def make_train_val_split(
    train_df:     pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed:         int   = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train/validation split from the training DataFrame.

    Stratification ensures every class appears in both sub-splits.
    We use training data ONLY — the test CSV is never touched here.

    Strategy:
        For each class, sample val_fraction of its clips into val.
        Remaining clips stay in train_sub.
        With ~139 samples/class, 15% gives ~21 val samples per class.

    Returns:
        (train_sub_df, val_df)
    """
    rng           = np.random.default_rng(seed)
    val_indices   = []
    train_indices = []

    for label, group in train_df.groupby("label"):
        idx   = group.index.tolist()
        n_val = max(1, int(len(idx) * val_fraction))
        rng.shuffle(idx)
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])

    train_sub = train_df.loc[train_indices].reset_index(drop=True)
    val_sub   = train_df.loc[val_indices].reset_index(drop=True)
    return train_sub, val_sub
