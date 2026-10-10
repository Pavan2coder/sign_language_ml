"""
train.py — STEP 5: GRU classifier + training loop.
Usage: python src/train.py [--check | --smoke | --resume]
       python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""

import argparse, csv, json, random, signal, sys, time
from pathlib import Path
from typing  import Dict, Optional, Tuple

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
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES, RANDOM_SEED, VAL_FRACTION,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    PATIENCE, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
)
from src.data_loader        import load_mapping
from src.preprocessing      import build_label_encoder
from src.feature_extraction import make_dataloaders, compute_seq_lengths

try:
    from tqdm import tqdm; HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(require_cuda=False):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[device] GPU: {p.name}  VRAM: {p.total_memory/1024**3:.1f} GB  SM: {p.major}.{p.minor}")
        return torch.device("cuda:0")
    if require_cuda:
        raise RuntimeError("CUDA not available.")
    print("[device] CPU mode."); return torch.device("cpu")


class GRUClassifier(nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=GRU_HIDDEN_SIZE,
                 num_layers=GRU_NUM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=GRU_DROPOUT, fc_hidden=FC_HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.gru  = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fc_hidden, num_classes))

    def forward(self, x, lengths=None):
        if lengths is not None:
            lc = lengths.clamp(min=1, max=x.size(1)).cpu()
            sl, si = lc.sort(descending=True)
            packed = nn.utils.rnn.pack_padded_sequence(x[si], sl, batch_first=True, enforce_sorted=True)
            _, h_n = self.gru(packed)
            _, ui  = si.sort(); h_last = h_n[-1][ui]
        else:
            _, h_n = self.gru(x); h_last = h_n[-1]
        return self.head(h_last)


def make_train_val_split(train_df, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    val_idx, tr_idx = [], []
    for _, group in train_df.groupby("label"):
        idx = group.index.tolist(); n_val = max(1, int(len(idx)*val_fraction))
        rng.shuffle(idx); val_idx.extend(idx[:n_val]); tr_idx.extend(idx[n_val:])
    return (train_df.loc[tr_idx].reset_index(drop=True),
            train_df.loc[val_idx].reset_index(drop=True))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _atomic_save(obj: dict, path: Path) -> None:
    """Write to .tmp then rename — protects the previous checkpoint on crash."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_loss, patience_counter, label_to_id, cfg):
    """
    Save a full resumable checkpoint.

    Contents: model weights, optimizer state, scheduler state, completed
    epoch, best_val_loss, patience_counter, label_to_id mapping,
    training config dict, and all RNG states.
    """
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "best_val_loss":        best_val_loss,
        "patience_counter":     patience_counter,
        "label_to_id":          label_to_id,
        "config":               cfg,
        "rng_state": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    _atomic_save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Restore model / optimizer / scheduler from a checkpoint.

    RNG states are always loaded onto CPU regardless of target device.

    Returns the full checkpoint dict so the caller can read epoch,
    best_val_loss, patience_counter, and label_to_id.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    rng = ckpt.get("rng_state", {})
    if rng.get("python"):  random.setstate(rng["python"])
    if rng.get("numpy"):   np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        t = rng["torch"]
        torch.set_rng_state(t.cpu() if hasattr(t, "cpu") else t)
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu() if hasattr(s, "cpu") else s for s in rng["cuda"]])
    return ckpt
