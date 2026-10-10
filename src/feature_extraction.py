"""
feature_extraction.py
=====================
STEP 4 of the ISL Sign Language Recognition ML Pipeline:
PyTorch Dataset wrapper for lazy HDF5 loading.

Public API
----------
ISLDataset(df, label_to_id, max_seq_len=MAX_SEQ_LEN)
    PyTorch Dataset. One index -> one preprocessed (features, label_id) pair.

compute_seq_lengths(features_batch) -> torch.LongTensor
    Real (non-padding) length of each sequence in a batch.
    Used to pack sequences so the GRU ignores padding frames.

make_dataloaders(train_df, val_df, label_to_id, batch_size, num_workers=0)
    Build train and validation DataLoaders from two DataFrames.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MAX_SEQ_LEN, FEATURE_DIM, DEFAULT_BATCH_SIZE
from src.data_loader   import load_h5_file
from src.preprocessing import preprocess_sample


# ── ISLDataset ────────────────────────────────────────────────────────────────

class ISLDataset(Dataset):
    """
    PyTorch Dataset for ISL500 MediaPipe landmark sequences.

    Each __getitem__ call opens one HDF5, preprocesses, and returns
    (FloatTensor(MAX_SEQ_LEN, 126), int label_id).
    No landmark data is cached between calls.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        label_to_id: Dict[str, int],
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        self._paths       = df["local_path"].tolist()
        self._label_ids   = [label_to_id[lbl] for lbl in df["label"].tolist()]
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, int]:
        raw      = load_h5_file(self._paths[idx])
        features = preprocess_sample(raw, self._max_seq_len)
        return torch.from_numpy(features), self._label_ids[idx]


# ── compute_seq_lengths ───────────────────────────────────────────────────────

def compute_seq_lengths(features_batch: torch.Tensor) -> torch.LongTensor:
    """
    Return the real (non-padding) frame count for each sequence in a batch.

    After preprocessing, real frames have at least one non-zero value.
    Padding frames are ALL exactly zero.

    Args:
        features_batch : torch.Tensor, shape (batch, seq_len, feature_dim).

    Returns:
        torch.LongTensor, shape (batch,), values in [1, seq_len].
    """
    real_mask = features_batch.abs().sum(dim=-1) > 0   # (B, T)
    B         = features_batch.size(0)
    lengths   = torch.ones(B, dtype=torch.long)

    for i in range(B):
        real_idx = real_mask[i].nonzero(as_tuple=False)
        if len(real_idx) > 0:
            lengths[i] = real_idx[-1].item() + 1

    return lengths


# ── make_dataloaders ──────────────────────────────────────────────────────────

def make_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    label_to_id: Dict[str, int],
    batch_size:  int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for training and validation splits.

    Why num_workers=0 by default on Windows:
        Python multiprocessing on Windows uses 'spawn', which re-imports the
        entire module tree per worker. With num_workers=0 all loading runs in
        the main process — slower but avoids Windows spawn overhead and HDF5
        locking issues.

    Args:
        train_df    : training DataFrame (rows from train CSV, no test data).
        val_df      : validation DataFrame (subset of train data only).
        label_to_id : {label_str: int_id} mapping.
        batch_size  : samples per batch.
        num_workers : DataLoader worker processes (0 = main process).

    Returns:
        (train_loader, val_loader)
    """
    train_ds = ISLDataset(train_df, label_to_id)
    val_ds   = ISLDataset(val_df,   label_to_id)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    return train_loader, val_loader
