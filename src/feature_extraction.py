"""
feature_extraction.py
=====================
STEP 4 of the ISL Sign Language Recognition ML Pipeline:
PyTorch Dataset wrapper for lazy HDF5 loading.

Public API
----------
ISLDataset(df, label_to_id, max_seq_len=MAX_SEQ_LEN)
compute_seq_lengths(features_batch) -> torch.LongTensor
make_dataloaders(train_df, val_df, label_to_id, batch_size, num_workers=0)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
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

    Each __getitem__ call:
      1. Opens exactly one HDF5 file with load_h5_file().
      2. Applies the full preprocessing pipeline via preprocess_sample().
      3. Returns (FloatTensor shape (MAX_SEQ_LEN, 126), int label_id).

    Only a lightweight list of paths and pre-computed integer IDs is held
    in memory — no landmark arrays are cached between calls.

    Args:
        df          : DataFrame with 'local_path' and 'label' columns.
        label_to_id : {label_str: int_id} from build_label_encoder().
        max_seq_len : frames per sequence after padding/truncation.
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
        raw      = load_h5_file(self._paths[idx])          # (F, 2, 21, 3)
        features = preprocess_sample(raw, self._max_seq_len)  # (T, 126)
        return torch.from_numpy(features), self._label_ids[idx]
