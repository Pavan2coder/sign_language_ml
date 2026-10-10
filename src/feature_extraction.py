"""
feature_extraction.py
=====================
STEP 4 of the ISL Sign Language Recognition ML Pipeline:
PyTorch Dataset wrapper for lazy HDF5 loading.

Why this module exists
-----------------------
PyTorch's DataLoader needs a Dataset object that returns (features, label)
pairs on demand. This module wraps the existing preprocessing pipeline and
HDF5 loader into a Dataset that:

  - holds only a lightweight DataFrame of paths and labels in memory,
  - opens and closes exactly ONE HDF5 file per __getitem__ call,
  - applies the full preprocessing pipeline to produce (150, 126) arrays,
  - returns a (torch.FloatTensor, int) pair ready for the DataLoader.

The full dataset is ~69k clips x (150, 126) x 4 bytes ≈ 5.5 GB.
Lazy loading keeps peak RAM well below 1 GB during training.

Public API
----------
ISLDataset(df, label_to_id, max_seq_len=MAX_SEQ_LEN)
    PyTorch Dataset. One index -> one preprocessed (features, label_id) pair.

compute_seq_lengths(features_batch) -> torch.LongTensor
    Real (non-padding) length of each sequence in a batch.
    Used to pack sequences so the GRU ignores padding.

make_dataloaders(train_df, val_df, label_to_id, batch_size, num_workers=0)
    Build train and validation DataLoaders from the two DataFrames.
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
