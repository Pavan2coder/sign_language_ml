"""
train.py
========
STEP 5 of the ISL Sign Language Recognition ML Pipeline:
GRU-based sequence classifier with full training loop.

Usage
-----
python src/train.py --check          # environment / CUDA check
python src/train.py --smoke          # 2-epoch GPU smoke test
python src/train.py                  # full training (30 epochs)
python src/train.py --resume         # resume from latest checkpoint
python src/train.py --epochs 30 --batch-size 32 --lr 1e-3

Model
-----
GRUClassifier:
  Input  : (batch, 150, 126)
  GRU    : 2 stacked layers, hidden=256, dropout=0.3
  Head   : LayerNorm -> Linear(256,256) -> ReLU -> Dropout -> Linear(256,500)
  Output : (batch, 500) logits

Checkpoints
-----------
latest_checkpoint.pt  — saved after every epoch (for resume)
best_model.pt         — saved when val_loss improves (for evaluation)
Both are written atomically: temp file -> rename.
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
from typing import Dict, List, Optional, Tuple

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
