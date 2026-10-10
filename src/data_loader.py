"""
data_loader.py
==============
STEP 2 of the ISL Sign Language Recognition ML Pipeline:
Reusable Dataset Loader.

Public API
----------
load_mapping(split)       -> pd.DataFrame
get_label_from_path(path) -> str
load_h5_file(path)        -> np.ndarray  shape (frames, 2, 21, 3)
load_sample(path, label)  -> (np.ndarray, str)
"""

import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    H5_DATASET_NAME,
    EXPECTED_LANDMARK_SHAPE,
)
from src.dataset_analysis import (
    load_mapping_csv,
    enrich_dataframe,
    _JUNK_VALUES,
)


# ── load_mapping ──────────────────────────────────────────────────────────────

def load_mapping(split: str) -> pd.DataFrame:
    """
    Load and resolve the official train or test mapping CSV.

    Returns a DataFrame with columns: raw_path, status, user_dir,
    disk_filename, local_path, label.

    Header artefacts (fold_1, fold_3, fold_4) are removed automatically.

    Args:
        split : "train" or "test"

    Raises:
        ValueError        — unknown split name
        FileNotFoundError — CSV file not found
        RuntimeError      — resolved files missing on disk
    """
    split_clean = split.strip().lower()
    if split_clean == "train":
        csv_path = TRAIN_CSV
    elif split_clean == "test":
        csv_path = TEST_CSV
    else:
        raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'test'.")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {csv_path}\n"
            "Check DATASET_ROOT in src/config.py."
        )

    raw_df = load_mapping_csv(csv_path)
    df     = enrich_dataframe(raw_df)

    n_before = len(df)
    df       = df[df["status"] != "exceptional"].copy()
    removed  = n_before - len(df)
    if removed:
        print(f"[load_mapping] Removed {removed} header-artefact row(s) "
              f"from {split_clean} mapping.")

    missing = df[~df["file_exists"]]
    if not missing.empty:
        raise RuntimeError(
            f"{len(missing)} resolved path(s) missing on disk.\n"
            f"First: {missing['local_path'].head(3).tolist()}\n"
            "Run src/dataset_analysis.py to investigate."
        )

    df = df.drop(columns=["file_exists"]).reset_index(drop=True)
    return df


# ── get_label_from_path ───────────────────────────────────────────────────────

def get_label_from_path(local_path: Path) -> str:
    """Return the gesture label for an already-resolved local H5 path."""
    from src.dataset_analysis import extract_label
    return extract_label(Path(local_path).name)


# ── load_h5_file ──────────────────────────────────────────────────────────────

def load_h5_file(path: Path) -> np.ndarray:
    """
    Load the 'intermediate' landmark array from one HDF5 file.

    Returns np.ndarray, dtype float32, shape (frames, 2, 21, 3).

    Raises:
        FileNotFoundError — file not found on disk
        KeyError          — 'intermediate' dataset absent
        ValueError        — shape does not match (frames, 2, 21, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as hf:
        if H5_DATASET_NAME not in hf:
            raise KeyError(
                f"Dataset '{H5_DATASET_NAME}' not in {path}. "
                f"Available: {list(hf.keys())}"
            )
        data = hf[H5_DATASET_NAME][:]

    data = data.astype(np.float32)

    if data.ndim != 4:
        raise ValueError(
            f"Expected 4-D (frames, 2, 21, 3), got {data.ndim}-D {data.shape}"
        )
    if data.shape[1:] != EXPECTED_LANDMARK_SHAPE:
        raise ValueError(
            f"Per-frame shape mismatch in {path}.\n"
            f"  Expected: {EXPECTED_LANDMARK_SHAPE}\n"
            f"  Got     : {data.shape[1:]}"
        )
    return data


# ── load_sample ───────────────────────────────────────────────────────────────

def load_sample(path: Path, label: str) -> Tuple[np.ndarray, str]:
    """
    Load one labelled landmark sample.

    Convenience wrapper used in training loops:
        data, label = load_sample(row["local_path"], row["label"])

    Args:
        path  : Path to the .h5 file.
        label : gesture label string (e.g. "Absent").

    Returns:
        (data, label) where data is float32 shape (frames, 2, 21, 3).
    """
    return load_h5_file(path), label


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("data_loader.py — self-test")

    train_df = load_mapping("train")
    test_df  = load_mapping("test")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test  rows : {len(test_df):,}")

    row  = train_df.iloc[0]
    lbl  = get_label_from_path(row["local_path"])
    data, _ = load_sample(row["local_path"], row["label"])

    print(f"  First sample: label={lbl}  shape={data.shape}  dtype={data.dtype}")
    print(f"  Finite: {np.all(np.isfinite(data))}")
    assert lbl == row["label"], "Label mismatch"
    assert data.shape[1:] == (2, 21, 3), "Shape mismatch"
    print("  PASSED")
