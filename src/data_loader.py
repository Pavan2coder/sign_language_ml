"""
data_loader.py
==============
STEP 2 of the ISL Sign Language Recognition ML Pipeline:
Reusable Dataset Loader.

What this module provides
--------------------------
load_mapping(split)
    Read the official CSV for 'train' or 'test', resolve every row to its
    local H5 file, extract the correct sign label, and return a clean
    pandas DataFrame ready for downstream use.
    Header artefacts (fold_1, fold_3, fold_4) are removed automatically.

get_label_from_path(local_path)
    Given a resolved local Windows path, return the gesture label string.
    Delegates to the verified label-extraction logic in dataset_analysis.py.

load_h5_file(path)
    Open one H5 file, read the 'intermediate' dataset, validate its shape,
    cast to float32, and return the NumPy array.
    The file is opened and closed immediately — no data stays in memory
    after the function returns except the one array you asked for.

load_sample(path, label)
    Convenience wrapper: calls load_h5_file and returns (array, label).
    Use this in training loops — call it only for the sample you need
    right now, not all at once.

Design principles
-----------------
- No global caches, no loading all files at startup.
  The mapping DataFrame is lightweight (strings + paths only).
  H5 arrays are loaded one at a time, on demand.
- The resolver and label extractor are IMPORTED from dataset_analysis.py,
  not duplicated. There is exactly one place where path resolution lives.
- Validation errors are raised as specific Python exceptions with messages
  that tell you exactly what went wrong and where.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import h5py
import numpy as np
import pandas as pd

# ── Ensure the project root is on sys.path so sibling imports work ────────────
# This lets you run  `python src/data_loader.py`  from the project root
# without installing the package.
_SCRIPT_DIR   = Path(__file__).resolve().parent   # .../sign-lan/src
_PROJECT_ROOT = _SCRIPT_DIR.parent                # .../sign-lan
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project config ────────────────────────────────────────────────────────────
from src.config import (
    TRAIN_CSV,               # D:\ISL-DATA\MappingFiles\PersonDependentTrain.csv
    TEST_CSV,                # D:\ISL-DATA\MappingFiles\PersonDependentTest.csv
    H5_DATASET_NAME,         # "intermediate"
    EXPECTED_LANDMARK_SHAPE, # (2, 21, 3) — the per-frame shape
)

# ── Import the verified resolver and label extractor from STEP 1 ──────────────
# We import the functions directly rather than copy-pasting their code.
# This guarantees that any future fix to the resolver automatically applies
# to the loader as well — there is only one source of truth.
from src.dataset_analysis import (
    load_mapping_csv,   # reads a CSV, renames the column to 'raw_path'
    enrich_dataframe,   # adds local_path, label, status, file_exists columns
    _JUNK_VALUES,       # frozenset {"fold_1", "fold_3", "fold_4"}
)


# ──────────────────────────────────────────────────────────────────────────────
# load_mapping
# ──────────────────────────────────────────────────────────────────────────────

def load_mapping(split: str) -> pd.DataFrame:
    """
    Load and resolve the official train or test mapping CSV.

    This is the main entry point for getting the list of samples.
    It returns a DataFrame where every row represents one valid H5 clip,
    with the local file path and the gesture label already extracted.

    Steps performed internally:
        1. Read the CSV (fold_0 / fold_2 column -> renamed to 'raw_path').
        2. Resolve every raw path to a local Windows Path object using the
           same regex-based logic that was verified in dataset_analysis.py.
        3. Extract the gesture label from each resolved filename.
        4. Drop the 3 header-artefact rows (fold_1, fold_3, fold_4).
        5. Drop any rows whose local file is not found on disk
           (none expected after our verification, but checked defensively).

    Args:
        split : "train" or "test"  (case-insensitive)

    Returns:
        pd.DataFrame with columns:
            raw_path      - original string from the CSV
            status        - "normal" or "mangled"
            user_dir      - e.g. "ISL_DATA_USER001"
            disk_filename - verbatim on-disk filename
            local_path    - resolved Windows Path object
            label         - gesture class string, e.g. "Absent"

    Raises:
        ValueError        - if split is not "train" or "test"
        FileNotFoundError - if the CSV file itself is missing
        RuntimeError      - if any resolved files are unexpectedly absent
    """
    # ── Validate split argument ───────────────────────────────────────────────
    split_clean = split.strip().lower()
    if split_clean == "train":
        csv_path = TRAIN_CSV
    elif split_clean == "test":
        csv_path = TEST_CSV
    else:
        raise ValueError(
            f"Unknown split {split!r}. "
            "Expected 'train' or 'test'."
        )

    # ── Load and rename the CSV ───────────────────────────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {csv_path}\n"
            "Check that DATASET_ROOT in src/config.py points to the right location."
        )

    raw_df = load_mapping_csv(csv_path)          # 'raw_path' column

    # ── Resolve paths and extract labels (no H5 data loaded) ─────────────────
    # enrich_dataframe() adds: status, user_dir, disk_filename,
    #                          local_path, file_exists, label
    df = enrich_dataframe(raw_df)

    # ── Remove header-artefact rows (fold_1, fold_3, fold_4) ─────────────────
    # These rows are CSV column-header values that leaked into the data when
    # per-fold CSVs were concatenated.  They are NOT real file paths.
    n_before = len(df)
    df = df[df["status"] != "exceptional"].copy()
    n_artefacts = n_before - len(df)
    if n_artefacts > 0:
        print(f"[load_mapping] Removed {n_artefacts} header-artefact row(s) "
              f"from {split_clean} mapping.")

    # ── Defensive check: no missing files expected ────────────────────────────
    missing = df[~df["file_exists"]]
    if not missing.empty:
        sample_paths = missing["local_path"].head(3).tolist()
        raise RuntimeError(
            f"{len(missing)} resolved path(s) in the {split_clean} split "
            f"do not exist on disk.\n"
            f"First missing: {sample_paths}\n"
            "Run src/dataset_analysis.py to investigate."
        )

    # ── Drop the file_exists column — it is always True in the result ─────────
    df = df.drop(columns=["file_exists"])

    # Reset index so rows are numbered 0 ... N-1
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick sanity check — will be replaced with full self-test in commit 5
    print("load_mapping smoke-check...")
    train_df = load_mapping("train")
    test_df  = load_mapping("test")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test  rows : {len(test_df):,}")
    print("OK")
