"""
dataset_analysis.py
===================
STEP 1 of the ISL Sign Language Recognition ML Pipeline:
Dataset and Label Verification.

What this script does (in plain English):
------------------------------------------
1. Loads the two CSV mapping files (train and test splits).
2. Prints their shapes, column names, and a few sample rows
   so you can see exactly what the raw data looks like.
3. Figures out how each CSV row maps to a real .h5 file on disk,
   handling BOTH path formats that appear in the CSVs.
4. Verifies that every expected .h5 file actually exists locally.
5. Derives the class label from the filename.
6. Prints class statistics.
7. Checks for train/test class mismatches and duplicate rows.
8. Saves a full report to reports/dataset_analysis.txt.

NOTE: This script NEVER loads the H5 landmark arrays into memory.
      It only inspects file paths and metadata.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── Project config ────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    MEDIAPIPE_ROOT,
    REPORT_DIR,
)


# ── Utility ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a visible section header to the console."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── STEP 1: Load mapping CSVs ─────────────────────────────────────────────────

def load_mapping_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a mapping CSV into a pandas DataFrame.

    Each CSV has exactly ONE column (e.g. 'fold_0' or 'fold_2') that
    contains Linux-style absolute paths to .h5 files.  We rename that
    column to 'raw_path' so the rest of the code has a stable name to
    work with regardless of which fold the CSV came from.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with column 'raw_path' and two metadata attrs:
            original_col  – the original column name
            source_file   – the CSV file path as a string
    """
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    original_col = df.columns[0]
    df = df.rename(columns={original_col: "raw_path"})

    df.attrs["original_col"] = original_col
    df.attrs["source_file"]  = str(csv_path)

    return df


# ── STEP 2: Resolve CSV paths → local Windows paths (FORMAT A only) ───────────

def resolve_path(raw_path: str) -> Path:
    """
    Convert a raw CSV path (Linux-style, from the original dataset server)
    to the corresponding local Windows .h5 file path under MEDIAPIPE_ROOT.

    FORMAT A — Normal path (handled in this commit)
    ------------------------------------------------
    The CSV stores full Linux paths like:
        /mnt/<uuid>/popsign/ISL_Goa_Data_h5_30fps/ISL_DATA_USER001/Absent__session82__clip000.h5

    All we need are the last two segments:
        - second-to-last  → the user directory  (e.g. ISL_DATA_USER001)
        - last            → the h5 filename      (e.g. Absent__session82__clip000.h5)

    We join those two with MEDIAPIPE_ROOT to get the local Windows path:
        D:\\ISL-DATA\\Landmarks\\MediaPipe\\ISL_DATA_USER001\\Absent__session82__clip000.h5

    Other path formats (mangled _mnt_, R2-prefixed) will be added in later commits.
    """
    # Split on forward slash (POSIX path), drop empty strings from leading slash
    parts = [p for p in raw_path.strip().split("/") if p]

    # The user directory is always second-to-last, filename is last
    user_dir     = parts[-2]   # e.g. "ISL_DATA_USER001"
    raw_filename = parts[-1]   # e.g. "Absent__session82__clip000.h5"

    return MEDIAPIPE_ROOT / user_dir / raw_filename


if __name__ == "__main__":
    section("ISL Dataset Analysis — STEP 1")
    print("MediaPipe root :", MEDIAPIPE_ROOT)
    print("Train CSV      :", TRAIN_CSV)
    print("Test  CSV      :", TEST_CSV)

    # ── Load ──────────────────────────────────────────────────────────────────
    section("Loading mapping CSVs")

    train_raw = load_mapping_csv(TRAIN_CSV)
    test_raw  = load_mapping_csv(TEST_CSV)

    print(f"Train CSV  : {TRAIN_CSV}")
    print(f"  Shape    : {train_raw.shape}")
    print(f"  Column   : {train_raw.attrs['original_col']}")
    print()
    print(f"Test CSV   : {TEST_CSV}")
    print(f"  Shape    : {test_raw.shape}")
    print(f"  Column   : {test_raw.attrs['original_col']}")

    section("First 5 rows of Train CSV")
    print(train_raw.head(5).to_string(index=True))

    section("First 5 rows of Test CSV")
    print(test_raw.head(5).to_string(index=True))

    # ── Quick resolver smoke-test ──────────────────────────────────────────────
    section("Path resolver smoke-test (Format A)")
    sample = train_raw["raw_path"].iloc[1]   # pick a normal (non-mangled) row
    print(f"  Raw  : {sample}")
    print(f"  Local: {resolve_path(sample)}")
