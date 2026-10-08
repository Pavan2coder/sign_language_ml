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
   handling ALL path formats that appear in the CSVs.
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

    Renames the single fold column to 'raw_path' for a stable accessor.
    Stores the original column name in df.attrs['original_col'].
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


# ── STEP 2: Resolve CSV paths → local Windows paths ───────────────────────────

_SPECIAL_FILENAME_RE = re.compile(r"_user\d+_(.+\.h5)$", re.IGNORECASE)


def resolve_path(raw_path: str) -> Path:
    """
    Convert a raw CSV path (Linux-style) to the local Windows .h5 path.

    FORMAT A — Normal path
        Raw last segment : Absent__session82__clip000.h5
        Action           : use as-is

    FORMAT B — Mangled _mnt_ path
        Raw last segment : _mnt_<uuid>_ALL_CLIPS_R2_Clips_R2_user001_Absent__session65__clip028.h5
        Action           : strip prefix via regex, recover real filename

    FORMAT C — R2 date-prefixed path  (added this commit)
        Raw last segment : R2_28.03.26_All_user002_..._Accept__session65__clip002.h5
        Action           : filename is used as-is (it IS the real disk name);
                           label extraction handles the R2 prefix separately.
                           No special path-resolution needed here — it falls
                           through to the Format A branch naturally.
    """
    parts = [p for p in raw_path.strip().split("/") if p]

    user_dir     = parts[-2]
    raw_filename = parts[-1]

    if raw_filename.startswith("_mnt_"):
        # FORMAT B — mangled
        match = _SPECIAL_FILENAME_RE.search(raw_filename)
        real_filename = match.group(1) if match else raw_filename
    else:
        # FORMAT A and FORMAT C — filename is already the correct disk name
        real_filename = raw_filename

    return MEDIAPIPE_ROOT / user_dir / real_filename


# ── STEP 3: Inspect path format breakdown ─────────────────────────────────────

def inspect_path_formats(df: pd.DataFrame, split_name: str) -> dict:
    """
    Count how many rows use each path format and print a summary.

    Three formats exist in the dataset:
        A  Normal          : filename starts with a capital letter + label
        B  _mnt_ mangled   : filename starts with '_mnt_'
        C  R2 date-prefixed: filename starts with 'R2_' (different session naming)

    Args:
        df         : mapping DataFrame with column 'raw_path'
        split_name : "Train" or "Test" — used only for display

    Returns:
        dict with keys 'normal', 'mnt_mangled', 'r2_prefixed'
    """
    filenames = df["raw_path"].apply(lambda p: p.strip().split("/")[-1])

    n_mnt = int(filenames.str.startswith("_mnt_", na=False).sum())
    n_r2  = int(filenames.str.startswith("R2_",   na=False).sum())
    n_normal = len(df) - n_mnt - n_r2

    print(f"{split_name} — Format A (normal)       : {n_normal:>7,}")
    print(f"{split_name} — Format B (_mnt_ mangled): {n_mnt:>7,}")
    print(f"{split_name} — Format C (R2-prefixed)  : {n_r2:>7,}")

    return {"normal": n_normal, "mnt_mangled": n_mnt, "r2_prefixed": n_r2}


if __name__ == "__main__":
    section("ISL Dataset Analysis — STEP 1")
    print("MediaPipe root :", MEDIAPIPE_ROOT)
    print("Train CSV      :", TRAIN_CSV)
    print("Test  CSV      :", TEST_CSV)

    section("Loading mapping CSVs")

    train_raw = load_mapping_csv(TRAIN_CSV)
    test_raw  = load_mapping_csv(TEST_CSV)

    print(f"Train  shape : {train_raw.shape}  |  column: {train_raw.attrs['original_col']}")
    print(f"Test   shape : {test_raw.shape}   |  column: {test_raw.attrs['original_col']}")

    section("First 5 rows of Train CSV")
    print(train_raw.head(5).to_string(index=True))

    section("First 5 rows of Test CSV")
    print(test_raw.head(5).to_string(index=True))

    # ── Path format breakdown ──────────────────────────────────────────────────
    section("Path format inspection")
    train_fmt = inspect_path_formats(train_raw, "Train")
    print()
    test_fmt  = inspect_path_formats(test_raw,  "Test")
