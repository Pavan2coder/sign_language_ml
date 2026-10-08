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

# Compiled once at module level for efficiency.
# Matches the real filename hidden inside a mangled _mnt_ filename.
#
# Example mangled filename:
#   _mnt_9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba_ALL_CLIPS_R2_Clips_R2_user001_Absent__session65__clip028.h5
#                                                                    ^^^^^^^^
#   The pattern captures everything after the last  _userNNN_  token.
#   Captured group(1) → "Absent__session65__clip028.h5"
_SPECIAL_FILENAME_RE = re.compile(
    r"_user\d+_(.+\.h5)$",
    re.IGNORECASE,
)


def resolve_path(raw_path: str) -> Path:
    """
    Convert a raw CSV path (Linux-style) to the local Windows .h5 path.

    Two formats are handled:

    FORMAT A — Normal path
    ----------------------
    Raw:
        /mnt/<uuid>/popsign/.../ISL_DATA_USER001/Absent__session82__clip000.h5
    Action:
        Take last two segments (user_dir / filename) and join with MEDIAPIPE_ROOT.
    Local:
        D:/ISL-DATA/Landmarks/MediaPipe/ISL_DATA_USER001/Absent__session82__clip000.h5

    FORMAT B — Mangled _mnt_ path  (added this commit)
    ---------------------------------------------------
    Raw:
        /mnt/<uuid>/popsign/.../ISL_DATA_USER001/_mnt_<uuid>_ALL_CLIPS_R2_Clips_R2_user001_Absent__session65__clip028.h5
    Action:
        The user directory (ISL_DATA_USER001) is still second-to-last.
        The filename is mangled — strip everything up to and including _userNNN_
        using the compiled regex to recover the real filename.
    Local:
        D:/ISL-DATA/Landmarks/MediaPipe/ISL_DATA_USER001/Absent__session65__clip028.h5
    """
    parts = [p for p in raw_path.strip().split("/") if p]

    user_dir     = parts[-2]
    raw_filename = parts[-1]

    # FORMAT B: filename starts with "_mnt_" — it is mangled
    if raw_filename.startswith("_mnt_"):
        match = _SPECIAL_FILENAME_RE.search(raw_filename)
        if match:
            real_filename = match.group(1)   # the clean filename hidden inside
        else:
            # Regex didn't match — keep original; file will be reported missing
            real_filename = raw_filename
    else:
        # FORMAT A: filename is already clean
        real_filename = raw_filename

    return MEDIAPIPE_ROOT / user_dir / real_filename


if __name__ == "__main__":
    section("ISL Dataset Analysis — STEP 1")
    print("MediaPipe root :", MEDIAPIPE_ROOT)
    print("Train CSV      :", TRAIN_CSV)
    print("Test  CSV      :", TEST_CSV)

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

    # ── Resolver smoke-test: both formats ─────────────────────────────────────
    section("Path resolver smoke-test (Format A and B)")

    normal_sample  = train_raw["raw_path"].iloc[1]   # clean path
    mangled_sample = train_raw["raw_path"].iloc[0]   # _mnt_ mangled path

    print("Format A (normal):")
    print(f"  Raw  : {normal_sample}")
    print(f"  Local: {resolve_path(normal_sample)}")
    print()
    print("Format B (_mnt_ mangled):")
    print(f"  Raw  : {mangled_sample}")
    print(f"  Local: {resolve_path(mangled_sample)}")
