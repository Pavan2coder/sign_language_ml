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

    FORMAT A  Normal path          — filename used as-is
    FORMAT B  _mnt_ mangled path   — real filename extracted via regex
    FORMAT C  R2 date-prefixed     — filename used as-is (same as A)
    GUARD     Junk / unparseable   — sentinel path returned
    """
    parts = [p for p in raw_path.strip().split("/") if p]

    if len(parts) < 2:
        label = parts[0] if parts else "__empty__"
        return MEDIAPIPE_ROOT / "__UNPARSEABLE__" / label

    user_dir     = parts[-2]
    raw_filename = parts[-1]

    if raw_filename.startswith("_mnt_"):
        match = _SPECIAL_FILENAME_RE.search(raw_filename)
        real_filename = match.group(1) if match else raw_filename
    else:
        real_filename = raw_filename

    return MEDIAPIPE_ROOT / user_dir / real_filename


# ── STEP 3: Inspect path format breakdown ─────────────────────────────────────

def inspect_path_formats(df: pd.DataFrame, split_name: str) -> dict:
    """Count and print how many rows use each path format."""
    filenames = df["raw_path"].apply(lambda p: p.strip().split("/")[-1])

    n_mnt  = int(filenames.str.startswith("_mnt_", na=False).sum())
    n_r2   = int(filenames.str.startswith("R2_",   na=False).sum())
    n_junk = int(df["raw_path"].apply(
        lambda p: len([x for x in p.strip().split("/") if x]) < 2
    ).sum())
    n_normal = len(df) - n_mnt - n_r2 - n_junk

    print(f"{split_name} — Format A (normal)       : {n_normal:>7,}")
    print(f"{split_name} — Format B (_mnt_ mangled): {n_mnt:>7,}")
    print(f"{split_name} — Format C (R2-prefixed)  : {n_r2:>7,}")
    print(f"{split_name} — Junk / unparseable rows : {n_junk:>7,}")

    return {"normal": n_normal, "mnt_mangled": n_mnt,
            "r2_prefixed": n_r2, "junk": n_junk}


# ── STEP 4: Extract the gesture label from a filename ─────────────────────────

def extract_label(filename: str) -> str:
    """
    Derive the ISL gesture label from an .h5 filename.

    FORMAT A/B : left of first '__' is the label.
    FORMAT C   : last capitalised word before first '__' (R2-prefixed stem).
    """
    stem          = Path(filename).stem
    first_segment = stem.split("__")[0]

    if first_segment.startswith("R2_") or (first_segment and first_segment[0].isdigit()):
        match = re.search(r"_([A-Z][a-zA-Z0-9]*)$", first_segment)
        if match:
            return match.group(1)
        return first_segment

    return first_segment


# ── STEP 5: Enrich the DataFrame ──────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add local_path, file_exists, and label columns.
    No H5 data is loaded — only path resolution and stat checks.
    """
    df = df.copy()
    df["local_path"]  = df["raw_path"].apply(resolve_path)
    df["file_exists"] = df["local_path"].apply(lambda p: p.exists())
    df["label"]       = df["local_path"].apply(lambda p: extract_label(p.name))
    return df


# ── STEP 6: File existence verification ───────────────────────────────────────

def verify_files(df: pd.DataFrame, split_name: str, n_samples: int = 5) -> dict:
    """
    Report how many .h5 files exist on disk vs how many are missing.

    Args:
        df         : enriched DataFrame (must have 'local_path' and 'file_exists')
        split_name : "Train" or "Test" — used for display only
        n_samples  : how many missing paths to print as examples (default 5)

    Returns:
        dict with keys: total, found, missing, missing_paths (list of str)

    Why files go missing
    --------------------
    The CSVs were generated on a Linux server with the full original dataset.
    When the dataset was copied locally not every clip was included.
    Additionally, the _mnt_-mangled entries encode filenames that were stored
    on an NFS mount; some of those clips may not have been exported at all.
    Missing files are NOT errors in this script — they are real data gaps
    that must be accounted for before training.
    """
    total   = len(df)
    found   = int(df["file_exists"].sum())
    missing = total - found

    print(f"{split_name} — total entries : {total:>7,}")
    print(f"{split_name} — files FOUND   : {found:>7,}")
    print(f"{split_name} — files MISSING : {missing:>7,}")

    # Show a sample of the missing paths so the user can investigate
    missing_paths = df.loc[~df["file_exists"], "local_path"].tolist()
    if missing_paths:
        print(f"\n  First {n_samples} missing {split_name} paths:")
        for p in missing_paths[:n_samples]:
            print(f"    {p}")

    return {
        "total":         total,
        "found":         found,
        "missing":       missing,
        "missing_paths": [str(p) for p in missing_paths],
    }


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

    section("Path format inspection")
    train_fmt = inspect_path_formats(train_raw, "Train")
    print()
    test_fmt  = inspect_path_formats(test_raw,  "Test")

    section("Enriching DataFrames (resolving paths, checking existence, extracting labels)")
    print("(Stat-checking ~87,000 file paths — takes a few seconds…)")
    train = enrich_dataframe(train_raw)
    test  = enrich_dataframe(test_raw)

    # ── File existence verification ────────────────────────────────────────────
    section("File existence verification")
    train_stats = verify_files(train, "Train")
    print()
    test_stats  = verify_files(test,  "Test")
