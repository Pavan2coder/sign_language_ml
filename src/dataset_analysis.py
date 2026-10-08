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
    Derive the ISL gesture / class label from an .h5 filename.

    The dataset contains several filename conventions.  All of them encode
    the label somewhere in the stem (filename without extension).  Here is
    how each is handled:

    FORMAT A — Standard  (most files, ~77 % of the dataset)
        Pattern : <Label>__session<N>__clip<N>.h5
        Examples:
            Absent__session106__clip014.h5     → "Absent"
            BalloonBlue__session12__clip003.h5 → "BalloonBlue"
            Campus__session14__clip000_1.h5    → "Campus"
            Beach__000001.h5                   → "Beach"
        Strategy: split stem on first '__', take left part.

    FORMAT B — USER007 uuid style  (~3 %)
        Pattern : <Label>__<6digits>__<uuid>_<Label>.h5
        Example : Absent__000003__c545daf6-s255-023_Absent.h5 → "Absent"
        Strategy: same split — left of first '__' is already the label.

    FORMAT C — R2 date-prefixed  (~12 %)
        Pattern : R2_<date>_..._<Label>__session<N>__clip<N>.h5
        Example : R2_28.03.26_All_user002_R2_clips_clips_R2_user002_Accept__session65__clip002.h5
                  → "Accept"
        Strategy: the prefix before the first '__' contains 'R2_' or starts
                  with a digit.  We look for the last uppercase-starting word
                  in that prefix (handles both multi-char "Accept" and
                  single-char "I" labels).

    JUNK sentinel  (fold_1, fold_3, fold_4 header artefacts)
        These come from the __UNPARSEABLE__ sentinel directory.
        The filename is the raw junk value; we return it unchanged so the
        caller can detect and report it separately.
    """
    stem          = Path(filename).stem       # drop ".h5"
    first_segment = stem.split("__")[0]       # left of first double-underscore

    # FORMAT C: prefix starts with "R2_" or a digit
    if first_segment.startswith("R2_") or (first_segment and first_segment[0].isdigit()):
        # Last capitalised word in the prefix is the label
        # Accepts single-letter labels like "I" as well as multi-letter ones
        match = re.search(r"_([A-Z][a-zA-Z0-9]*)$", first_segment)
        if match:
            return match.group(1)
        # Fallback: couldn't parse — return whole first segment
        return first_segment

    # FORMAT A and FORMAT B: first segment IS the label already
    return first_segment


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

    # ── extract_label smoke-test ───────────────────────────────────────────────
    section("Label extraction smoke-test")
    test_cases = [
        ("Absent__session106__clip014.h5",                                          "Absent"),
        ("BalloonBlue__session12__clip003.h5",                                      "BalloonBlue"),
        ("Beach__000001.h5",                                                        "Beach"),
        ("Absent__000003__c545daf6-s255-023_Absent.h5",                             "Absent"),
        ("R2_28.03.26_All_user002_R2_clips_clips_R2_user002_Accept__session65__clip002.h5", "Accept"),
        ("R2_28.03_All_user009_R2_Clips_user009_R2_I__session161__clip020.h5",      "I"),
    ]
    all_ok = True
    for filename, expected in test_cases:
        got = extract_label(filename)
        status = "OK" if got == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}]  {filename}")
        print(f"         expected={expected!r}  got={got!r}")
    print()
    print("All label tests passed!" if all_ok else "Some label tests FAILED — check output above.")
