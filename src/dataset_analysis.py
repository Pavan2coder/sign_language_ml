"""
dataset_analysis.py
===================
STEP 1 of the ISL Sign Language Recognition ML Pipeline:
Dataset and Label Verification.

What this script does
---------------------
1.  Loads PersonDependentTrain.csv and PersonDependentTest.csv.
2.  Prints shapes, column names, and first 5 rows of each.
3.  Classifies every CSV row into one of three categories:
      Format A  - normal filename    e.g. Absent__session82__clip000.h5
      Format B  - mangled _mnt_ or R2-prefixed name that contains _userNNN_
                  e.g. _mnt_..._user001_Absent__session65__clip028.h5
                       R2_28.03_..._user002_Accept__session65__clip002.h5
      Exceptional - no ISL_DATA_USERxxx in path at all (fold_1/3/4 artefacts)
4.  Resolves every row to a local Windows .h5 path:
        USER DIR  -> regex (ISL_DATA_USER[0-9]+) on the full CSV path string
        DISK FILE -> verbatim last segment of the CSV path
                     (files are stored on disk with their exact CSV filename)
        LABEL     -> extracted from the REAL underlying name (see Step 5)
5.  Extracts the gesture label from the real filename using two stages:
      Stage 1 - if the disk filename contains _userNNN_, find the LAST
                such token and take everything after it as the "tail".
                This handles both _mnt_ mangled names and R2-prefixed names
                that embed _userNNN_ one or more times.
                If no _userNNN_ is present, the filename itself is the tail.
      Stage 2 - from the tail, extract the label:
                - split on __session (most reliable boundary)
                - if the left part still contains underscores (e.g. R2_Absent),
                  take only the last underscore-separated token.
6.  Stat-checks every resolved path (no H5 data loaded).
7.  Filters out junk/header rows (fold_1, fold_3, fold_4) before statistics.
8.  Reports found/missing/exceptional counts, class statistics, and integrity.
9.  Saves the full report to reports/dataset_analysis.txt.

KEY FIX SUMMARY
---------------
Problem 1 (0 missing reported as 9,416 missing):
  The old resolver stripped the _mnt_ prefix and looked for the clean name
  on disk. But local files ARE stored with the full mangled name verbatim.
  Fix: use disk_filename = last CSV segment as-is for disk lookup.

Problem 2 (4,400 garbled classes instead of ~500):
  extract_label() split on __session but didn't handle filenames like
  R2_28.03_..._user002_Accept__session... which produced a garbage prefix.
  Fix: use the LAST _userNNN_ token as the extraction anchor, then clean
  any remaining prefix by taking the last underscore-separated token.

Problem 3 (SyntaxWarning on backslash sequences in docstring):
  Windows paths in plain docstrings triggered SyntaxWarning.
  Fix: use raw strings (r"...") or rewrite paths without backslashes.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── Project config ────────────────────────────────────────────────────────────
# Add the project root to sys.path so "from src.config import ..." works
# regardless of which directory you launch the script from.
_SCRIPT_DIR   = Path(__file__).resolve().parent   # .../sign-lan/src
_PROJECT_ROOT = _SCRIPT_DIR.parent                # .../sign-lan
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,       # D:\ISL-DATA\MappingFiles\PersonDependentTrain.csv
    TEST_CSV,        # D:\ISL-DATA\MappingFiles\PersonDependentTest.csv
    MEDIAPIPE_ROOT,  # D:\ISL-DATA\Landmarks\MediaPipe
    REPORT_DIR,      # sign-lan/reports
)


# ── Compiled regex patterns ───────────────────────────────────────────────────

# Extracts the ISL_DATA_USERxxx directory name from anywhere in a path string.
# This is used on the FULL CSV path (not just the filename) so it always finds
# the user directory even when the filename itself doesn't start with one.
_USER_DIR_RE = re.compile(r"(ISL_DATA_USER\d+)", re.IGNORECASE)

# Finds the LAST occurrence of _userNNN_ in a filename and captures everything
# after it as the "clean tail" used for label extraction.
#
# Using ".*" (greedy) before "_user\d+_" forces the match to the last token.
#
# Examples:
#   _mnt_9a528_ALL_CLIPS_R2_Clips_R2_user001_Absent__session65__clip028.h5
#     -> tail = "Absent__session65__clip028.h5"
#
#   R2_28.03.26_All_user002_R2_clips_clips_R2_user002_Accept__session65.h5
#     -> tail = "Accept__session65.h5"
#
#   R2_28.03_All_user007_R2_clips_Clips_user007_R2_Absent__session34.h5
#     -> tail = "R2_Absent__session34.h5"   (one more cleanup step needed)
_LAST_USER_TOKEN_RE = re.compile(r".*_user\d+_(.+\.h5)$", re.IGNORECASE)

# Header artefact values — these are fold column names that leaked into
# the data rows when per-fold CSVs were concatenated without stripping headers.
_JUNK_VALUES = frozenset({"fold_1", "fold_3", "fold_4"})


# ── Utility ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a visible section header to the console."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── STEP 1: Load the mapping CSVs ─────────────────────────────────────────────

def load_mapping_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a mapping CSV and rename its single column to 'raw_path'.

    Both CSVs have exactly one column ('fold_0' or 'fold_2').
    We rename it so the rest of the code uses a stable key.

    Stores original_col and source_file in df.attrs.
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


# ── STEP 2: Label extraction from a filename ─────────────────────────────────

def extract_label(disk_filename: str) -> str:
    """
    Derive the ISL gesture label from the on-disk filename.

    Two-stage approach:

    Stage 1 — find the clean underlying filename
        If the disk filename contains '_userNNN_', apply _LAST_USER_TOKEN_RE
        to capture everything after the LAST such token. This correctly handles:
            - plain mangled names:  _mnt_..._user001_Absent__session65__clip028.h5
            - R2-prefixed names:    R2_28.03_..._user002_Accept__session65__clip002.h5
            - USER007 variant:      R2_28.03_..._user007_R2_Absent__session34__clip019.h5
        For normal filenames (no _userNNN_), the filename itself is used directly.

    Stage 2 — extract the label word from the clean tail
        Split on '__session' (the most unambiguous boundary in this dataset).
        If the left part still contains underscores (e.g. 'R2_Absent'), the
        actual label is the last token after the last underscore.

    All filename conventions handled:
        Absent__session82__clip000.h5              -> "Absent"
        BalloonBlue__session12__clip003.h5         -> "BalloonBlue"
        Beach__000001.h5                           -> "Beach"
        Actor__000001__c545daf6-s004-003_Actor.h5  -> "Actor"
        _mnt_..._user001_Absent__session65.h5      -> "Absent"
        R2_..._user002_Accept__session65.h5        -> "Accept"
        R2_..._user007_R2_Absent__session34.h5     -> "Absent"
        R2_..._user005_R2_I__session67.h5          -> "I"
    """
    # ── Stage 1: unwrap any _userNNN_ prefix to get the clean tail ────────────
    m = _LAST_USER_TOKEN_RE.search(disk_filename)
    tail = m.group(1) if m else disk_filename

    # ── Stage 2: extract the label word ──────────────────────────────────────
    stem = Path(tail).stem   # drop ".h5"

    if "__session" in stem:
        before_session = stem.split("__session")[0]
    elif "__" in stem:
        before_session = stem.split("__")[0]
    else:
        before_session = stem

    # If before_session still contains underscores, the real label is the
    # last token (handles "R2_Absent" -> "Absent", "R2_I" -> "I", etc.)
    if "_" in before_session:
        return before_session.rsplit("_", 1)[-1]

    return before_session


# ── STEP 3: Per-row path resolution ──────────────────────────────────────────

def resolve_row(raw_path: str) -> dict:
    """
    Convert one raw CSV path into a structured resolution result.

    Returns a dict:
        status        - "normal" | "mangled" | "exceptional"
        user_dir      - e.g. "ISL_DATA_USER001"  (None if exceptional)
        disk_filename - verbatim last segment of the CSV path;
                        this is the ACTUAL filename stored on disk
        local_path    - MEDIAPIPE_ROOT / user_dir / disk_filename  (or None)
        label         - gesture class string  (or None if exceptional)

    Resolution rules:
        A. Search for ISL_DATA_USERxxx in the FULL path string (not just
           the filename) so the user directory is found reliably even when
           the filename itself starts with _mnt_ or R2_.
        B. The disk filename is the verbatim LAST segment of the CSV path.
           IMPORTANT: mangled and R2-prefixed files are stored on disk with
           their full original names. We must NOT strip or alter them for
           the disk lookup.
        C. For exceptional rows (fold_1/3/4) that contain no ISL_DATA_USERxxx,
           return status='exceptional' and all path fields as None.
    """
    raw = raw_path.strip()

    # ── A: locate the user directory ─────────────────────────────────────────
    user_match = _USER_DIR_RE.search(raw)
    if not user_match:
        return {
            "status":        "exceptional",
            "user_dir":       None,
            "disk_filename":  None,
            "local_path":     None,
            "label":          None,
        }

    user_dir = user_match.group(1)              # "ISL_DATA_USER001"

    # ── B: disk filename = verbatim last path segment ─────────────────────────
    disk_filename = raw.split("/")[-1]           # works for POSIX-style CSV paths

    # ── C: classify and build local path ──────────────────────────────────────
    if "_mnt_" in disk_filename or _LAST_USER_TOKEN_RE.search(disk_filename):
        status = "mangled"
    else:
        status = "normal"

    local_path = MEDIAPIPE_ROOT / user_dir / disk_filename
    label      = extract_label(disk_filename)

    return {
        "status":        status,
        "user_dir":       user_dir,
        "disk_filename":  disk_filename,
        "local_path":     local_path,
        "label":          label,
    }


# ── STEP 4: Enrich the full DataFrame ────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply resolve_row() to every row and add computed columns.

    Columns added:
        status        - "normal" | "mangled" | "exceptional"
        user_dir      - ISL_DATA_USERxxx string  (None for exceptional)
        disk_filename - verbatim on-disk filename
        local_path    - resolved Windows Path  (None for exceptional)
        file_exists   - True if the file is present on disk
        label         - gesture class  (None for exceptional)

    No .h5 file contents are opened — only Path.exists() stat calls.
    """
    df = df.copy()

    resolved = df["raw_path"].apply(resolve_row)

    df["status"]        = resolved.apply(lambda r: r["status"])
    df["user_dir"]      = resolved.apply(lambda r: r["user_dir"])
    df["disk_filename"] = resolved.apply(lambda r: r["disk_filename"])
    df["local_path"]    = resolved.apply(lambda r: r["local_path"])
    df["label"]         = resolved.apply(lambda r: r["label"])

    # Path.exists() on None returns an error, so guard with a conditional
    df["file_exists"] = df["local_path"].apply(
        lambda p: p.exists() if p is not None else False
    )

    return df


# ── STEP 5: Path format inspection ───────────────────────────────────────────

def inspect_path_formats(df: pd.DataFrame, split_name: str) -> dict:
    """
    Count and print how many rows fall into each format category.

    Returns dict with keys: normal, mangled, exceptional
    """
    counts        = df["status"].value_counts()
    n_normal      = int(counts.get("normal",      0))
    n_mangled     = int(counts.get("mangled",     0))
    n_exceptional = int(counts.get("exceptional", 0))

    print(f"{split_name} — Format A (normal filename)      : {n_normal:>7,}")
    print(f"{split_name} — Format B (mangled / R2-prefixed): {n_mangled:>7,}")
    print(f"{split_name} — Exceptional (no USER ID in path): {n_exceptional:>7,}")

    if n_exceptional:
        exc_vals = df.loc[df["status"] == "exceptional", "raw_path"].tolist()
        print(f"\n  [INFO] {n_exceptional} exceptional row(s) — CSV header artefacts, NOT real file paths.")
        print("  These are excluded from all file-existence and class statistics.")
        for v in exc_vals:
            print(f"    value: {v!r}")

    return {"normal": n_normal, "mangled": n_mangled, "exceptional": n_exceptional}


# ── STEP 6: File existence verification ──────────────────────────────────────

def verify_files(df: pd.DataFrame, split_name: str, n_samples: int = 5) -> dict:
    """
    Report found / missing / exceptional counts.

    Exceptional rows are excluded from the found/missing count because
    they are header artefacts rather than real file references.

    Returns dict: normalizable, found, missing, exceptional, missing_paths
    """
    exc_mask     = df["status"] == "exceptional"
    norm_df      = df[~exc_mask]
    normalizable = len(norm_df)
    found        = int(norm_df["file_exists"].sum())
    missing      = normalizable - found
    exceptional  = int(exc_mask.sum())

    print(f"{split_name} — normalizable entries : {normalizable:>7,}")
    print(f"{split_name} — files FOUND          : {found:>7,}")
    print(f"{split_name} — files MISSING        : {missing:>7,}")
    print(f"{split_name} — exceptional rows     : {exceptional:>7,}  (excluded from above)")

    missing_paths = norm_df.loc[~norm_df["file_exists"], "local_path"].tolist()
    if missing_paths:
        print(f"\n  First {n_samples} missing {split_name} paths:")
        for p in missing_paths[:n_samples]:
            print(f"    {p}")

    return {
        "normalizable":  normalizable,
        "found":         found,
        "missing":       missing,
        "exceptional":   exceptional,
        "missing_paths": [str(p) for p in missing_paths],
    }


# ── STEP 7: Class / label statistics ─────────────────────────────────────────

def class_statistics(train: pd.DataFrame, test: pd.DataFrame,
                     top_n: int = 20) -> dict:
    """
    Compute class distribution statistics after filtering out:
        - Exceptional rows  (label is None)
        - Junk header values (fold_1, fold_3, fold_4)

    Returns dict with all metrics and label count Series objects.

    Why these numbers matter for ML:
        n_unique_all  -> number of output neurons needed in the classifier
        train_min     -> classes with very few samples may need augmentation
        class imbalance (large gap between min and max) can bias the model
    """
    def clean_labels(df_in: pd.DataFrame) -> pd.Series:
        """Return label Series with exceptional and junk rows removed."""
        mask = (
            df_in["label"].notna() &
            ~df_in["label"].isin(_JUNK_VALUES)
        )
        return df_in.loc[mask, "label"]

    train_labels = clean_labels(train)
    test_labels  = clean_labels(test)

    train_counts = train_labels.value_counts()
    test_counts  = test_labels.value_counts()

    n_unique_train = int(train_labels.nunique())
    n_unique_test  = int(test_labels.nunique())

    all_labels   = pd.concat([train_labels, test_labels])
    all_counts   = all_labels.value_counts()
    n_unique_all = int(all_counts.shape[0])

    train_min  = int(train_counts.min())
    train_max  = int(train_counts.max())
    train_mean = float(train_counts.mean())
    test_min   = int(test_counts.min())
    test_max   = int(test_counts.max())
    test_mean  = float(test_counts.mean())

    print(f"Unique classes in Train  : {n_unique_train:,}")
    print(f"Unique classes in Test   : {n_unique_test:,}")
    print(f"Overall unique classes   : {n_unique_all:,}")
    print()
    print(f"Train — min samples/class : {train_min}")
    print(f"Train — max samples/class : {train_max}")
    print(f"Train — mean samples/class: {train_mean:.1f}")
    print()
    print(f"Test  — min samples/class : {test_min}")
    print(f"Test  — max samples/class : {test_max}")
    print(f"Test  — mean samples/class: {test_mean:.1f}")
    print()
    print(f"Top {top_n} classes by TRAIN sample count:")
    print(train_counts.head(top_n).to_string())

    return {
        "n_unique_train": n_unique_train,
        "n_unique_test":  n_unique_test,
        "n_unique_all":   n_unique_all,
        "train_min":      train_min,
        "train_max":      train_max,
        "train_mean":     train_mean,
        "test_min":       test_min,
        "test_max":       test_max,
        "test_mean":      test_mean,
        "train_counts":   train_counts,
        "test_counts":    test_counts,
        "all_counts":     all_counts,
        "top_n_str":      train_counts.head(top_n).to_string(),
    }


# ── STEP 8: Train vs Test class comparison ───────────────────────────────────

def compare_class_sets(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Find classes present in only one split.

    Classes in TEST only are important — the model will be evaluated on
    gestures it has never seen during training, leading to poor accuracy.
    Junk/exceptional rows are excluded before comparison.
    """
    def clean_set(df_in):
        mask = df_in["label"].notna() & ~df_in["label"].isin(_JUNK_VALUES)
        return set(df_in.loc[mask, "label"].unique())

    train_classes = clean_set(train)
    test_classes  = clean_set(test)

    train_only = sorted(train_classes - test_classes)
    test_only  = sorted(test_classes  - train_classes)
    common     = sorted(train_classes & test_classes)

    print(f"Classes in BOTH splits   : {len(common):>5,}")
    print(f"Classes in TRAIN only    : {len(train_only):>5,}")
    print(f"Classes in TEST only     : {len(test_only):>5,}")

    if train_only:
        preview = ", ".join(train_only[:10])
        suffix  = f" ... and {len(train_only) - 10} more" if len(train_only) > 10 else ""
        print(f"\n  Train-only classes (first 10): {preview}{suffix}")

    if test_only:
        preview = ", ".join(test_only[:10])
        suffix  = f" ... and {len(test_only) - 10} more" if len(test_only) > 10 else ""
        print(f"\n  [WARNING] Test-only classes (first 10): {preview}{suffix}")
        print("  The model cannot learn these during training.")
    else:
        print("\n  No test-only classes — every test class appears in train. Good.")

    return {
        "train_only":   train_only,
        "test_only":    test_only,
        "common":       common,
        "n_train_only": len(train_only),
        "n_test_only":  len(test_only),
        "n_common":     len(common),
    }


# ── STEP 9: Integrity checks ──────────────────────────────────────────────────

def check_integrity(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Three checks:
        1. Duplicate raw_path rows within each split.
        2. Header-artefact rows (fold_1, fold_3, fold_4).
        3. Train/test path overlap — same file in both splits (data leakage).
    """
    # 1. Duplicates
    train_dupes = int(train["raw_path"].duplicated().sum())
    test_dupes  = int(test["raw_path"].duplicated().sum())
    print(f"Duplicate rows in Train CSV : {train_dupes}")
    print(f"Duplicate rows in Test  CSV : {test_dupes}")

    # 2. Header artefacts — rows whose raw_path is a fold_ value
    train_junk_mask = train["raw_path"].isin(_JUNK_VALUES)
    test_junk_mask  = test["raw_path"].isin(_JUNK_VALUES)
    train_junk = int(train_junk_mask.sum())
    test_junk  = int(test_junk_mask.sum())
    junk_vals  = train.loc[train_junk_mask, "raw_path"].tolist()
    junk_vals += test.loc[test_junk_mask,   "raw_path"].tolist()

    print(f"\nHeader-artefact rows in Train : {train_junk}")
    if train_junk:
        print(f"  Values: {train.loc[train_junk_mask, 'raw_path'].tolist()}")
        print("  (These are fold-header values embedded as data rows — not file paths.)")
    print(f"Header-artefact rows in Test  : {test_junk}")

    # 3. Path overlap (data leakage)
    train_paths = {str(p) for p in train["local_path"] if p is not None}
    test_paths  = {str(p) for p in test["local_path"]  if p is not None}
    overlap   = train_paths & test_paths
    n_overlap = len(overlap)

    print(f"\nPaths in BOTH splits (leakage) : {n_overlap}")
    if n_overlap:
        print("  [WARNING] Files appearing in both splits:")
        for p in sorted(overlap)[:5]:
            print(f"    {p}")
    else:
        print("  No train/test path overlap — splits are clean.")

    return {
        "train_dupes":  train_dupes,
        "test_dupes":   test_dupes,
        "train_junk":   train_junk,
        "test_junk":    test_junk,
        "junk_values":  junk_vals,
        "path_overlap": n_overlap,
    }


# ── STEP 10: Save report to disk ─────────────────────────────────────────────

def write_report(
    train_raw:   pd.DataFrame,
    test_raw:    pd.DataFrame,
    train_fmt:   dict,
    test_fmt:    dict,
    train_stats: dict,
    test_stats:  dict,
    label_stats: dict,
    class_comp:  dict,
    integrity:   dict,
) -> Path:
    """
    Write all analysis results to reports/dataset_analysis.txt.
    Returns the Path of the saved file.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dataset_analysis.txt"

    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def kv(key, value):
        lines.append(f"  {key:<44}: {value}")

    h("ISL Sign Language Dataset Analysis Report - STEP 1")
    lines.append(f"  Train CSV  : {TRAIN_CSV}")
    lines.append(f"  Test  CSV  : {TEST_CSV}")
    lines.append(f"  MediaPipe  : {MEDIAPIPE_ROOT}")
    lines.append(f"  Report dir : {REPORT_DIR}")

    h("CSV Metadata")
    kv("Train original column",  train_raw.attrs.get("original_col", "?"))
    kv("Test  original column",  test_raw.attrs.get("original_col", "?"))
    kv("Train shape",            str(train_raw.shape))
    kv("Test  shape",            str(test_raw.shape))

    h("First 5 rows of Train CSV")
    lines.append(train_raw.head(5).to_string(index=True))

    h("First 5 rows of Test CSV")
    lines.append(test_raw.head(5).to_string(index=True))

    h("Path Format Breakdown")
    for split, fmt in [("Train", train_fmt), ("Test", test_fmt)]:
        kv(f"{split} Format A (normal)",         f"{fmt['normal']:,}")
        kv(f"{split} Format B (mangled/R2)",      f"{fmt['mangled']:,}")
        kv(f"{split} Exceptional (no USER ID)",   f"{fmt['exceptional']:,}")

    h("File Existence Verification")
    for split, stats in [("Train", train_stats), ("Test", test_stats)]:
        kv(f"{split} normalizable entries",  f"{stats['normalizable']:,}")
        kv(f"{split} files FOUND",           f"{stats['found']:,}")
        kv(f"{split} files MISSING",         f"{stats['missing']:,}")
        kv(f"{split} exceptional rows",      f"{stats['exceptional']:,}")

    if train_stats["missing_paths"]:
        lines.append("")
        lines.append("  Sample missing TRAIN paths (first 5):")
        for p in train_stats["missing_paths"][:5]:
            lines.append(f"    {p}")

    if test_stats["missing_paths"]:
        lines.append("")
        lines.append("  Sample missing TEST paths (first 5):")
        for p in test_stats["missing_paths"][:5]:
            lines.append(f"    {p}")

    h("Class / Label Statistics")
    kv("Unique classes in Train",    label_stats["n_unique_train"])
    kv("Unique classes in Test",     label_stats["n_unique_test"])
    kv("Overall unique classes",     label_stats["n_unique_all"])
    lines.append("")
    kv("Train min  samples/class",   label_stats["train_min"])
    kv("Train max  samples/class",   label_stats["train_max"])
    kv("Train mean samples/class",   f"{label_stats['train_mean']:.1f}")
    kv("Test  min  samples/class",   label_stats["test_min"])
    kv("Test  max  samples/class",   label_stats["test_max"])
    kv("Test  mean samples/class",   f"{label_stats['test_mean']:.1f}")

    h("Top 20 Classes by Train Sample Count")
    lines.append(label_stats["top_n_str"])

    h("Full Label Distribution (combined train+test, sorted by count)")
    lines.append(f"  {'Label':<40} {'Combined':>8}")
    lines.append(f"  {'-'*40} {'-'*8}")
    for lbl, cnt in label_stats["all_counts"].items():
        lines.append(f"  {lbl:<40} {cnt:>8,}")

    h("Train vs Test Class Set Comparison")
    kv("Classes in BOTH splits",  class_comp["n_common"])
    kv("Classes in TRAIN only",   class_comp["n_train_only"])
    kv("Classes in TEST only",    class_comp["n_test_only"])

    if class_comp["train_only"]:
        lines.append("")
        lines.append("  Train-only classes:")
        lines.append("    " + ", ".join(class_comp["train_only"]))

    if class_comp["test_only"]:
        lines.append("")
        lines.append("  [WARNING] Test-only classes (model cannot learn these):")
        lines.append("    " + ", ".join(class_comp["test_only"]))

    h("Integrity Checks")
    kv("Duplicate rows in Train CSV",          integrity["train_dupes"])
    kv("Duplicate rows in Test  CSV",          integrity["test_dupes"])
    kv("Header-artefact rows in Train",        integrity["train_junk"])
    kv("Header-artefact rows in Test",         integrity["test_junk"])
    kv("Paths in BOTH splits (leakage)",       integrity["path_overlap"])

    if integrity["junk_values"]:
        lines.append("")
        lines.append("  Junk fold-header values found:")
        lines.append("    " + ", ".join(integrity["junk_values"]))

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_analysis() -> None:
    """
    Run the complete STEP 1 analysis end-to-end.
    Prints every result to the console and saves reports/dataset_analysis.txt.
    """
    section("ISL Dataset Analysis - STEP 1")
    print(f"  MediaPipe root : {MEDIAPIPE_ROOT}")
    print(f"  Train CSV      : {TRAIN_CSV}")
    print(f"  Test  CSV      : {TEST_CSV}")

    # 1. Load
    section("1 - Loading mapping CSVs")
    train_raw = load_mapping_csv(TRAIN_CSV)
    test_raw  = load_mapping_csv(TEST_CSV)
    print(f"  Train shape : {train_raw.shape}  |  column: {train_raw.attrs['original_col']}")
    print(f"  Test  shape : {test_raw.shape}  |  column: {test_raw.attrs['original_col']}")

    section("  First 5 rows of Train CSV")
    print(train_raw.head(5).to_string(index=True))
    section("  First 5 rows of Test CSV")
    print(test_raw.head(5).to_string(index=True))

    # 2. Enrich (resolve paths + extract labels — no H5 data loaded)
    section("2 - Resolving paths and extracting labels")
    print("  (Stat-checking ~87,000 file paths, please wait...)")
    train = enrich_dataframe(train_raw)
    test  = enrich_dataframe(test_raw)

    # 3. Path format breakdown
    section("3 - Path format counts")
    train_fmt = inspect_path_formats(train, "Train")
    print()
    test_fmt  = inspect_path_formats(test,  "Test")

    # 4. File existence
    section("4 - File existence verification")
    train_stats = verify_files(train, "Train")
    print()
    test_stats  = verify_files(test,  "Test")

    # 5. Class statistics (junk rows excluded)
    section("5 - Class / label statistics  [junk rows excluded]")
    label_stats = class_statistics(train, test)

    # 6. Train vs test class sets
    section("6 - Train vs Test class set comparison")
    class_comp = compare_class_sets(train, test)

    # 7. Integrity
    section("7 - Integrity checks")
    integrity = check_integrity(train, test)

    # 8. Save report
    section("8 - Saving report")
    report_path = write_report(
        train_raw   = train_raw,
        test_raw    = test_raw,
        train_fmt   = train_fmt,
        test_fmt    = test_fmt,
        train_stats = train_stats,
        test_stats  = test_stats,
        label_stats = label_stats,
        class_comp  = class_comp,
        integrity   = integrity,
    )
    print(f"  [OK] Report saved to: {report_path}")

    # 9. Final summary
    section("SUMMARY")
    print(f"  Train : {train_stats['found']:,} / {train_stats['normalizable']:,} files found"
          f"  |  {train_stats['missing']:,} missing"
          f"  |  {train_stats['exceptional']:,} exceptional (header artefacts)")
    print(f"  Test  : {test_stats['found']:,} / {test_stats['normalizable']:,} files found"
          f"  |  {test_stats['missing']:,} missing"
          f"  |  {test_stats['exceptional']:,} exceptional")
    print(f"  Unique classes (cleaned) : {label_stats['n_unique_all']:,}")
    print(f"  No data leakage          : {integrity['path_overlap'] == 0}")


if __name__ == "__main__":
    run_analysis()
