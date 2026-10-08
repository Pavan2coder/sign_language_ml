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

    Returns dict with keys: total, found, missing, missing_paths
    """
    total   = len(df)
    found   = int(df["file_exists"].sum())
    missing = total - found

    print(f"{split_name} — total entries : {total:>7,}")
    print(f"{split_name} — files FOUND   : {found:>7,}")
    print(f"{split_name} — files MISSING : {missing:>7,}")

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


# ── STEP 7: Class / label statistics ──────────────────────────────────────────

def class_statistics(train: pd.DataFrame, test: pd.DataFrame, top_n: int = 20) -> dict:
    """
    Compute and print class distribution statistics for both splits.

    Returns dict with all metrics and the full label count Series objects.
    """
    train_counts = train["label"].value_counts()
    test_counts  = test["label"].value_counts()

    n_unique_train = int(train["label"].nunique())
    n_unique_test  = int(test["label"].nunique())

    all_labels   = pd.concat([train["label"], test["label"]])
    all_counts   = all_labels.value_counts()
    n_unique_all = int(all_counts.shape[0])

    train_min  = int(train_counts.min())
    train_max  = int(train_counts.max())
    train_mean = float(train_counts.mean())
    test_min   = int(test_counts.min())
    test_max   = int(test_counts.max())
    test_mean  = float(test_counts.mean())

    print(f"Unique classes in Train  : {n_unique_train}")
    print(f"Unique classes in Test   : {n_unique_test}")
    print(f"Overall unique classes   : {n_unique_all}")
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


# ── STEP 8: Train vs Test class set comparison ────────────────────────────────

def compare_class_sets(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Compare which classes appear in train, test, or both.

    Returns dict with train_only, test_only, common lists and counts.
    """
    train_classes = set(train["label"].unique())
    test_classes  = set(test["label"].unique())

    train_only = sorted(train_classes - test_classes)
    test_only  = sorted(test_classes  - train_classes)
    common     = sorted(train_classes & test_classes)

    print(f"Classes in BOTH splits   : {len(common):>5}")
    print(f"Classes in TRAIN only    : {len(train_only):>5}")
    print(f"Classes in TEST only     : {len(test_only):>5}")

    if train_only:
        preview = ", ".join(train_only[:10])
        suffix  = f" ... and {len(train_only) - 10} more" if len(train_only) > 10 else ""
        print(f"\n  Train-only labels (first 10): {preview}{suffix}")

    if test_only:
        preview = ", ".join(test_only[:10])
        suffix  = f" ... and {len(test_only) - 10} more" if len(test_only) > 10 else ""
        print(f"\n  [WARNING] Test-only labels (first 10): {preview}{suffix}")
        print("  These classes appear in TEST but not TRAIN — model will never")
        print("  have learned them, leading to poor evaluation accuracy.")
    else:
        print("\n  No test-only classes — every test class also appears in train. Good.")

    return {
        "train_only":   train_only,
        "test_only":    test_only,
        "common":       common,
        "n_train_only": len(train_only),
        "n_test_only":  len(test_only),
        "n_common":     len(common),
    }


# ── STEP 9: Duplicate and junk row detection ──────────────────────────────────

def check_duplicates_and_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Three integrity checks: duplicate rows, junk fold-header artefacts,
    and train/test path overlap (data leakage).

    Returns dict with counts and lists of offending values.
    """
    train_dupes = int(train["raw_path"].duplicated().sum())
    test_dupes  = int(test["raw_path"].duplicated().sum())

    print(f"Duplicate rows in Train CSV : {train_dupes}")
    print(f"Duplicate rows in Test  CSV : {test_dupes}")

    train_junk_mask = train["raw_path"].str.startswith("fold_", na=False)
    test_junk_mask  = test["raw_path"].str.startswith("fold_", na=False)

    train_junk = int(train_junk_mask.sum())
    test_junk  = int(test_junk_mask.sum())
    junk_vals  = train.loc[train_junk_mask, "raw_path"].tolist()
    junk_vals += test.loc[test_junk_mask,   "raw_path"].tolist()

    print(f"\nEmbedded fold-header rows in Train : {train_junk}")
    if train_junk:
        print(f"  Values: {train.loc[train_junk_mask, 'raw_path'].tolist()}")
        print("  (These are CSV header artefacts, not file paths.)")
    print(f"Embedded fold-header rows in Test  : {test_junk}")

    train_paths = set(train["local_path"].astype(str))
    test_paths  = set(test["local_path"].astype(str))
    overlap     = train_paths & test_paths
    n_overlap   = len(overlap)

    print(f"\nPaths in BOTH splits (leakage) : {n_overlap}")
    if n_overlap:
        print("  [WARNING] These files appear in both train AND test:")
        for p in sorted(overlap)[:5]:
            print(f"    {p}")
    else:
        print("  No train/test path overlap — splits are clean.")

    return {
        "train_dupes":       train_dupes,
        "test_dupes":        test_dupes,
        "train_junk":        train_junk,
        "test_junk":         test_junk,
        "junk_values_train": junk_vals,
        "path_overlap":      n_overlap,
    }


# ── STEP 10: Write the full analysis report to disk ───────────────────────────

def write_report(
    train_raw:    pd.DataFrame,
    test_raw:     pd.DataFrame,
    train_fmt:    dict,
    test_fmt:     dict,
    train_stats:  dict,
    test_stats:   dict,
    label_stats:  dict,
    class_comp:   dict,
    integrity:    dict,
) -> Path:
    """
    Assemble all analysis results into a single human-readable text file
    and save it to REPORT_DIR/dataset_analysis.txt.

    Args:
        All dicts returned by the analysis functions above.

    Returns:
        Path to the saved report file.

    Why a text report?
    ------------------
    A plain-text file is easy to open anywhere, version-control friendly
    (git tracks changes line-by-line), and can be read without Python.
    It serves as a permanent record of what the dataset looked like before
    any preprocessing or training.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dataset_analysis.txt"

    lines = []

    # ── helpers ───────────────────────────────────────────────────────────────
    def h(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def kv(key: str, value) -> None:
        lines.append(f"  {key:<42}: {value}")

    # ── header ────────────────────────────────────────────────────────────────
    h("ISL Sign Language Dataset Analysis Report — STEP 1")
    lines.append(f"  Train CSV    : {TRAIN_CSV}")
    lines.append(f"  Test  CSV    : {TEST_CSV}")
    lines.append(f"  MediaPipe    : {MEDIAPIPE_ROOT}")
    lines.append(f"  Report dir   : {REPORT_DIR}")

    # ── CSV metadata ──────────────────────────────────────────────────────────
    h("CSV Metadata")
    kv("Train original column",  train_raw.attrs.get("original_col", "?"))
    kv("Test  original column",  test_raw.attrs.get("original_col", "?"))
    kv("Train shape",            str(train_raw.shape))
    kv("Test  shape",            str(test_raw.shape))

    # ── first rows ────────────────────────────────────────────────────────────
    h("First 5 rows of Train CSV")
    lines.append(train_raw.head(5).to_string(index=True))

    h("First 5 rows of Test CSV")
    lines.append(test_raw.head(5).to_string(index=True))

    # ── path format breakdown ─────────────────────────────────────────────────
    h("Path Format Breakdown")
    for split_name, fmt in [("Train", train_fmt), ("Test", test_fmt)]:
        kv(f"{split_name} Format A (normal)",        f"{fmt['normal']:,}")
        kv(f"{split_name} Format B (_mnt_ mangled)", f"{fmt['mnt_mangled']:,}")
        kv(f"{split_name} Format C (R2-prefixed)",   f"{fmt['r2_prefixed']:,}")
        kv(f"{split_name} Junk / unparseable rows",  f"{fmt['junk']:,}")

    # ── file existence ────────────────────────────────────────────────────────
    h("File Existence Verification")
    for split_name, stats in [("Train", train_stats), ("Test", test_stats)]:
        kv(f"{split_name} total entries",  f"{stats['total']:,}")
        kv(f"{split_name} files FOUND",    f"{stats['found']:,}")
        kv(f"{split_name} files MISSING",  f"{stats['missing']:,}")

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

    # ── class statistics ──────────────────────────────────────────────────────
    h("Class / Label Statistics")
    kv("Unique classes in Train",   label_stats["n_unique_train"])
    kv("Unique classes in Test",    label_stats["n_unique_test"])
    kv("Overall unique classes",    label_stats["n_unique_all"])
    lines.append("")
    kv("Train min  samples/class",  label_stats["train_min"])
    kv("Train max  samples/class",  label_stats["train_max"])
    kv("Train mean samples/class",  f"{label_stats['train_mean']:.1f}")
    kv("Test  min  samples/class",  label_stats["test_min"])
    kv("Test  max  samples/class",  label_stats["test_max"])
    kv("Test  mean samples/class",  f"{label_stats['test_mean']:.1f}")

    h("Top 20 Classes by Train Sample Count")
    lines.append(label_stats["top_n_str"])

    h("Full Label Distribution (all classes, combined train+test, sorted by count)")
    lines.append(f"  {'Label':<40} {'Combined':>8}")
    lines.append(f"  {'-'*40} {'-'*8}")
    for label, cnt in label_stats["all_counts"].items():
        lines.append(f"  {label:<40} {cnt:>8,}")

    # ── class set comparison ──────────────────────────────────────────────────
    h("Train vs Test Class Set Comparison")
    kv("Classes in BOTH splits",   class_comp["n_common"])
    kv("Classes in TRAIN only",    class_comp["n_train_only"])
    kv("Classes in TEST only",     class_comp["n_test_only"])

    if class_comp["train_only"]:
        lines.append("")
        lines.append("  Train-only labels:")
        lines.append("    " + ", ".join(class_comp["train_only"]))

    if class_comp["test_only"]:
        lines.append("")
        lines.append("  [WARNING] Test-only labels (model will not have seen these):")
        lines.append("    " + ", ".join(class_comp["test_only"]))

    # ── integrity checks ──────────────────────────────────────────────────────
    h("Integrity Checks (Duplicates, Junk Rows, Data Leakage)")
    kv("Duplicate rows in Train CSV",        integrity["train_dupes"])
    kv("Duplicate rows in Test  CSV",        integrity["test_dupes"])
    kv("Embedded fold-header rows in Train", integrity["train_junk"])
    kv("Embedded fold-header rows in Test",  integrity["test_junk"])
    kv("Paths in BOTH splits (leakage)",     integrity["path_overlap"])

    if integrity["junk_values_train"]:
        lines.append("")
        lines.append("  Junk fold-header values found in Train:")
        lines.append("    " + ", ".join(integrity["junk_values_train"]))

    # ── footer ────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Main: wire all steps together ─────────────────────────────────────────────

def run_analysis() -> None:
    """
    Run the complete STEP 1 dataset analysis end-to-end.

    Calls every analysis function in order, prints results to the console,
    then saves the full report to reports/dataset_analysis.txt.
    """
    section("ISL Dataset Analysis — STEP 1")
    print("MediaPipe root :", MEDIAPIPE_ROOT)
    print("Train CSV      :", TRAIN_CSV)
    print("Test  CSV      :", TEST_CSV)

    # 1. Load CSVs
    section("Loading mapping CSVs")
    train_raw = load_mapping_csv(TRAIN_CSV)
    test_raw  = load_mapping_csv(TEST_CSV)
    print(f"Train  shape : {train_raw.shape}  |  column: {train_raw.attrs['original_col']}")
    print(f"Test   shape : {test_raw.shape}   |  column: {test_raw.attrs['original_col']}")

    section("First 5 rows of Train CSV")
    print(train_raw.head(5).to_string(index=True))

    section("First 5 rows of Test CSV")
    print(test_raw.head(5).to_string(index=True))

    # 2. Path format inspection
    section("Path format inspection")
    train_fmt = inspect_path_formats(train_raw, "Train")
    print()
    test_fmt  = inspect_path_formats(test_raw,  "Test")

    # 3. Enrich (resolve paths, stat-check existence, extract labels)
    section("Enriching DataFrames (resolving paths, checking existence, extracting labels)")
    print("(Stat-checking ~87,000 file paths — takes a few seconds...)")
    train = enrich_dataframe(train_raw)
    test  = enrich_dataframe(test_raw)

    # 4. File existence report
    section("File existence verification")
    train_stats = verify_files(train, "Train")
    print()
    test_stats  = verify_files(test,  "Test")

    # 5. Class statistics
    section("Class / label statistics")
    label_stats = class_statistics(train, test)

    # 6. Train vs test class set comparison
    section("Train vs Test class set comparison")
    class_comp = compare_class_sets(train, test)

    # 7. Integrity checks
    section("Duplicate and data-leakage checks")
    integrity = check_duplicates_and_leakage(train, test)

    # 8. Save report
    section("Saving analysis report")
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
    print(f"[OK] Report saved to: {report_path}")

    # 9. Summary
    section("Done — Summary")
    print(
        f"  Train : {train_stats['found']:,} / {train_stats['total']:,} files found"
        f"  ({train_stats['missing']:,} missing)"
    )
    print(
        f"  Test  : {test_stats['found']:,} / {test_stats['total']:,} files found"
        f"  ({test_stats['missing']:,} missing)"
    )
    print(f"  Total unique classes : {label_stats['n_unique_all']}")
    print(f"  No data leakage      : {integrity['path_overlap'] == 0}")


if __name__ == "__main__":
    run_analysis()
