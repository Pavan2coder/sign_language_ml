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
# Add project root to sys.path so "from src.config import …" works from any cwd
_SCRIPT_DIR   = Path(__file__).resolve().parent   # …/sign-lan/src
_PROJECT_ROOT = _SCRIPT_DIR.parent                # …/sign-lan
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,       # D:\ISL-DATA\MappingFiles\PersonDependentTrain.csv
    TEST_CSV,        # D:\ISL-DATA\MappingFiles\PersonDependentTest.csv
    MEDIAPIPE_ROOT,  # D:\ISL-DATA\Landmarks\MediaPipe
    REPORT_DIR,      # sign-lan/reports
)


# ── Utility ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a visible section header to the console."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


if __name__ == "__main__":
    section("ISL Dataset Analysis — STEP 1")
    print("Imports OK. Config loaded.")
    print("MediaPipe root :", MEDIAPIPE_ROOT)
    print("Train CSV      :", TRAIN_CSV)
    print("Test  CSV      :", TEST_CSV)
