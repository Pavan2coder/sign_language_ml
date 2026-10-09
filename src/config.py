from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset is stored outside the Git repository
DATASET_ROOT = Path(r"D:\ISL-DATA")

# MediaPipe landmark data
MEDIAPIPE_ROOT = DATASET_ROOT / "Landmarks" / "MediaPipe"

# Train/Test mapping files
MAPPING_ROOT = DATASET_ROOT / "MappingFiles"

TRAIN_CSV = MAPPING_ROOT / "PersonDependentTrain.csv"
TEST_CSV = MAPPING_ROOT / "PersonDependentTest.csv"


# ============================================================
# PROJECT OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# HDF5 CONFIGURATION
# ============================================================

H5_DATASET_NAME = "intermediate"


# ============================================================
# LANDMARK CONFIGURATION
# ============================================================

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES = 3

FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDINATES

# Expected shape of one H5 landmark array, ignoring the frame axis.
# Full array shape is (frames, 2, 21, 3) — frames vary per clip.
# This tuple is used by data_loader.py to validate loaded arrays.
EXPECTED_LANDMARK_SHAPE = (NUM_HANDS, LANDMARKS_PER_HAND, COORDINATES)


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

# Fixed sequence length used by the preprocessor.
#
# Real frame counts in the dataset (measured on 200 samples):
#   min=43, max=357, mean=136, median=130, p75=159, p90=182, p95=206, p99=268
#
# 150 is chosen because it:
#   - sits above the median (130), so most clips need padding not truncation
#   - truncates only the longest ~25 % of clips
#   - keeps the input tensor small enough for efficient training
#   - is easy to increase if a model needs more temporal context
#
# Change this value to experiment; the rest of the pipeline adapts automatically.
MAX_SEQ_LEN: int = 150

# Feature dimension per frame: flatten (2, 21, 3) -> 126 values.
# This equals FEATURES_PER_FRAME and is here for explicit documentation.
FEATURE_DIM: int = FEATURES_PER_FRAME  # 126

# Number of sign classes discovered during dataset verification (Step 1).
NUM_CLASSES: int = 500
