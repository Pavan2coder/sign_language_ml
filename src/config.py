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