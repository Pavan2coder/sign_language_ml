from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(r"D:\ISL-DATA")
MEDIAPIPE_ROOT = DATASET_ROOT / "Landmarks" / "MediaPipe"
MAPPING_ROOT = DATASET_ROOT / "MappingFiles"
TRAIN_CSV = MAPPING_ROOT / "PersonDependentTrain.csv"
TEST_CSV  = MAPPING_ROOT / "PersonDependentTest.csv"


# ============================================================
# PROJECT OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR     = PROJECT_ROOT / "models"
REPORT_DIR    = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# HDF5 CONFIGURATION
# ============================================================

H5_DATASET_NAME = "intermediate"


# ============================================================
# LANDMARK CONFIGURATION
# ============================================================

NUM_HANDS          = 2
LANDMARKS_PER_HAND = 21
COORDINATES        = 3
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDINATES
EXPECTED_LANDMARK_SHAPE = (NUM_HANDS, LANDMARKS_PER_HAND, COORDINATES)


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

MAX_SEQ_LEN: int  = 150
FEATURE_DIM: int  = FEATURES_PER_FRAME   # 126
NUM_CLASSES: int  = 500


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_SEED:   int   = 42
VAL_FRACTION:  float = 0.15

DEFAULT_EPOCHS:       int   = 30
DEFAULT_BATCH_SIZE:   int   = 32
DEFAULT_LR:           float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
PATIENCE:             int   = 5

GRU_HIDDEN_SIZE: int   = 256
GRU_NUM_LAYERS:  int   = 2
GRU_DROPOUT:     float = 0.3
FC_HIDDEN_SIZE:  int   = 256


# ============================================================
# CHECKPOINT CONFIGURATION
# ============================================================

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Saved at the end of every epoch — use this to resume training
LATEST_CKPT_NAME: str = "latest_checkpoint.pt"

# Saved only when val_loss improves — use this for final evaluation
BEST_CKPT_NAME:   str = "best_model.pt"

# Class-label -> integer mapping saved alongside checkpoints
LABEL_MAP_NAME:   str = "label_to_id.json"

# Per-epoch loss / accuracy log saved in reports/
HISTORY_CSV_NAME: str = "training_history.csv"
