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

# GRU model architecture
# input  : (batch, MAX_SEQ_LEN=150, FEATURE_DIM=126)
# GRU    : GRU_NUM_LAYERS stacked layers, hidden=GRU_HIDDEN_SIZE
# head   : LayerNorm -> Linear(GRU_HIDDEN_SIZE, FC_HIDDEN_SIZE) -> ReLU
#          -> Dropout -> Linear(FC_HIDDEN_SIZE, NUM_CLASSES)
GRU_HIDDEN_SIZE: int   = 256   # hidden units per GRU layer
GRU_NUM_LAYERS:  int   = 2     # stacked layers
GRU_DROPOUT:     float = 0.3   # dropout between layers (ignored if layers=1)
FC_HIDDEN_SIZE:  int   = 256   # fully-connected head hidden dimension
