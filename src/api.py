"""
api.py
======
FastAPI Inference Backend for the ISL500 Sign Language Recognition Model.

Exposes RESTful endpoints for receiving MediaPipe landmark sequences
and returning predicted Indian Sign Language signs.

Endpoints
---------
- GET  /         : Service info and navigation
- GET  /health   : Health check, model status, and hardware device info
- POST /predict  : Predict gesture label from landmark sequence (frames, 2, 21, 3)
- GET  /docs     : Interactive Swagger API documentation

Usage
-----
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import EXPECTED_LANDMARK_SHAPE, NUM_CLASSES
from src.predict import SignPredictor
from src.hybrid_predict import HybridSignPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("isl500_api")

# Global hybrid predictor instance (loaded once on application lifespan startup)
predictor: Optional[HybridSignPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads the trained models on startup."""
    global predictor
    logger.info("Initializing HybridSignPredictor on application startup...")
    try:
        predictor = HybridSignPredictor()
        logger.info(
            f"HybridSignPredictor initialized successfully: device={predictor.device}, "
            f"ISL500 classes={predictor.isl500_predictor.num_classes}, "
            f"Custom loaded={predictor.has_custom_model}"
        )
    except Exception as e:
        logger.error(f"Failed to load HybridSignPredictor on startup: {e}", exc_info=True)
        predictor = None
    yield
    logger.info("Shutting down ISL500 API service.")


# Initialize FastAPI app
app = FastAPI(
    title="ISL500 Sign Language Recognition API",
    description=(
        "Production-grade inference backend for classifying Indian Sign Language (ISL) gestures "
        "from MediaPipe hand landmark sequences using a trained GRU neural network."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
CORS_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
if CORS_ORIGINS_ENV.strip():
    allowed_origins = [orig.strip() for orig in CORS_ORIGINS_ENV.split(",") if orig.strip()]
else:
    # Default local development origins
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Pydantic Request & Response Models ────────────────────────────────────────

class PredictRequest(BaseModel):
    landmarks: List[List[List[List[float]]]] = Field(
        ...,
        description="4D nested array of shape (frames, 2, 21, 3) representing 2 hands x 21 landmarks x (x, y, z).",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top predictions to return (1-50, default: 5).",
    )
    recognizer: Optional[str] = Field(
        default="isl500",
        description="Recognizer selection: 'isl500' (default), 'custom', or 'hybrid'.",
    )


class TopPrediction(BaseModel):
    label: str = Field(..., description="Gesture label name")
    confidence: float = Field(..., description="Predicted probability in range [0.0, 1.0]")
    class_id: int = Field(..., description="Integer class identifier")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Top-1 predicted gesture name")
    confidence: float = Field(..., description="Top-1 confidence score in range [0.0, 1.0]")
    class_id: int = Field(..., description="Top-1 integer class ID")
    top_k: List[TopPrediction] = Field(..., description="Top-K predicted candidates ranked by confidence")
    raw_frame_count: int = Field(..., description="Number of frames received in the input sequence")
    effective_seq_len: int = Field(..., description="Active non-padded sequence length after preprocessing")
    recognizer: Optional[str] = Field(
        default="isl500",
        description="Recognizer model used for the prediction ('isl500' or 'custom').",
    )
    hybrid_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitration details when evaluated under hybrid mode.",
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    device: Optional[str] = None
    device_name: Optional[str] = None
    num_classes: Optional[int] = None
    checkpoint_name: Optional[str] = None
    custom_model_loaded: bool = False
    custom_classes: Optional[int] = None


class RootResponse(BaseModel):
    service: str
    version: str
    status: str
    docs_url: str
    health_url: str
    predict_url: str


# ── Global Exception Handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Prevent exposing internal stack traces or filesystem paths to clients."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred during request processing."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=RootResponse, tags=["General"])
async def root():
    """Service identification and API documentation index."""
    return RootResponse(
        service="ISL500 Sign Language Recognition Inference API",
        version="1.0.0",
        status="online",
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint: verifies service status and model availability."""
    if predictor is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            device=None,
            device_name=None,
            num_classes=None,
            checkpoint_name=None,
            custom_model_loaded=False,
            custom_classes=None,
        )

    dev_str = str(predictor.device)
    dev_name = (
        torch.cuda.get_device_name(predictor.device)
        if predictor.device.type == "cuda"
        else "CPU"
    )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=dev_str,
        device_name=dev_name,
        num_classes=predictor.isl500_predictor.num_classes,
        checkpoint_name=predictor.isl500_predictor.checkpoint_path.name,
        custom_model_loaded=predictor.has_custom_model,
        custom_classes=predictor.custom_predictor.num_classes if predictor.has_custom_model else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_gesture(request: PredictRequest):
    """
    Classify a MediaPipe hand landmark sequence into an Indian Sign Language gesture.

    Input:
      - landmarks: 4D array of shape (frames, 2, 21, 3).
      - top_k: Number of predictions to return (default: 5).

    Returns:
      - Top-1 prediction label, confidence, and top-K ranked candidates.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not loaded or is currently unavailable.",
        )

    if not request.landmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Landmark sequence is empty (0 frames). At least 1 frame is required.",
        )

    # Convert to NumPy array
    try:
        arr = np.asarray(request.landmarks, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed landmark data: nested elements could not be converted to float32 matrix.",
        )

    # Dimension and shape validation
    if arr.ndim != 4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid landmark array dimensions: got {arr.ndim}D array with shape {list(arr.shape)}. "
                "Expected 4D array with shape (frames, 2, 21, 3)."
            ),
        )

    if arr.shape[0] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input landmark sequence contains 0 frames. At least 1 frame is required.",
        )

    if arr.shape[1:] != EXPECTED_LANDMARK_SHAPE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid per-frame landmark shape {list(arr.shape[1:])}. "
                f"Expected [2, 21, 3] representing 2 hands, 21 landmarks, and 3 coordinates (x, y, z)."
            ),
        )

    # Non-finite value check
    if not np.all(np.isfinite(arr)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Landmark sequence contains non-finite values (NaN or Inf).",
        )

    # Inference execution
    try:
        rec_mode = request.recognizer or "isl500"
        result = predictor.predict(arr, top_k=request.top_k, mode=rec_mode)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Inference execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during model inference.",
        )

    return PredictResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
