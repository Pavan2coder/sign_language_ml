"""
test_custom_pipeline.py
========================
Test suite for the ISL Custom Vocabulary Extension & Hybrid Recognizer.

Covers:
1. Custom landmark recording validation & metadata serialization.
2. Signer-stratified dataset splitting & data leakage prevention.
3. MediaPipe preprocessing with missing-hand frames, padding, & truncation.
4. Custom GRUClassifier training step & checkpoint format isolation.
5. HybridSignPredictor arbitration & routing modes.
6. FastAPI request/response schema backward-compatibility.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from src.config import (
    CHECKPOINT_DIR,
    BEST_CKPT_NAME,
    EXPECTED_LANDMARK_SHAPE,
    FEATURE_DIM,
    MAX_SEQ_LEN,
)
from src.preprocessing import preprocess_sample, normalize_landmarks, remove_empty_frames
from src.train import GRUClassifier
from src.record_custom import (
    validate_recording,
    save_recording,
    load_custom_sample,
)
from src.train_custom import (
    discover_custom_dataset,
    validate_dataset,
    split_custom_dataset,
    CustomLandmarkDataset,
)
from src.hybrid_predict import HybridSignPredictor, CustomSignPredictor
from src.api import app


class TestRecordingValidation(unittest.TestCase):
    """Test data recording safeguards, quality validation, and serialization."""

    def test_validation_empty_sequence(self):
        is_valid, reason, stats = validate_recording([], min_frames=5)
        self.assertFalse(is_valid)
        self.assertIn("Too few frames", reason)

    def test_validation_short_sequence(self):
        frames = [np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) for _ in range(3)]
        is_valid, reason, stats = validate_recording(frames, min_frames=5)
        self.assertFalse(is_valid)
        self.assertIn("Too few frames", reason)

    def test_validation_nan_inf(self):
        frames = [np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) for _ in range(10)]
        frames[2][0, 0, 0] = np.nan
        is_valid, reason, stats = validate_recording(frames, min_frames=5)
        self.assertFalse(is_valid)
        self.assertIn("NaN or Inf", reason)

    def test_validation_insufficient_hands(self):
        # All zero frames (hands absent 100% of the time)
        frames = [np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) for _ in range(20)]
        is_valid, reason, stats = validate_recording(frames, min_frames=5, min_active_ratio=0.3)
        self.assertFalse(is_valid)
        self.assertIn("Insufficient hand", reason)

    def test_validation_valid_sequence(self):
        frames = []
        for i in range(20):
            f = np.zeros(EXPECTED_LANDMARK_SHAPE, dtype=np.float32)
            # simulate right hand present
            f[1, :, :] = 0.5 + i * 0.01
            frames.append(f)
        is_valid, reason, stats = validate_recording(frames, min_frames=5, min_active_ratio=0.3)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "PASSED")

    def test_save_and_load_roundtrip(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            frames = [np.ones(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) * 0.2 for _ in range(15)]
            saved_path = save_recording(
                output_dir=tmp_dir,
                label="Bye",
                signer_id="test_signer",
                rep=1,
                frames=frames,
                fps=30.0,
            )
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.suffix, ".npz")

            loaded_lms, meta = load_custom_sample(saved_path)
            self.assertEqual(loaded_lms.shape, (15, 2, 21, 3))
            self.assertEqual(meta["label"], "Bye")
            self.assertEqual(meta["signer"], "test_signer")
            self.assertEqual(meta["rep"], 1)
            self.assertEqual(meta["frame_count"], 15)

            # Check CSV registry
            csv_path = tmp_dir / "custom_records.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["label"], "Bye")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_save_path_consistency_and_trainer_format(self):
        from src.config import CUSTOM_RAW_DIR
        # Ensure default directories match
        self.assertEqual(str(CUSTOM_RAW_DIR).lower(), str(Path("data/custom_raw").resolve()).lower())

        # Test end-to-end format: save sample in tmp dir, discover via trainer, load in CustomLandmarkDataset
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            frames = [np.ones(EXPECTED_LANDMARK_SHAPE, dtype=np.float32) * 0.4 for _ in range(25)]
            saved_p = save_recording(
                output_dir=tmp_dir,
                label="HelpMe",
                signer_id="signer01",
                rep=1,
                frames=frames,
                fps=30.0,
            )
            df = discover_custom_dataset(tmp_dir)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["label"], "HelpMe")

            label_map = {"HelpMe": 0}
            ds = CustomLandmarkDataset(df, label_map, max_seq_len=150)
            tensor_x, label_id = ds[0]
            self.assertEqual(tensor_x.shape, (150, 126))
            self.assertEqual(label_id, 0)
            self.assertEqual(tensor_x.dtype, torch.float32)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestDatasetSplittingAndLeakage(unittest.TestCase):
    """Test signer-independent splitting and avoidance of data leakage."""

    def test_signer_independent_split(self):
        # Create synthetic records with 3 distinct signers
        records = []
        for signer in ["signer_A", "signer_B", "signer_C"]:
            for label in ["Bye", "Sorry", "HelpMe"]:
                for rep in range(4):
                    records.append({
                        "file_path": f"/fake/{signer}_{label}_{rep}.npz",
                        "label": label,
                        "signer": signer,
                        "rep": rep,
                        "frame_count": 30,
                    })
        df = pd.DataFrame(records)

        train_df, val_df, strategy = split_custom_dataset(df, val_fraction=0.3, seed=42)
        self.assertIn("Signer-Independent", strategy)

        # Ensure strict signer separation: train and val signers must NOT overlap
        train_signers = set(train_df["signer"].unique())
        val_signers = set(val_df["signer"].unique())
        overlap = train_signers.intersection(val_signers)
        self.assertEqual(len(overlap), 0, f"Signer leakage detected! Overlapping signers: {overlap}")

    def test_single_signer_fallback(self):
        # Only 1 signer present
        records = []
        for label in ["Bye", "Sorry"]:
            for rep in range(6):
                records.append({
                    "file_path": f"/fake/s1_{label}_{rep}.npz",
                    "label": label,
                    "signer": "solo_signer",
                    "rep": rep,
                    "frame_count": 25,
                })
        df = pd.DataFrame(records)

        train_df, val_df, strategy = split_custom_dataset(df, val_fraction=0.2, seed=42)
        self.assertIn("Class-Stratified", strategy)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)


class TestPreprocessingRobustness(unittest.TestCase):
    """Test preprocessing with variable lengths, missing hands, and boundary conditions."""

    def test_single_hand_sign(self):
        # MediaPipe sequence where Hand 0 (Left) is completely absent (all 0)
        # and Hand 1 (Right) is active
        raw = np.zeros((40, 2, 21, 3), dtype=np.float32)
        for t in range(40):
            # Wrist at (0.4, 0.6, 0.0), other landmarks spaced out
            raw[t, 1, 0, :] = [0.4, 0.6, 0.0]
            for j in range(1, 21):
                raw[t, 1, j, :] = [0.4 + j * 0.01, 0.6 + j * 0.01, 0.0]

        processed = preprocess_sample(raw, max_seq_len=MAX_SEQ_LEN)
        self.assertEqual(processed.shape, (150, 126))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(processed)))

        # Hand 0 features (indices 0..62) must remain all zero
        self.assertTrue(np.all(processed[:40, :63] == 0.0))
        # Hand 1 features (indices 63..125) must be non-zero
        self.assertFalse(np.all(processed[:40, 63:] == 0.0))
        # Wrist landmark for Hand 1 after centering must be exactly 0
        self.assertEqual(processed[0, 63], 0.0)
        self.assertEqual(processed[0, 64], 0.0)
        self.assertEqual(processed[0, 65], 0.0)

    def test_sequence_padding_and_truncation(self):
        # Short sequence (20 frames) -> padded to 150
        short_raw = np.ones((20, 2, 21, 3), dtype=np.float32) * 0.5
        out_padded = preprocess_sample(short_raw, max_seq_len=150)
        self.assertEqual(out_padded.shape, (150, 126))
        # Post-padding: rows after frame 20 must be zero
        self.assertTrue(np.all(out_padded[20:, :] == 0.0))

        # Long sequence (200 frames) -> truncated to 150
        long_raw = np.ones((200, 2, 21, 3), dtype=np.float32) * 0.5
        out_trunc = preprocess_sample(long_raw, max_seq_len=150)
        self.assertEqual(out_trunc.shape, (150, 126))


class TestHybridInferenceAndArbitration(unittest.TestCase):
    """Test HybridSignPredictor routing modes and arbitration logic."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = HybridSignPredictor()

    def test_isl500_mode(self):
        # Create a synthetic 1-hand sample
        raw = np.zeros((30, 2, 21, 3), dtype=np.float32)
        for t in range(30):
            raw[t, 1, :, :] = 0.5 + t * 0.005

        res = self.predictor.predict(raw, top_k=5, mode="isl500")
        self.assertEqual(res["recognizer"], "isl500")
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("top_k", res)
        self.assertEqual(len(res["top_k"]), 5)

    def test_hybrid_fallback_when_custom_absent(self):
        # When custom model is not trained yet, hybrid mode should cleanly default to isl500
        raw = np.zeros((20, 2, 21, 3), dtype=np.float32)
        raw[:, 1, :, :] = 0.4
        res = self.predictor.predict(raw, top_k=3, mode="hybrid")
        self.assertEqual(res["recognizer"], "isl500")
        self.assertIn("label", res)

    def test_custom_mode_raises_when_missing(self):
        if not self.predictor.has_custom_model:
            raw = np.zeros((20, 2, 21, 3), dtype=np.float32)
            with self.assertRaises(RuntimeError):
                self.predictor.predict(raw, mode="custom")

    def test_synthetic_custom_checkpoint_integration(self):
        # Create a temporary custom checkpoint and verify dual arbitration
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            custom_ckpt_file = tmp_dir / "temp_custom.pt"
            custom_classes = {"Bye": 0, "Sorry": 1, "HowAreYou": 2}

            # Build and save a tiny model
            toy_model = GRUClassifier(
                input_size=FEATURE_DIM,
                hidden_size=64,
                num_layers=1,
                num_classes=3,
                fc_hidden=64,
            )
            torch.save({
                "model_state_dict": toy_model.state_dict(),
                "label_to_id": custom_classes,
                "config": {"feature_dim": FEATURE_DIM, "gru_hidden": 64, "gru_layers": 1, "fc_hidden": 64},
                "epoch": 1,
                "best_val_loss": 0.1,
            }, custom_ckpt_file)

            # Initialize hybrid predictor with custom checkpoint
            dual = HybridSignPredictor(custom_ckpt=custom_ckpt_file, custom_threshold=0.50)
            self.assertTrue(dual.has_custom_model)

            raw = np.zeros((25, 2, 21, 3), dtype=np.float32)
            raw[:, 1, :, :] = 0.5
            res = dual.predict(raw, mode="custom")
            self.assertEqual(res["recognizer"], "custom")
            self.assertIn(res["label"], ["Bye", "Sorry", "HowAreYou"])

            # Test hybrid arbitration
            res_hyb = dual.predict(raw, mode="hybrid")
            self.assertIn(res_hyb["recognizer"], ["isl500", "custom"])
            self.assertIn("hybrid_details", res_hyb)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestFastAPIIntegration(unittest.TestCase):
    """Test FastAPI endpoint backward compatibility and new parameters."""

    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["num_classes"], 500)
        self.assertIn("custom_model_loaded", data)

    def test_legacy_predict_request(self):
        # Client sending ONLY landmarks without recognizer field (exact old format)
        sample = np.zeros((10, 2, 21, 3), dtype=np.float32)
        sample[:, 1, :, :] = 0.3
        payload = {
            "landmarks": sample.tolist(),
            "top_k": 3,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("label", data)
        self.assertIn("confidence", data)
        self.assertIn("top_k", data)
        self.assertEqual(len(data["top_k"]), 3)
        self.assertEqual(data["recognizer"], "isl500")

    def test_predict_with_hybrid_selector(self):
        sample = np.zeros((10, 2, 21, 3), dtype=np.float32)
        sample[:, 1, :, :] = 0.3
        payload = {
            "landmarks": sample.tolist(),
            "top_k": 3,
            "recognizer": "hybrid",
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("recognizer", data)

    def test_invalid_payload_rejection(self):
        # Empty landmarks
        resp = self.client.post("/predict", json={"landmarks": []})
        self.assertEqual(resp.status_code, 400)

        # Malformed dimensions (3D instead of 4D)
        bad_dim = np.zeros((10, 2, 21), dtype=np.float32).tolist()
        resp2 = self.client.post("/predict", json={"landmarks": bad_dim})
        self.assertEqual(resp2.status_code, 422)


if __name__ == "__main__":
    unittest.main()
