"""
train_custom.py
===============
Training pipeline for the Custom Vocabulary Sign Language Classifier.

Trains a separate GRU classifier on custom-recorded MediaPipe landmark
sequences while preserving existing ISL500 checkpoints intact.

Outputs
-------
- Model Checkpoint : models/checkpoints/custom_model.pt
- Label Encoder    : models/checkpoints/custom_label_to_id.json
- Training History : reports/custom_training_history.csv
- Test Metrics     : reports/custom_test_metrics.json

Usage
-----
# Train with default parameters on data/custom_raw:
python src/train_custom.py

# Specify data directory and training options:
python src/train_custom.py --data-dir data/custom_raw --epochs 30 --batch-size 16 --lr 1e-3

# Verify dataset without training:
python src/train_custom.py --validate-only
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR,
    REPORT_DIR,
    CUSTOM_RAW_DIR,
    CUSTOM_CKPT_NAME,
    CUSTOM_LABEL_MAP_NAME,
    CUSTOM_RECORDS_CSV,
    MAX_SEQ_LEN,
    FEATURE_DIM,
    EXPECTED_LANDMARK_SHAPE,
    RANDOM_SEED,
)
from src.preprocessing import preprocess_sample
from src.feature_extraction import compute_seq_lengths
from src.train import GRUClassifier, get_device, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
)
logger = logging.getLogger("train_custom")


# ── Dataset Loading & Validation ──────────────────────────────────────────────

def discover_custom_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Scan data_dir for .npz custom recording files.

    Returns DataFrame with columns:
        file_path, label, signer, rep, frame_count
    """
    records = []
    npz_files = sorted(list(data_dir.glob("*.npz")))

    for p in npz_files:
        try:
            with np.load(p, allow_pickle=False) as d:
                if "landmarks" not in d or "metadata_json" not in d:
                    logger.warning(f"Skipping malformed archive: {p.name}")
                    continue
                meta = json.loads(str(d["metadata_json"]))
                records.append({
                    "file_path": str(p),
                    "label": meta.get("label", "Unknown"),
                    "signer": meta.get("signer", "signer01"),
                    "rep": meta.get("rep", 1),
                    "frame_count": meta.get("frame_count", len(d["landmarks"])),
                })
        except Exception as e:
            logger.warning(f"Error reading {p.name}: {e}")

    df = pd.DataFrame(records)
    return df


def validate_dataset(df: pd.DataFrame, min_samples_per_class: int = 3) -> Dict[str, Any]:
    """
    Validate custom dataset sufficiency and balance.
    """
    if df.empty:
        raise ValueError(
            "No valid .npz recordings found in dataset directory.\n"
            "Please record samples first using:  python src/record_custom.py"
        )

    class_counts = df["label"].value_counts().to_dict()
    signer_counts = df["signer"].value_counts().to_dict()
    total_samples = len(df)
    unique_classes = sorted(list(class_counts.keys()))
    unique_signers = sorted(list(signer_counts.keys()))

    warnings = []
    for cls, cnt in class_counts.items():
        if cnt < min_samples_per_class:
            warnings.append(f"Class '{cls}' has only {cnt} sample(s) (recommended: >= {min_samples_per_class}).")

    if len(unique_signers) < 2:
        warnings.append(
            f"Only {len(unique_signers)} signer detected ({unique_signers}). "
            "Signer-independent splitting requires at least 2 signers. "
            "Falling back to sample-stratified splitting."
        )

    return {
        "total_samples": total_samples,
        "class_counts": class_counts,
        "signer_counts": signer_counts,
        "unique_classes": unique_classes,
        "unique_signers": unique_signers,
        "warnings": warnings,
    }


def split_custom_dataset(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split custom dataset into train and validation sets.

    Prefers signer-independent splitting if >= 2 signers are present.
    Falls back to class-stratified splitting if only 1 signer is present.

    Returns:
        (train_df, val_df, split_strategy_name)
    """
    signers = df["signer"].unique().tolist()

    if len(signers) >= 2:
        # Signer-independent split: reserve at least 1 signer with ~val_fraction of samples for val
        rng = np.random.default_rng(seed)
        shuffled_signers = list(signers)
        rng.shuffle(shuffled_signers)

        val_target_count = max(1, int(len(df) * val_fraction))
        val_signers = []
        curr_val_count = 0

        for s in shuffled_signers:
            s_count = len(df[df["signer"] == s])
            if curr_val_count + s_count <= val_target_count or not val_signers:
                val_signers.append(s)
                curr_val_count += s_count
            else:
                break

        train_df = df[~df["signer"].isin(val_signers)].reset_index(drop=True)
        val_df = df[df["signer"].isin(val_signers)].reset_index(drop=True)

        # Ensure all classes in val are present in train
        val_classes = set(val_df["label"].unique())
        train_classes = set(train_df["label"].unique())
        if val_classes - train_classes:
            logger.info("Signer split produced unseen classes in val; falling back to stratified split.")
            # Fall through to stratified split
        else:
            return train_df, val_df, f"Signer-Independent (Val Signers: {val_signers})"

    # Fallback: Class-stratified split within available samples
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for _, group in df.groupby("label"):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction)) if len(idx) > 1 else 0
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])

    # If any sample had only 1 instance, ensure it's in train
    if not train_indices:
        train_indices = val_indices
        val_indices = []

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        "Class-Stratified (Single/Mixed Signer)",
    )


# ── PyTorch Dataset for Custom Recordings ─────────────────────────────────────

class CustomLandmarkDataset(Dataset):
    """
    Dataset loader for custom .npz landmark archives.
    Applies standard preprocess_sample() pipeline.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_id: Dict[str, int],
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.paths = df["file_path"].tolist()
        self.labels = [label_to_id[lbl] for lbl in df["label"].tolist()]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, int]:
        npz_path = Path(self.paths[idx])
        with np.load(npz_path, allow_pickle=False) as d:
            raw_lms = d["landmarks"].astype(np.float32)

        # Standard preprocessing: remove_empty -> normalize -> flatten -> pad/truncate
        features = preprocess_sample(raw_lms, max_seq_len=self.max_seq_len)
        return torch.from_numpy(features), self.labels[idx]


# ── Training & Evaluation Loops ───────────────────────────────────────────────

def run_custom_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            lengths = compute_seq_lengths(bx)
            logits = model(bx, lengths)
            loss = criterion(logits, by)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item() * len(by)
            total_correct += (logits.argmax(dim=-1) == by).sum().item()
            total_n += len(by)

    avg_loss = total_loss / max(total_n, 1)
    acc = (total_correct / max(total_n, 1)) * 100.0
    return avg_loss, acc


def evaluate_custom_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            lengths = compute_seq_lengths(bx)
            logits = model(bx, lengths)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(by.tolist())

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    target_ids = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [id_to_label[i] for i in target_ids]

    rep_dict = classification_report(
        all_targets,
        all_preds,
        labels=target_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    rep_str = classification_report(
        all_targets,
        all_preds,
        labels=target_ids,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report_dict": rep_dict,
        "report_str": rep_str,
    }


def train_custom_model(
    data_dir: Optional[Path] = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 128,
    patience: int = 6,
) -> None:
    set_seeds(RANDOM_SEED)
    device = get_device()
    src_dir = Path(data_dir) if data_dir else CUSTOM_RAW_DIR

    print("\n" + "=" * 65)
    print("  Custom Vocabulary Classifier Training Pipeline")
    print("=" * 65)
    print(f"  Data Source : {src_dir}")
    print(f"  Device      : {device}")
    print(f"  Max Epochs  : {epochs}  |  Batch Size: {batch_size}  |  LR: {lr}")
    print("-" * 65)

    # 1. Discover and validate dataset
    df = discover_custom_dataset(src_dir)
    stats = validate_dataset(df)

    print(f"[*] Discovered {stats['total_samples']} samples across {len(stats['unique_classes'])} classes:")
    for cls, cnt in stats["class_counts"].items():
        print(f"      - {cls:<25} : {cnt:>3} samples")
    print(f"[*] Signers: {stats['unique_signers']}")

    if stats["warnings"]:
        print("\n[!] Validation Warnings:")
        for w in stats["warnings"]:
            print(f"    WARNING: {w}")

    # 2. Build custom label encoder
    custom_label_to_id = {lbl: idx for idx, lbl in enumerate(stats["unique_classes"])}
    custom_id_to_label = {idx: lbl for lbl, idx in custom_label_to_id.items()}
    num_classes = len(custom_label_to_id)

    # 3. Split dataset
    train_df, val_df, split_strategy = split_custom_dataset(df)
    print(f"\n[*] Split Strategy : {split_strategy}")
    print(f"    Train Samples  : {len(train_df)}")
    print(f"    Val Samples    : {len(val_df)}")

    # 4. DataLoaders
    train_ds = CustomLandmarkDataset(train_df, custom_label_to_id)
    val_ds = CustomLandmarkDataset(val_df, custom_label_to_id) if not val_df.empty else train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 5. Model Architecture
    model = GRUClassifier(
        input_size=FEATURE_DIM,
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        fc_hidden=hidden_size,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    custom_ckpt_path = CHECKPOINT_DIR / CUSTOM_CKPT_NAME
    custom_label_map_path = CHECKPOINT_DIR / CUSTOM_LABEL_MAP_NAME
    history_csv = REPORT_DIR / "custom_training_history.csv"

    # Save label encoder JSON
    with open(custom_label_map_path, "w", encoding="utf-8") as f:
        json.dump(custom_label_to_id, f, indent=2)
    print(f"[*] Saved custom label map: {custom_label_map_path}")

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_cnt = 0
    history = []

    print("\n" + "-" * 65)
    print("  Starting Training Loop...")
    print("-" * 65)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = run_custom_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_custom_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:>3}/{epochs}"
            f"  loss: {train_loss:.4f}  acc: {train_acc:.1f}%"
            f"  |  val_loss: {val_loss:.4f}  val_acc: {val_acc:.1f}%"
            f"  |  lr: {curr_lr:.2e}  ({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 2),
            "lr": curr_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0

            # Atomic save custom checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label_to_id": custom_label_to_id,
                "config": {
                    "feature_dim": FEATURE_DIM,
                    "num_classes": num_classes,
                    "gru_hidden": hidden_size,
                    "gru_layers": 2,
                    "fc_hidden": hidden_size,
                    "max_seq_len": MAX_SEQ_LEN,
                },
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
            }, custom_ckpt_path)
            print(f"  -> Best custom model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[*] Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save history CSV
    pd.DataFrame(history).to_csv(history_csv, index=False)

    # 7. Final Evaluation on Best Model
    print("\n" + "=" * 65)
    print("  Final Custom Model Evaluation")
    print("=" * 65)
    best_ckpt = torch.load(custom_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    eval_results = evaluate_custom_model(model, val_loader, device, custom_id_to_label)
    print(f"  Accuracy : {eval_results['accuracy']:.2f}%")
    print(f"  Macro F1 : {eval_results['macro_f1']:.4f}")
    print("\nClassification Report:\n" + eval_results["report_str"])

    # Save metrics JSON
    metrics_path = REPORT_DIR / "custom_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(custom_ckpt_path),
            "classes": custom_label_to_id,
            "accuracy": round(eval_results["accuracy"], 2),
            "macro_f1": round(eval_results["macro_f1"], 4),
            "weighted_f1": round(eval_results["weighted_f1"], 4),
            "split_strategy": split_strategy,
            "total_samples": len(df),
        }, f, indent=2)

    print(f"[*] Checkpoint saved : {custom_ckpt_path}")
    print(f"[*] Label map saved  : {custom_label_map_path}")
    print(f"[*] Metrics saved    : {metrics_path}")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Custom Vocabulary GRU Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom raw landmark directory (default: data/custom_raw)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="GRU hidden units (default: 128)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset statistics without training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir = Path(args.data_dir) if args.data_dir else CUSTOM_RAW_DIR

    if args.validate_only:
        df = discover_custom_dataset(target_dir)
        stats = validate_dataset(df)
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    train_custom_model(
        data_dir=target_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
    )
