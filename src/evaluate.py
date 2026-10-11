"""
evaluate.py
===========
STEP 6 of the ISL Sign Language Recognition ML Pipeline:
Final evaluation on the untouched test split.

Usage
-----
python src/evaluate.py                        # evaluate best_model.pt
python src/evaluate.py --checkpoint path.pt   # specific checkpoint
python src/evaluate.py --batch-size 16        # custom batch size
python src/evaluate.py --smoke                # 100-sample sanity check

The test split is NEVER used during training or model selection.
This script is the only place it is read.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CHECKPOINT_DIR, BEST_CKPT_NAME, REPORT_DIR,
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
    DEFAULT_BATCH_SIZE,
)
from src.data_loader        import load_mapping
from src.feature_extraction import ISLDataset, compute_seq_lengths
from src.train              import GRUClassifier, get_device

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def evaluate(ckpt_path=None, batch_size=DEFAULT_BATCH_SIZE, smoke=False):
    """
    Load checkpoint and evaluate on the untouched test split.

    Steps:
      1. Resolve checkpoint path (default: best_model.pt)
      2. Load checkpoint; extract label_to_id mapping
      3. Build GRUClassifier and load saved weights
      4. Load test mapping (never used during training)
      5. Validate test samples (detect unknown labels, missing files)
      6. Build non-shuffled test DataLoader using ISLDataset
      7. Run inference under torch.inference_mode()
      8. Compute test loss, accuracy, classification report, confusion matrix
      9. Save test_metrics.json, classification_report.csv, confusion_matrix.csv
      10. Print readable summary of final metrics

    Args:
        ckpt_path  : Path to a .pt file; defaults to models/checkpoints/best_model.pt.
        batch_size : evaluation batch size.
        smoke      : if True, use only 100 test samples.

    Returns:
        dict with test_loss, accuracy, top1_accuracy, top5_accuracy,
        report_str, confusion_matrix, metrics, report_df, cm_df.
    """
    if ckpt_path is None:
        ckpt_path = CHECKPOINT_DIR / BEST_CKPT_NAME
    else:
        ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run training first:  python src/train.py"
        )

    device = get_device()

    # 1. Load checkpoint
    print(f"\n[1] Loading checkpoint: {ckpt_path}")
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)
    cfg         = ckpt.get("config", {})
    best_vloss  = ckpt.get("best_val_loss")
    best_vloss_str = f"{best_vloss:.4f}" if best_vloss is not None else "N/A"
    print(f"    Epoch: {ckpt.get('epoch', '?')}  "
          f"Best val loss: {best_vloss_str}  "
          f"Classes: {num_classes}")

    # 2. Build model and load weights
    print("[2] Building model...")
    hidden_size = cfg.get("gru_hidden", GRU_HIDDEN_SIZE)
    num_layers  = cfg.get("gru_layers", GRU_NUM_LAYERS)
    fc_hidden   = cfg.get("fc_hidden", FC_HIDDEN_SIZE)
    dropout     = cfg.get("gru_dropout", GRU_DROPOUT)
    input_size  = cfg.get("feature_dim", FEATURE_DIM)

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        fc_hidden=fc_hidden,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Load test split (untouched during training)
    print("[3] Loading test mapping...")
    test_df = load_mapping("test")
    initial_total = len(test_df)
    print(f"    Total test samples in mapping: {initial_total:,}")

    if smoke:
        smoke_count = min(100, len(test_df))
        test_df = test_df.sample(smoke_count, random_state=42).reset_index(drop=True)
        print(f"    [smoke] Sampled {len(test_df)} test samples")

    # 4. Detect and report invalid samples (unknown labels, missing files)
    print("[4] Validating test samples...")
    excluded_records = []

    # Check unknown labels
    unknown_labels = set(test_df["label"].unique()) - set(label_to_id.keys())
    if unknown_labels:
        print(f"    WARNING: {len(unknown_labels)} unique label(s) not in checkpoint encoder: {sorted(unknown_labels)[:5]}")
        unknown_mask = test_df["label"].isin(unknown_labels)
        for _, row in test_df[unknown_mask].iterrows():
            excluded_records.append({
                "path": str(row["local_path"]),
                "label": str(row["label"]),
                "reason": f"Unknown label '{row['label']}' not in checkpoint label_to_id",
            })
        test_df = test_df[~unknown_mask].reset_index(drop=True)

    # Check missing files
    missing_mask = ~test_df["local_path"].map(lambda p: Path(p).is_file())
    if missing_mask.any():
        print(f"    WARNING: {missing_mask.sum()} test sample file(s) missing on disk.")
        for _, row in test_df[missing_mask].iterrows():
            excluded_records.append({
                "path": str(row["local_path"]),
                "label": str(row["label"]),
                "reason": f"File not found on disk: {row['local_path']}",
            })
        test_df = test_df[~missing_mask].reset_index(drop=True)

    if excluded_records:
        print(f"    Excluded {len(excluded_records)} invalid sample(s). Remaining for evaluation: {len(test_df):,}")
    else:
        print(f"    All {len(test_df):,} test samples verified successfully (0 excluded).")

    if len(test_df) == 0:
        raise RuntimeError("No valid test samples remaining for evaluation.")

    # 5. Create DataLoader
    print("[5] Creating DataLoader...")
    test_ds = ISLDataset(test_df, label_to_id, max_seq_len=MAX_SEQ_LEN)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"    Batches: {len(test_loader)}  (batch_size={batch_size})")

    # 6. Run inference
    print("[6] Running inference...")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_targets = []
    total_top5_correct = 0

    start_time = time.time()
    iterator = test_loader
    if HAS_TQDM:
        iterator = tqdm(test_loader, desc="Evaluating", unit="batch")

    with torch.inference_mode():
        for bx, by in iterator:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            lengths = compute_seq_lengths(bx)
            logits = model(bx, lengths)

            loss = criterion(logits, by)
            total_loss += loss.item()
            total_n += len(by)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(by.cpu().tolist())

            if num_classes >= 5:
                _, top5 = logits.topk(5, dim=-1)
                total_top5_correct += (top5 == by.unsqueeze(-1)).any(dim=-1).sum().item()
            else:
                total_top5_correct += (preds == by).sum().item()

    elapsed_time = time.time() - start_time
    print(f"    Inference completed in {elapsed_time:.2f}s ({total_n / max(elapsed_time, 1e-6):.1f} samples/sec)")

    # 7. Compute metrics
    test_loss = total_loss / max(total_n, 1)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    test_acc = (correct / max(total_n, 1)) * 100.0
    top5_acc = (total_top5_correct / max(total_n, 1)) * 100.0

    print("[7] Computing classification metrics...")
    if smoke:
        eval_label_ids = sorted(list(set(all_targets) | set(all_preds)))
        eval_target_names = [id_to_label[i] for i in eval_label_ids]
    else:
        eval_label_ids = list(range(num_classes))
        eval_target_names = [id_to_label[i] for i in eval_label_ids]

    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=eval_label_ids,
        target_names=eval_target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_targets,
        all_preds,
        labels=eval_label_ids,
        target_names=eval_target_names,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
    )

    # 8. Save outputs to reports/
    print("[8] Saving reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Classification report CSV
    report_df = pd.DataFrame(report_dict).transpose()
    if "accuracy" in report_df.index:
        report_df.loc["accuracy", "support"] = total_n
    report_csv_path = REPORT_DIR / "classification_report.csv"
    report_df.to_csv(report_csv_path, index_label="class")

    # Confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=[id_to_label[i] for i in range(num_classes)],
        columns=[id_to_label[i] for i in range(num_classes)],
    )
    cm_csv_path = REPORT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index_label="true_label")

    # Numpy confusion matrix
    np.save(REPORT_DIR / "confusion_matrix.npy", cm)

    # Test metrics JSON
    metrics_json_path = REPORT_DIR / "test_metrics.json"
    metrics = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "best_val_loss_at_training": ckpt.get("best_val_loss"),
        "num_classes": num_classes,
        "total_mapping_samples": initial_total,
        "evaluated_samples": total_n,
        "excluded_samples_count": len(excluded_records),
        "excluded_samples": excluded_records,
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 4),
        "top1_accuracy": round(test_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0)), 6),
        "weighted_f1": round(float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)), 6),
        "macro_precision": round(float(report_dict.get("macro avg", {}).get("precision", 0.0)), 6),
        "weighted_precision": round(float(report_dict.get("weighted avg", {}).get("precision", 0.0)), 6),
        "macro_recall": round(float(report_dict.get("macro avg", {}).get("recall", 0.0)), 6),
        "weighted_recall": round(float(report_dict.get("weighted avg", {}).get("recall", 0.0)), 6),
        "elapsed_seconds": round(elapsed_time, 2),
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Text report
    eval_report_txt_path = REPORT_DIR / "evaluation_report.txt"
    with open(eval_report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint     : {ckpt_path}\n")
        f.write(f"Test loss      : {test_loss:.4f}\n")
        f.write(f"Top-1 accuracy : {test_acc:.2f}%\n")
        f.write(f"Top-5 accuracy : {top5_acc:.2f}%\n")
        f.write(f"Evaluated      : {total_n} samples\n")
        f.write(f"Excluded       : {len(excluded_records)} samples\n\n")
        f.write(report_str)

    # 9. Terminal summary
    print("\n" + "=" * 65)
    print("  ISL500 Sign Language Model Evaluation Summary")
    print("=" * 65)
    print(f"  Checkpoint     : {ckpt_path.name}")
    print(f"  Device         : {device}")
    print(f"  Evaluated      : {total_n:,} samples")
    print(f"  Excluded       : {len(excluded_records)} samples")
    print(f"  Classes        : {num_classes}")
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Top-1 Accuracy : {test_acc:.2f}%")
    print(f"  Top-5 Accuracy : {top5_acc:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print("-" * 65)
    print("  Generated Reports:")
    print(f"    - {metrics_json_path}")
    print(f"    - {report_csv_path}")
    print(f"    - {cm_csv_path}")
    print(f"    - {eval_report_txt_path}")
    print("=" * 65)

    return {
        "test_loss": test_loss,
        "accuracy": test_acc,
        "top1_accuracy": test_acc,
        "top5_accuracy": top5_acc,
        "report_str": report_str,
        "confusion_matrix": cm,
        "metrics": metrics,
        "report_df": report_df,
        "cm_df": cm_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ISL500 GRU Model Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (default: models/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest="batch_size",
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test evaluating at most 100 samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt_path=args.checkpoint,
        batch_size=args.batch_size,
        smoke=args.smoke,
    )
