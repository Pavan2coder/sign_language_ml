"""
train.py — STEP 5: GRU classifier + training loop.
Usage: python src/train.py [--check | --smoke | --resume]
       python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""

import argparse, csv, json, random, signal, sys, time
from pathlib import Path
from typing  import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    MODEL_DIR, REPORT_DIR, CHECKPOINT_DIR,
    LATEST_CKPT_NAME, BEST_CKPT_NAME, LABEL_MAP_NAME, HISTORY_CSV_NAME,
    MAX_SEQ_LEN, FEATURE_DIM, NUM_CLASSES, RANDOM_SEED, VAL_FRACTION,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    PATIENCE, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT, FC_HIDDEN_SIZE,
)
from src.data_loader        import load_mapping
from src.preprocessing      import build_label_encoder
from src.feature_extraction import make_dataloaders, compute_seq_lengths

try:
    from tqdm import tqdm; HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(require_cuda=False):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[device] GPU: {p.name}  VRAM: {p.total_memory/1024**3:.1f} GB  SM: {p.major}.{p.minor}")
        return torch.device("cuda:0")
    if require_cuda: raise RuntimeError("CUDA not available.")
    print("[device] CPU mode."); return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class GRUClassifier(nn.Module):
    """2-layer GRU + LayerNorm head. Input (B,150,126) -> logits (B,500)."""

    def __init__(self, input_size=FEATURE_DIM, hidden_size=GRU_HIDDEN_SIZE,
                 num_layers=GRU_NUM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=GRU_DROPOUT, fc_hidden=FC_HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.gru  = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes))

    def forward(self, x, lengths=None):
        if lengths is not None:
            lc = lengths.clamp(min=1, max=x.size(1)).cpu()
            sl, si = lc.sort(descending=True)
            packed = nn.utils.rnn.pack_padded_sequence(
                x[si], sl, batch_first=True, enforce_sorted=True)
            _, h_n = self.gru(packed)
            _, ui  = si.sort(); h_last = h_n[-1][ui]
        else:
            _, h_n = self.gru(x); h_last = h_n[-1]
        return self.head(h_last)


# ── Data split ────────────────────────────────────────────────────────────────

def make_train_val_split(train_df, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """Stratified 85/15 split — uses training data only, never touches test."""
    rng = np.random.default_rng(seed)
    val_idx, tr_idx = [], []
    for _, group in train_df.groupby("label"):
        idx = group.index.tolist(); n_val = max(1, int(len(idx)*val_fraction))
        rng.shuffle(idx); val_idx.extend(idx[:n_val]); tr_idx.extend(idx[n_val:])
    return (train_df.loc[tr_idx].reset_index(drop=True),
            train_df.loc[val_idx].reset_index(drop=True))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _atomic_save(obj, path):
    tmp = path.with_suffix(".tmp"); torch.save(obj, tmp); tmp.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_loss, patience_counter, label_to_id, cfg):
    """Atomically save full training state including RNG for exact resume."""
    _atomic_save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "label_to_id": label_to_id, "config": cfg,
        "rng_state": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Restore model/optimizer/scheduler; always loads RNG states on CPU."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    rng = ckpt.get("rng_state", {})
    if rng.get("python"):  random.setstate(rng["python"])
    if rng.get("numpy"):   np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        t = rng["torch"]
        torch.set_rng_state(t.cpu() if hasattr(t, "cpu") else t)
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu() if hasattr(s, "cpu") else s for s in rng["cuda"]])
    return ckpt


# ── Epoch loop ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train (optimizer provided) or evaluate. Returns (mean_loss, acc%)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_correct = total_n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            logits = model(bx, compute_seq_lengths(bx))
            loss   = criterion(logits, by)
            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss    += loss.item() * len(by)
            total_correct += (logits.argmax(1) == by).sum().item()
            total_n       += len(by)
    return total_loss / total_n, total_correct / total_n * 100.0


def _append_history_row(csv_path, row):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)


# ── Full training function ────────────────────────────────────────────────────

def train(
    epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
    resume=False, smoke=False,
):
    """
    Full training loop — setup, data, model, epoch loop, checkpointing.

    Validation split comes from training data only (15%).
    Test CSV is never loaded here.
    Checkpoints written atomically after every epoch.
    Ctrl+C saves a checkpoint before exiting.
    """
    set_seeds(RANDOM_SEED)
    device = get_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ISL500 Sign Language Recognition — Training")
    print(f"{'='*60}")
    if smoke: print("  [SMOKE TEST MODE] — 2 epochs, 200 samples")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}  device={device}\n")

    # Load data
    print("[1] Loading train mapping...")
    train_df = load_mapping("train")
    print(f"    Total: {len(train_df):,}")

    print("[2] Building label encoder...")
    label_to_id = build_label_encoder(train_df)
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"    Classes: {len(label_to_id)}")

    if smoke:
        train_df = train_df.sample(
            min(200, len(train_df)), random_state=RANDOM_SEED
        ).reset_index(drop=True)
        epochs = 2
        print(f"    [smoke] {len(train_df)} samples, 2 epochs")

    print("[3] Stratified train/val split (15% val)...")
    train_sub, val_sub = make_train_val_split(train_df)
    print(f"    Train: {len(train_sub):,}   Val: {len(val_sub):,}")

    print("[4] DataLoaders...")
    train_loader, val_loader = make_dataloaders(
        train_sub, val_sub, label_to_id, batch_size, num_workers=0)
    print(f"    Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    print("[5] Building model...")
    model     = GRUClassifier().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0; best_val_loss = float("inf"); patience_counter = 0
    history_csv = REPORT_DIR / HISTORY_CSV_NAME
    latest_ckpt = CHECKPOINT_DIR / LATEST_CKPT_NAME

    train_cfg = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "max_seq_len": MAX_SEQ_LEN,
        "feature_dim": FEATURE_DIM, "num_classes": NUM_CLASSES,
        "gru_hidden": GRU_HIDDEN_SIZE, "gru_layers": GRU_NUM_LAYERS,
        "fc_hidden": FC_HIDDEN_SIZE, "seed": RANDOM_SEED,
    }

    if resume:
        if not latest_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {latest_ckpt}. Run without --resume to start fresh.")
        print(f"[6] Resuming: {latest_ckpt}")
        ckpt             = load_checkpoint(latest_ckpt, model, optimizer, scheduler, device)
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        label_to_id      = ckpt["label_to_id"]
        id_to_label      = {v: k for k, v in label_to_id.items()}
        print(f"    Epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        print("[6] Fresh training run")

    lm_path = CHECKPOINT_DIR / LABEL_MAP_NAME
    with open(lm_path, "w") as f:
        json.dump(label_to_id, f, indent=2)
    print(f"    Label map: {lm_path}")

    # ── Ctrl+C handler ─────────────────────────────────────────────────────────
    interrupted = False
    def _sigint(sig, frame):
        nonlocal interrupted
        print("\n\n[!] Ctrl+C — saving checkpoint and exiting...")
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Epochs {start_epoch+1} to {epochs}  |  patience={PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        if interrupted:
            break

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)

        if interrupted:
            break

        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        mem_str = ""
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_reserved(0) / 1024**2
            mem_str = f"  GPU {mem_mb:.0f} MB"

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:>3}/{epochs}"
            f"  loss {train_loss:.4f}  acc {train_acc:.1f}%"
            f"  |  val_loss {val_loss:.4f}  val_acc {val_acc:.1f}%"
            f"  |  lr {current_lr:.2e}  {elapsed:.1f}s{mem_str}"
        )

        _append_history_row(history_csv, {
            "epoch": epoch+1, "train_loss": round(train_loss,6),
            "train_acc": round(train_acc,4), "val_loss": round(val_loss,6),
            "val_acc": round(val_acc,4), "lr": current_lr,
            "epoch_time": round(elapsed,2),
        })

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_path = CHECKPOINT_DIR / BEST_CKPT_NAME
            save_checkpoint(best_path, model, optimizer, scheduler, epoch,
                            best_val_loss, patience_counter, label_to_id, train_cfg)
            print(f"  -> Best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Latest (resumable) checkpoint
        save_checkpoint(latest_ckpt, model, optimizer, scheduler, epoch,
                        best_val_loss, patience_counter, label_to_id, train_cfg)

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n[Early stopping] No improvement for {PATIENCE} epochs.")
            break

    # ── Wrap up ────────────────────────────────────────────────────────────────
    if interrupted:
        save_checkpoint(latest_ckpt, model, optimizer, scheduler,
                        epoch, best_val_loss, patience_counter, label_to_id, train_cfg)
        print(f"\n[!] Checkpoint saved: {latest_ckpt}")
        print("    Resume:  python src/train.py --resume")
    else:
        print(f"\n{'='*60}")
        print("  Training complete.")
        print(f"  Best val loss : {best_val_loss:.4f}")
        print(f"  History CSV   : {history_csv}")
        print(f"  Best ckpt     : {CHECKPOINT_DIR / BEST_CKPT_NAME}")
        print(f"  Evaluate with : python src/evaluate.py")
        print(f"{'='*60}")
