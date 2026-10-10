"""
preprocessing.py
================
STEP 3 of the ISL Sign Language Recognition ML Pipeline:
Landmark Sequence Preprocessing.

What this module does
---------------------
Given a raw landmark array of shape (frames, 2, 21, 3) from load_h5_file(),
it produces a fixed-length float32 feature array ready to feed into a model.

Pipeline for one sample
-----------------------
  raw array (F, 2, 21, 3)
       │
       ▼
  1. remove_empty_frames()
       Keep only frames where at least one hand is present (non-zero).
       Frames where BOTH hands are all-zero are treated as padding artefacts
       and discarded. Frames where exactly ONE hand is absent are kept — a
       missing hand is meaningful silence, not noise.
       → (F', 2, 21, 3)  where F' ≥ 1
       │
       ▼
  2. normalize_landmarks()
       Per-hand, per-frame wrist-centering and scale normalisation.
       Absent hands (all-zero) are left as all-zero — they already convey
       "no hand detected" and there is nothing sensible to normalise.
       Present hands: subtract wrist (landmark 0) then divide by the
       maximum landmark distance from the wrist. Scale is never 0 for
       present hands (verified on the real data).
       → same shape (F', 2, 21, 3), values in roughly [-1, +1]
       │
       ▼
  3. flatten_frame()
       Flatten each frame from (2, 21, 3) to a 126-element vector.
       Temporal order is preserved; hand 0 occupies positions 0–62,
       hand 1 occupies positions 63–125.
       → (F', 126)
       │
       ▼
  4. pad_or_truncate()
       Make every sequence exactly MAX_SEQ_LEN frames long.
       Truncation: keep the FIRST MAX_SEQ_LEN frames (preserve sign onset).
       Padding:    append zero rows after the last real frame (post-padding).
       Zero padding is safe because the model can learn to ignore it, and
       it is distinct from normalised landmark values.
       → (MAX_SEQ_LEN, 126)
       │
       ▼
  5. preprocess_sample()
       Thin wrapper that runs steps 1-4 and verifies the output shape.
       Returns np.ndarray of shape (MAX_SEQ_LEN, FEATURE_DIM).

Label encoding
--------------
  build_label_encoder(train_df)
       Build a {label_string → integer_id} mapping from training labels.
       The mapping is alphabetically sorted so it is deterministic and
       reproducible. Returns a plain dict; no external library required.

  encode_labels(labels, label_to_id)
       Convert a sequence of label strings to a NumPy int64 array.

  decode_labels(ids, id_to_label)
       Convert integer IDs back to label strings (for display / eval).

Public API summary
------------------
  preprocess_sample(raw_array, max_seq_len=MAX_SEQ_LEN)   → np.ndarray (T, 126)
  build_label_encoder(train_df)                           → dict[str, int]
  encode_labels(labels, label_to_id)                      → np.ndarray int64
  decode_labels(ids, id_to_label)                         → list[str]

Smoke-test (run with:  python src/preprocessing.py)
----------------------------------------------------
  Processes SMOKE_N samples from the training split (default 10).
  Prints shape, dtype, finite-check, label IDs, and a brief stats summary.
  Does NOT load the whole dataset.

Key design decisions (explained for learners)
---------------------------------------------
• WHY remove fully-empty frames?
  Some clips are padded at the start or end with all-zero frames. These
  add no information and distort sequence statistics. We drop them while
  KEEPING frames where only one hand is absent — that absence is part of
  the sign's choreography.

• WHY wrist-centering?
  Different signers hold their hands at different positions in the camera
  frame. Centering on the wrist makes the model focus on hand shape and
  motion rather than absolute screen position.

• WHY scale normalisation?
  People have different hand sizes and record at different distances.
  Dividing by the maximum landmark spread makes the representation scale-
  invariant while preserving relative finger positions.

• WHY post-padding with zeros?
  Pre-padding (padding at the front) would shift all sign onsets, making
  temporal alignment harder. Post-padding leaves the sign signal at the
  start of the sequence, which is more natural for recurrent models.

• WHY truncate from the end?
  Sign onset is at the beginning. Cutting from the end loses only the
  return-to-rest motion, which carries less class-discriminative information.

• WHY alphabetical label IDs?
  Sorted order is stable across runs, machines, and Python versions,
  unlike hash-based orderings. Any new clip of a known class always gets
  the same ID.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent   # .../sign-lan/src
_PROJECT_ROOT = _SCRIPT_DIR.parent                # .../sign-lan
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project config ────────────────────────────────────────────────────────────
from src.config import (
    MAX_SEQ_LEN,   # 150 — fixed temporal length after pad/truncate
    FEATURE_DIM,   # 126 — 2 hands × 21 landmarks × 3 coords
    NUM_CLASSES,   # 500
)

# ── Data loader (Step 2) ──────────────────────────────────────────────────────
from src.data_loader import load_mapping, load_h5_file


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — remove fully-empty frames
# ──────────────────────────────────────────────────────────────────────────────

def remove_empty_frames(data: np.ndarray) -> np.ndarray:
    """
    Drop frames where BOTH hands are entirely zero.

    Background
    ----------
    MediaPipe outputs all-zero landmarks for a hand when it is not
    detected in a frame.  When BOTH hands are zero the frame carries
    no information and is usually a padding artefact at the clip boundary.

    When only ONE hand is zero the frame is kept: the absent hand is
    meaningful (the sign may use only the other hand at that moment).

    Args:
        data : np.ndarray, shape (frames, 2, 21, 3), dtype float32.

    Returns:
        np.ndarray, shape (F', 2, 21, 3), F' ≥ 1.
        If all frames are empty (should not happen — verified on the data),
        the original array is returned unchanged to avoid crashing downstream.

    Raises:
        ValueError : if data is not 4-D.
    """
    if data.ndim != 4:
        raise ValueError(
            f"remove_empty_frames expects shape (F, 2, 21, 3), got {data.shape}"
        )

    # Check whether each hand is completely absent in each frame.
    # axis=(2, 3) collapses the (21, 3) landmark axes.
    hand0_absent = np.all(data[:, 0, :, :] == 0, axis=(1, 2))  # (F,) bool
    hand1_absent = np.all(data[:, 1, :, :] == 0, axis=(1, 2))  # (F,) bool

    # A frame is "empty" only when BOTH hands are absent
    both_absent = hand0_absent & hand1_absent

    # Keep frames where at least one hand is present
    keep_mask = ~both_absent

    if not keep_mask.any():
        # Defensive fallback: this should never happen (verified on data)
        return data

    return data[keep_mask]


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — per-hand wrist-centering and scale normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_landmarks(data: np.ndarray) -> np.ndarray:
    """
    Normalise MediaPipe hand landmarks: wrist-centering + scale normalisation.

    For each hand in each frame independently:
        1. Translate: subtract landmark 0 (wrist) so the wrist is at origin.
        2. Scale:     divide by the maximum Euclidean distance from the wrist
                      among all 21 landmarks.

    Absent hands (all-zero) are left as all-zero.
    The probe confirmed that scale is never 0 for present hands, so no
    special divide-by-zero guard is needed beyond the absent-hand check.

    Args:
        data : np.ndarray, shape (F, 2, 21, 3), dtype float32.
               Should already have empty frames removed.

    Returns:
        np.ndarray, same shape, dtype float32.
        Values are in approximately [-1, +1]; wrist is exactly 0 for
        present hands.

    Raises:
        ValueError : if data is not 4-D.
    """
    if data.ndim != 4:
        raise ValueError(
            f"normalize_landmarks expects shape (F, 2, 21, 3), got {data.shape}"
        )

    out = data.astype(np.float32).copy()
    F = out.shape[0]

    for h in range(2):
        hand = out[:, h, :, :]              # (F, 21, 3) — view into out

        # Build a mask: True where this hand is present (not all-zero)
        present = ~np.all(hand == 0, axis=(1, 2))  # (F,)

        if not present.any():
            # Hand is never present in this clip — leave as zeros
            continue

        # Work only on present frames to avoid corrupting absent ones
        pf = hand[present]                  # (n_present, 21, 3)

        # Wrist-centering: landmark 0 is the wrist
        wrist   = pf[:, 0:1, :]            # (n_present, 1, 3) — keeps dim
        pf      = pf - wrist               # translate to wrist-origin

        # Scale: maximum distance from wrist across all 21 landmarks
        dists   = np.linalg.norm(pf, axis=-1)        # (n_present, 21)
        scale   = dists.max(axis=-1, keepdims=True)   # (n_present, 1)

        # The probe confirmed scale > 0 for all present frames,
        # but we guard anyway to be safe
        scale   = np.where(scale == 0.0, 1.0, scale)

        pf      = pf / scale[:, np.newaxis, :]        # normalise

        # Write back into the output array
        out[present, h, :, :] = pf

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — flatten each frame to a 1-D feature vector
# ──────────────────────────────────────────────────────────────────────────────

def flatten_frames(data: np.ndarray) -> np.ndarray:
    """
    Flatten each frame from (2, 21, 3) to a 126-element vector.

    The flattening order is C-order (row-major), meaning:
        positions 0–62   → hand 0, all 21 landmarks, x/y/z each
        positions 63–125 → hand 1, all 21 landmarks, x/y/z each

    This order is consistent and must be preserved across train and test.

    Args:
        data : np.ndarray, shape (F, 2, 21, 3).

    Returns:
        np.ndarray, shape (F, 126), dtype float32.

    Raises:
        ValueError : if data is not 4-D or if the flattened feature size
                     does not match FEATURE_DIM (126).
    """
    if data.ndim != 4:
        raise ValueError(
            f"flatten_frames expects shape (F, 2, 21, 3), got {data.shape}"
        )

    F = data.shape[0]
    # Reshape: keep frame axis, collapse the rest
    flat = data.reshape(F, -1).astype(np.float32)

    if flat.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"After flattening expected {FEATURE_DIM} features per frame, "
            f"got {flat.shape[1]}. Check FEATURE_DIM in config.py."
        )

    return flat


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — pad or truncate to a fixed sequence length
# ──────────────────────────────────────────────────────────────────────────────

def pad_or_truncate(seq: np.ndarray, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Make a variable-length sequence exactly max_seq_len frames long.

    Truncation policy  (seq longer than max_seq_len):
        Keep the FIRST max_seq_len frames.
        Rationale: sign onset is at the start; the return-to-rest tail
        carries less class-discriminative information.

    Padding policy  (seq shorter than max_seq_len):
        Append zero rows AFTER the last real frame (post-padding).
        Rationale: pre-padding shifts onset positions, which disrupts
        temporal alignment for recurrent and attention models.
        Zero is safe because normalised landmarks never produce all-zero
        rows for a present hand (wrist would be [0,0,0] but other
        landmarks would be non-zero after centering).

    Args:
        seq         : np.ndarray, shape (F, feature_dim).
        max_seq_len : int, target number of frames (default MAX_SEQ_LEN).

    Returns:
        np.ndarray, shape (max_seq_len, feature_dim), dtype float32.

    Raises:
        ValueError : if seq is not 2-D or is empty.
    """
    if seq.ndim != 2:
        raise ValueError(
            f"pad_or_truncate expects a 2-D array (frames, features), "
            f"got shape {seq.shape}"
        )
    if len(seq) == 0:
        raise ValueError("pad_or_truncate received an empty sequence.")

    F, D = seq.shape

    if F >= max_seq_len:
        # Truncate: keep first max_seq_len frames
        return seq[:max_seq_len].astype(np.float32)

    # Pad: stack real frames + zero rows
    pad_len = max_seq_len - F
    padding = np.zeros((pad_len, D), dtype=np.float32)
    return np.vstack([seq, padding])


# ──────────────────────────────────────────────────────────────────────────────
# Combined pipeline: one sample end-to-end
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_sample(
    raw_array: np.ndarray,
    max_seq_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on one raw landmark array.

    Pipeline:
        remove_empty_frames  →  normalize_landmarks  →
        flatten_frames       →  pad_or_truncate

    Args:
        raw_array   : np.ndarray, shape (frames, 2, 21, 3), float32.
                      As returned by load_h5_file().
        max_seq_len : target sequence length (default MAX_SEQ_LEN = 150).

    Returns:
        np.ndarray, shape (max_seq_len, FEATURE_DIM), dtype float32.
        Guaranteed: no NaN, no Inf, finite values only.

    Raises:
        ValueError  : bad input shape or unexpected output shape.
        RuntimeError: if the output contains non-finite values (indicates
                      a bug in normalization — should not occur on this data).
    """
    # Step 1: discard fully-empty frames
    data = remove_empty_frames(raw_array)

    # Step 2: wrist-center and scale-normalise each present hand
    data = normalize_landmarks(data)

    # Step 3: flatten (F', 2, 21, 3) → (F', 126)
    data = flatten_frames(data)

    # Step 4: pad or truncate to fixed length
    data = pad_or_truncate(data, max_seq_len=max_seq_len)

    # Final shape check
    expected = (max_seq_len, FEATURE_DIM)
    if data.shape != expected:
        raise ValueError(
            f"preprocess_sample: expected output shape {expected}, "
            f"got {data.shape}."
        )

    # Sanity: no NaN / Inf should ever appear
    if not np.all(np.isfinite(data)):
        n_bad = int(np.sum(~np.isfinite(data)))
        raise RuntimeError(
            f"preprocess_sample produced {n_bad} non-finite values. "
            "This indicates a bug in normalization."
        )

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Label encoding
# ──────────────────────────────────────────────────────────────────────────────

def build_label_encoder(train_df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a deterministic {label_string → integer_id} mapping.

    The mapping is built from the TRAINING split only, then reused for
    the test split.  This ensures train and test use identical class IDs.

    Sorted alphabetically so the mapping is the same on every run, on
    every machine, and with every Python version — unlike hash-based
    orderings.

    The integer IDs are 0-indexed: the first label alphabetically gets 0,
    the second gets 1, etc.

    Args:
        train_df : pd.DataFrame returned by load_mapping("train").
                   Must have a 'label' column.

    Returns:
        dict mapping each label string to its integer class ID.
        Example: {"Absent": 0, "Accept": 1, "Accident": 2, ...}

    Raises:
        KeyError : if train_df has no 'label' column.
        ValueError : if the number of unique labels does not equal NUM_CLASSES.
    """
    if "label" not in train_df.columns:
        raise KeyError(
            "build_label_encoder: train_df must have a 'label' column. "
            "Use load_mapping('train') to get the DataFrame."
        )

    # Get sorted unique labels (excluding any junk that slipped through)
    unique_labels = sorted(train_df["label"].dropna().unique().tolist())
    n = len(unique_labels)

    if n != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} unique classes in train split, "
            f"found {n}. "
            "Re-run dataset_analysis.py to check the label distribution."
        )

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_id


def encode_labels(
    labels: List[str],
    label_to_id: Dict[str, int],
) -> np.ndarray:
    """
    Convert a list of label strings to a NumPy int64 array of class IDs.

    Args:
        labels     : list of label strings, e.g. ["Absent", "Accept", ...]
        label_to_id: mapping from build_label_encoder()

    Returns:
        np.ndarray, dtype int64, shape (len(labels),).

    Raises:
        KeyError : if any label is not in label_to_id.
    """
    try:
        ids = [label_to_id[lbl] for lbl in labels]
    except KeyError as e:
        raise KeyError(
            f"Label {e} not found in the label encoder. "
            "Ensure you built the encoder from the training split "
            "and that test labels match training labels exactly."
        ) from e

    return np.array(ids, dtype=np.int64)


def decode_labels(
    ids: np.ndarray,
    id_to_label: Dict[int, str],
) -> List[str]:
    """
    Convert integer class IDs back to label strings.

    Args:
        ids         : array-like of integer IDs.
        id_to_label : reverse mapping, e.g. {0: "Absent", 1: "Accept", ...}
                      Build with:  id_to_label = {v: k for k, v in label_to_id.items()}

    Returns:
        list of label strings.
    """
    return [id_to_label[int(i)] for i in ids]


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (run with:  python src/preprocessing.py)
# ──────────────────────────────────────────────────────────────────────────────

# Number of samples to process in the smoke test.
# Keep this small — the goal is to verify correctness, not speed.
SMOKE_N = 10


if __name__ == "__main__":

    print("=" * 65)
    print("  preprocessing.py  —  STEP 3 smoke-test")
    print(f"  Processing {SMOKE_N} training samples")
    print("=" * 65)

    # ── 1. Load metadata (no H5 files opened yet) ─────────────────────────────
    print("\n[1] Loading train and test mappings...")
    train_df = load_mapping("train")
    test_df  = load_mapping("test")
    print(f"    Train rows : {len(train_df):,}")
    print(f"    Test  rows : {len(test_df):,}")

    # ── 2. Build the label encoder from training labels only ──────────────────
    print("\n[2] Building label encoder from training split...")
    label_to_id = build_label_encoder(train_df)
    id_to_label = {v: k for k, v in label_to_id.items()}

    print(f"    Unique classes    : {len(label_to_id)}")
    print(f"    First 5 entries   : ", end="")
    first5 = list(label_to_id.items())[:5]
    print(", ".join(f"{k!r}→{v}" for k, v in first5))
    print(f"    Last  5 entries   : ", end="")
    last5 = list(label_to_id.items())[-5:]
    print(", ".join(f"{k!r}→{v}" for k, v in last5))

    # ── 3. Process SMOKE_N samples ────────────────────────────────────────────
    print(f"\n[3] Processing first {SMOKE_N} training samples...")
    print(f"    Config: MAX_SEQ_LEN={MAX_SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")
    print()

    results = []   # collect processed arrays for stats
    labels_seen = []

    for i, row in train_df.head(SMOKE_N).iterrows():
        # Load the raw landmark array (one file at a time — lazy loading)
        raw = load_h5_file(row["local_path"])

        # Preprocess
        processed = preprocess_sample(raw)

        # Encode label
        label_id = label_to_id[row["label"]]

        results.append(processed)
        labels_seen.append(row["label"])

        print(
            f"    [{i:>2}] {row['label']:<20} "
            f"raw_frames={raw.shape[0]:>4}  "
            f"→ processed={processed.shape}  "
            f"label_id={label_id:>3}  "
            f"finite={np.all(np.isfinite(processed))}"
        )

    # ── 4. Shape and dtype checks ─────────────────────────────────────────────
    print(f"\n[4] Output verification:")

    expected_shape = (MAX_SEQ_LEN, FEATURE_DIM)
    shapes_ok = all(r.shape == expected_shape for r in results)
    dtypes_ok = all(r.dtype == np.float32     for r in results)
    finite_ok = all(np.all(np.isfinite(r))    for r in results)

    print(f"    All shapes == {expected_shape} : {shapes_ok}")
    print(f"    All dtypes == float32           : {dtypes_ok}")
    print(f"    All values finite               : {finite_ok}")

    # ── 5. Quick value statistics ─────────────────────────────────────────────
    stacked = np.stack(results)   # (SMOKE_N, MAX_SEQ_LEN, FEATURE_DIM)
    print(f"\n[5] Value statistics across {SMOKE_N} processed samples:")
    print(f"    Stacked shape : {stacked.shape}")
    print(f"    Global min    : {stacked.min():.4f}")
    print(f"    Global max    : {stacked.max():.4f}")
    print(f"    Global mean   : {stacked.mean():.4f}")
    print(f"    Global std    : {stacked.std():.4f}")

    # ── 6. Verify test labels are all known ───────────────────────────────────
    print(f"\n[6] Checking test labels are all in the encoder...")
    unknown = set(test_df["label"].unique()) - set(label_to_id.keys())
    print(f"    Unknown test labels : {len(unknown)}")
    if unknown:
        print(f"    Examples: {sorted(unknown)[:5]}")
    else:
        print("    All test labels known — encoder is consistent with test split.")

    # ── 7. Encode and decode round-trip check ─────────────────────────────────
    print(f"\n[7] Encode/decode round-trip on first {SMOKE_N} labels...")
    encoded = encode_labels(labels_seen, label_to_id)
    decoded = decode_labels(encoded, id_to_label)
    roundtrip_ok = (decoded == labels_seen)
    print(f"    Encoded IDs  : {encoded.tolist()}")
    print(f"    Decoded back : {decoded}")
    print(f"    Round-trip OK: {roundtrip_ok}")

    print("\n" + "=" * 65)
    if shapes_ok and dtypes_ok and finite_ok and not unknown and roundtrip_ok:
        print("  Smoke-test PASSED — preprocessing is ready for STEP 4.")
    else:
        print("  Smoke-test FAILED — check output above for details.")
    print("=" * 65)
