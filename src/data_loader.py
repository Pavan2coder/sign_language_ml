import h5py
import numpy as np
from pathlib import Path


H5_DATASET_NAME = "intermediate"


def load_h5_file(file_path: str | Path) -> np.ndarray:
    """
    Load MediaPipe landmark sequence from an H5 file.

    Expected shape:
        (frames, 2, 21, 3)

    Returns:
        NumPy array containing the landmark sequence.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"H5 file not found: {file_path}")

    with h5py.File(file_path, "r") as h5_file:
        if H5_DATASET_NAME not in h5_file:
            raise KeyError(
                f"Dataset '{H5_DATASET_NAME}' not found in {file_path}"
            )

        data = h5_file[H5_DATASET_NAME][:]

    return data.astype(np.float32)


def get_valid_frame_mask(data: np.ndarray) -> np.ndarray:
    """
    Identify frames that contain at least one non-zero landmark.

    Args:
        data: Shape (frames, 2, 21, 3)

    Returns:
        Boolean mask of shape (frames,)
    """

    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D array (frames, 2, 21, 3), got {data.shape}"
        )

    return np.any(data != 0, axis=(1, 2, 3))


def get_valid_frames(data: np.ndarray) -> np.ndarray:
    """
    Remove completely empty/padded frames.
    """

    mask = get_valid_frame_mask(data)
    return data[mask]


def load_h5_with_valid_frames(file_path: str | Path) -> np.ndarray:
    """
    Load an H5 file and remove completely empty frames.
    """

    data = load_h5_file(file_path)
    return get_valid_frames(data)


if __name__ == "__main__":

    test_file = (
        Path(r"D:\ISL-DATA")
        / "Landmarks"
        / "MediaPipe"
        / "ISL_DATA_USER001"
        / "Absent__session106__clip014.h5"
    )

    data = load_h5_file(test_file)

    print("File:", test_file)
    print("Original shape:", data.shape)
    print("Data type:", data.dtype)

    valid_data = get_valid_frames(data)

    print("Valid frames shape:", valid_data.shape)
    print("Total frames:", len(data))
    print("Valid frames:", len(valid_data))
    print("Empty frames:", len(data) - len(valid_data))