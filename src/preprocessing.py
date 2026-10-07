import numpy as np


def normalize_landmarks(data: np.ndarray) -> np.ndarray:
    """
    Normalize MediaPipe hand landmarks.

    Input:
        Shape: (frames, 2, 21, 3)

    Steps:
        1. Translate each hand so the wrist (landmark 0)
           becomes the origin.
        2. Normalize each hand by its maximum landmark distance
           from the wrist.

    Returns:
        Normalized landmarks with the same shape.
    """

    if data.ndim != 4:
        raise ValueError(
            f"Expected shape (frames, 2, 21, 3), got {data.shape}"
        )

    normalized = data.astype(np.float32).copy()

    # Wrist = landmark 0
    wrist = normalized[:, :, 0:1, :]

    # Move wrist to origin
    normalized = normalized - wrist

    # Calculate distance of every landmark from wrist
    distances = np.linalg.norm(normalized, axis=-1)

    # Maximum distance for each frame and hand
    scale = np.max(distances, axis=-1, keepdims=True)

    # Prevent division by zero
    scale = np.where(scale == 0, 1.0, scale)

    # Add coordinate dimension for broadcasting
    scale = scale[..., np.newaxis]

    normalized = normalized / scale

    return normalized


def normalize_sequence(data: np.ndarray) -> np.ndarray:
    """
    Convenience function for validating and normalizing
    a landmark sequence.
    """

    if data.size == 0:
        raise ValueError("Cannot normalize an empty sequence.")

    return normalize_landmarks(data)


if __name__ == "__main__":

    from data_loader import load_h5_with_valid_frames

    test_file = (
        r"D:\ISL-DATA\Landmarks\MediaPipe"
        r"\ISL_DATA_USER001"
        r"\Absent__session106__clip014.h5"
    )

    data = load_h5_with_valid_frames(test_file)

    print("Before normalization:")
    print("Shape:", data.shape)
    print("Min:", data.min())
    print("Max:", data.max())

    normalized = normalize_sequence(data)

    print("\nAfter normalization:")
    print("Shape:", normalized.shape)
    print("Min:", normalized.min())
    print("Max:", normalized.max())

    # Check wrist is approximately zero
    wrist_values = normalized[:, :, 0, :]

    print(
        "\nMaximum absolute wrist value:",
        np.max(np.abs(wrist_values))
    )