from __future__ import annotations

import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path, size=(96, 96)) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def color_histogram(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)


def edge_features(arr: np.ndarray) -> np.ndarray:
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    return np.array(
        [
            float(mag.mean()),
            float(mag.std()),
            float(gray.mean()),
            float(gray.std()),
        ],
        dtype=np.float32,
    )


def spatial_features(arr: np.ndarray, size=(24, 24)) -> np.ndarray:
    image = Image.fromarray((arr * 255).astype("uint8")).resize(size)
    flat = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return flat


def extract_features(image_path) -> np.ndarray:
    arr = load_and_preprocess_image(image_path)
    features = np.concatenate(
        [
            color_histogram(arr, bins=16),
            edge_features(arr),
            spatial_features(arr, size=(24, 24)),
        ]
    )
    return features.astype(np.float32)
