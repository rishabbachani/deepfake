"""tf.data pipelines over the prepared face-crop dataset.

Expected layout (produced by prepare_dataset.py):

    data/processed/
        train/{real,fake}/*.jpg
        val/{real,fake}/*.jpg
        test/{real,fake}/*.jpg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

from .config import CLASS_NAMES, IMG_SIZE, IMAGE_EXTENSIONS, SEED


def load_split(data_dir: str | Path, split: str, batch_size: int = 32,
               shuffle: bool = False) -> tf.data.Dataset:
    """Batched (image, label) dataset for one split. Labels: real=0, fake=1."""
    ds = keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def count_split(data_dir: str | Path, split: str) -> dict[str, int]:
    """Number of images per class in a split."""
    root = Path(data_dir) / split
    return {
        name: sum(1 for p in (root / name).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        for name in CLASS_NAMES
    }


def balanced_class_weights(counts: dict[str, int]) -> dict[int, float]:
    """Inverse-frequency class weights (all 1.0 when the split is balanced)."""
    values = np.array([counts[name] for name in CLASS_NAMES], dtype=float)
    weights = values.sum() / (len(values) * np.maximum(values, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def labels_of(ds: tf.data.Dataset) -> np.ndarray:
    """Concatenate the labels of an unshuffled dataset into a flat array."""
    return np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
