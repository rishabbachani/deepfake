"""EfficientNet-B0 transfer-learning classifier (real vs fake)."""

from __future__ import annotations

import keras
from keras import layers

from .config import IMG_SIZE

BACKBONE_NAME = "efficientnetb0"


def build_augmentation() -> keras.Sequential:
    """Light augmentation; only active during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_model(img_size: int = IMG_SIZE, dropout: float = 0.3,
                weights: str | None = "imagenet") -> keras.Model:
    """EfficientNet-B0 backbone (frozen) + custom dense head with a sigmoid output.

    The network takes raw 0-255 RGB pixels: Keras' EfficientNet includes its own
    rescaling/normalisation layers, so no manual preprocessing is needed.
    The output is P(fake).
    """
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    backbone.name = BACKBONE_NAME
    backbone.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="face")
    x = build_augmentation()(inputs)
    # training=False keeps BatchNorm statistics frozen, even after unfreezing.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_fake")(x)

    return keras.Model(inputs, outputs, name="deepfake_efficientnet_b0")


def unfreeze_top_layers(model: keras.Model, num_layers: int) -> None:
    """Make the last `num_layers` backbone layers trainable (BatchNorm stays frozen)."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    cutoff = len(backbone.layers) - num_layers
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def load_trained_model(model_path, weights_path=None) -> keras.Model:
    """Load the saved model.

    If the full .keras file cannot be read (for example it was saved by a newer
    Keras on Colab), rebuild the architecture and load the weights file instead.
    """
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as full_model_error:  # noqa: BLE001 - fall back to weights
        if weights_path is None:
            raise
        try:
            model = build_model(weights=None)
            model.load_weights(weights_path)
            return model
        except Exception:
            raise full_model_error
