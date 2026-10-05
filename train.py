"""Two-stage transfer learning of the EfficientNet-B0 deepfake classifier.

Stage 1  backbone frozen, only the new dense head is trained (higher LR).
Stage 2  the top N backbone layers are unfrozen and fine-tuned (low LR).

The checkpoint with the best validation ROC-AUC across both stages is kept as
models/best_model.keras (+ models/best_model.weights.h5).

    python train.py
    python train.py --head-epochs 5 --finetune-epochs 10 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import tensorflow as tf

from deepfake_detector.config import (DATA_DIR, MODEL_DIR, MODEL_PATH, REPORTS_DIR, SEED,
                                      WEIGHTS_PATH)
from deepfake_detector.data import balanced_class_weights, count_split, load_split
from deepfake_detector.model import build_model, compile_model, unfreeze_top_layers


def stage_callbacks(log_path: Path, best_so_far: float | None, patience: int):
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True,
        initial_value_threshold=best_so_far, verbose=1,
    )
    return checkpoint, [
        checkpoint,
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(log_path), append=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=60,
                        help="How many of the last backbone layers to fine-tune in stage 2.")
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "history.csv"
    log_path.unlink(missing_ok=True)

    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (training on CPU)")
    counts = {s: count_split(args.data_dir, s) for s in ("train", "val")}
    print("Images per class:", counts)
    class_weight = balanced_class_weights(counts["train"])

    train_ds = load_split(args.data_dir, "train", args.batch_size, shuffle=True)
    val_ds = load_split(args.data_dir, "val", args.batch_size)

    model = build_model()
    started = time.time()

    # ---- Stage 1: train the head on frozen ImageNet features.
    compile_model(model, args.head_lr)
    model.summary(show_trainable=True)
    checkpoint, callbacks = stage_callbacks(log_path, None, args.patience)
    head = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                     class_weight=class_weight, callbacks=callbacks)
    best_auc = checkpoint.best
    head_epochs_run = len(head.history["loss"])

    # ---- Stage 2: fine-tune the top of the backbone.
    unfreeze_top_layers(model, args.unfreeze_layers)
    compile_model(model, args.finetune_lr)
    print(f"\nFine-tuning the last {args.unfreeze_layers} backbone layers "
          f"(best val AUC so far: {best_auc:.4f})")
    checkpoint, callbacks = stage_callbacks(log_path, best_auc, args.patience)
    model.fit(train_ds, validation_data=val_ds,
              epochs=head_epochs_run + args.finetune_epochs,
              initial_epoch=head_epochs_run,
              class_weight=class_weight, callbacks=callbacks)
    best_auc = max(best_auc, checkpoint.best)

    # Weights-only copy: loads on any Keras 3 version (see model.load_trained_model).
    best = keras.models.load_model(MODEL_PATH, compile=False)
    best.save_weights(WEIGHTS_PATH)

    info = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "head_epochs_run": head_epochs_run,
        "best_val_auc": float(best_auc),
        "class_weight": class_weight,
        "images_per_class": counts,
        "training_minutes": round((time.time() - started) / 60, 1),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    (REPORTS_DIR / "training_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nBest validation ROC-AUC: {best_auc:.4f}")
    print(f"Saved {MODEL_PATH} and {WEIGHTS_PATH}. Next: python evaluate.py")


if __name__ == "__main__":
    main()
