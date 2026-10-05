"""Evaluate the trained model on the held-out test split.

Produces, in reports/:
    metrics.json               accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    classification_report.txt  per-class precision / recall / F1
    confusion_matrix.png
    roc_curve.png
    training_curves.png        (from reports/history.csv written by train.py)
    calibration.png            reliability diagram before / after calibration

It also fits Platt scaling on the validation split (never the test split) and
saves it to models/calibration.json; the web app uses it automatically so the
"confidence" it shows is a calibrated probability.

    python evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from deepfake_detector.config import (CALIBRATION_PATH, CLASS_NAMES, DATA_DIR,  # noqa: E402
                                      DECISION_THRESHOLD, MODEL_PATH, REPORTS_DIR,
                                      WEIGHTS_PATH)
from deepfake_detector.data import labels_of, load_split  # noqa: E402
from deepfake_detector.model import load_trained_model  # noqa: E402


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def predict_split(model, data_dir: Path, split: str, batch_size: int):
    ds = load_split(data_dir, split, batch_size, shuffle=False)
    return labels_of(ds), model.predict(ds, verbose=1).ravel()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> dict:
    lr = LogisticRegression(C=1.0)
    lr.fit(logit(p_val).reshape(-1, 1), y_val)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(p: np.ndarray, params: dict) -> np.ndarray:
    return 1 / (1 + np.exp(-(params["a"] * logit(p) + params["b"])))


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_score(y, pred, pos_label=idx, zero_division=0)),
            "recall": float(recall_score(y, pred, pos_label=idx, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=idx, zero_division=0)),
            "support": int((y == idx).sum()),
        }
    return {
        "threshold": threshold,
        "n_images": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CLASS_NAMES), "rows_actual_cols_predicted": cm.tolist()},
    }


# ------------------------------------------------------------------------ plots
def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {cm.sum()} test images")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(y: np.ndarray, p: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#1f6f43", lw=2, label=f"EfficientNet-B0 (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve - test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history_csv: Path, info_json: Path, path: Path) -> bool:
    if not history_csv.exists():
        return False
    rows = np.genfromtxt(history_csv, delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    epochs = rows["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label, style in [("accuracy", "Train accuracy", "--"),
                              ("val_accuracy", "Val accuracy", "-"),
                              ("val_auc", "Val ROC-AUC", "-")]:
        axes[0].plot(epochs, rows[key], style, marker="o", label=label)
    axes[1].plot(epochs, rows["loss"], "--", marker="o", label="Train loss")
    axes[1].plot(epochs, rows["val_loss"], "-", marker="o", label="Val loss")

    if info_json.exists():
        boundary = json.loads(info_json.read_text()).get("head_epochs_run")
        if boundary:
            for ax in axes:
                ax.axvline(boundary + 0.5, color="grey", ls=":", lw=1)
                ax.text(boundary + 0.6, ax.get_ylim()[1], "fine-tuning", va="top",
                        fontsize=8, color="grey")
    axes[0].set_title("Accuracy / AUC")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_calibration(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for p, label in [(p_raw, "Raw sigmoid"), (p_cal, "Platt-calibrated")]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Perfect")
    ax.set_xlabel("Predicted P(fake)")
    ax.set_ylabel("Observed fraction fake")
    ax.set_title("Reliability diagram - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_trained_model(args.model, WEIGHTS_PATH)

    print("Scoring validation split (for calibration)...")
    y_val, p_val = predict_split(model, args.data_dir, "val", args.batch_size)
    print("Scoring test split...")
    y_test, p_test = predict_split(model, args.data_dir, "test", args.batch_size)

    platt = fit_platt(y_val, p_val)
    CALIBRATION_PATH.write_text(json.dumps(platt, indent=2))
    p_test_cal = apply_platt(p_test, platt)

    metrics = compute_metrics(y_test, p_test, args.threshold)
    metrics["calibration"] = {
        "method": "Platt scaling fitted on validation split",
        "params": platt,
        "brier_raw": float(brier_score_loss(y_test, p_test)),
        "brier_calibrated": float(brier_score_loss(y_test, p_test_cal)),
        "ece_raw": expected_calibration_error(y_test, p_test),
        "ece_calibrated": expected_calibration_error(y_test, p_test_cal),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pred = (p_test >= args.threshold).astype(int)
    report = classification_report(y_test, pred, target_names=list(CLASS_NAMES), digits=4)
    (REPORTS_DIR / "classification_report.txt").write_text(report)

    cm = np.array(metrics["confusion_matrix"]["rows_actual_cols_predicted"])
    plot_confusion(cm, REPORTS_DIR / "confusion_matrix.png")
    plot_roc(y_test, p_test, REPORTS_DIR / "roc_curve.png")
    plot_calibration(y_test, p_test, p_test_cal, REPORTS_DIR / "calibration.png")
    plot_history(REPORTS_DIR / "history.csv", REPORTS_DIR / "training_info.json",
                 REPORTS_DIR / "training_curves.png")

    print("\n" + report)
    print(f"Accuracy  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC   {metrics['roc_auc']:.4f}")
    print(f"Macro F1  {metrics['macro_f1']:.4f}")
    print(f"Confusion matrix (rows = actual {CLASS_NAMES}, cols = predicted):\n{cm}")
    c = metrics["calibration"]
    print(f"Calibration ECE {c['ece_raw']:.4f} -> {c['ece_calibrated']:.4f}, "
          f"Brier {c['brier_raw']:.4f} -> {c['brier_calibrated']:.4f}")
    print(f"\nAll outputs saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
