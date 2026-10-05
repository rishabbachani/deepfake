"""End-to-end prediction for a single image or video: crop faces, score, aggregate."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .config import (CALIBRATION_PATH, DECISION_THRESHOLD, MODEL_PATH, VIDEO_FRAMES,
                     WEIGHTS_PATH)
from .faces import FaceExtractor, resize_full, sample_video_frames
from .model import load_trained_model


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _encode_preview(face: np.ndarray, size: int = 112) -> str:
    small = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


class DeepfakeDetector:
    """Loads the trained model once and analyses uploaded media."""

    def __init__(self, model_path: str | Path = MODEL_PATH,
                 weights_path: str | Path | None = WEIGHTS_PATH,
                 calibration_path: str | Path | None = CALIBRATION_PATH,
                 threshold: float = DECISION_THRESHOLD,
                 video_frames: int = VIDEO_FRAMES):
        self.model = load_trained_model(model_path, weights_path)
        self.faces = FaceExtractor()
        self.threshold = threshold
        self.video_frames = video_frames

        # Optional Platt scaling fitted on the validation set by evaluate.py.
        self.calibration = None
        if calibration_path and Path(calibration_path).exists():
            self.calibration = json.loads(Path(calibration_path).read_text())

    # ------------------------------------------------------------------ scoring
    def _calibrate(self, p: np.ndarray) -> np.ndarray:
        if not self.calibration:
            return p
        z = self.calibration["a"] * _logit(p) + self.calibration["b"]
        return 1.0 / (1.0 + np.exp(-z))

    def score_faces(self, faces: list[np.ndarray]) -> np.ndarray:
        """P(fake) for each 224x224 RGB face crop."""
        batch = np.stack(faces).astype("float32")
        raw = self.model.predict(batch, verbose=0).ravel()
        return self._calibrate(raw)

    def _verdict(self, scores: np.ndarray, faces: list[np.ndarray], frames: int,
                 face_found: bool, media_type: str) -> dict:
        p_fake = float(np.mean(scores))
        is_fake = p_fake >= self.threshold
        return {
            "media_type": media_type,
            "label": "fake" if is_fake else "real",
            "fake_probability": round(p_fake, 4),
            "confidence": round(p_fake if is_fake else 1.0 - p_fake, 4),
            "threshold": self.threshold,
            "calibrated": self.calibration is not None,
            "faces_scored": len(faces),
            "frames_sampled": frames,
            "face_found": face_found,
            "detector": self.faces.name,
            "frame_scores": [round(float(s), 4) for s in scores],
            "face_previews": [_encode_preview(f) for f in faces[:8]],
        }

    # --------------------------------------------------------------- public API
    def analyze_image(self, image: np.ndarray) -> dict:
        face = self.faces.largest_face(image)
        face_found = face is not None
        if not face_found:
            face = resize_full(image)
        scores = self.score_faces([face])
        return self._verdict(scores, [face], 1, face_found, "image")

    def analyze_video(self, path: str | Path) -> dict:
        frames = sample_video_frames(path, self.video_frames)
        if not frames:
            raise ValueError("Could not read any frames from the video.")

        faces = [f for f in self.faces.largest_faces(frames) if f is not None]
        face_found = bool(faces)
        if not face_found:
            faces = [resize_full(fr) for fr in frames]
        scores = self.score_faces(faces)
        return self._verdict(scores, faces, len(frames), face_found, "video")


def describe(result: dict) -> str:
    """One-line human-readable summary of a result dict."""
    label = "FAKE" if result["label"] == "fake" else "REAL"
    note = "" if result["face_found"] else "  (no face detected - scored the whole frame)"
    return (f"{label}  confidence {result['confidence']:.1%}  "
            f"P(fake)={result['fake_probability']:.3f}  "
            f"faces={result['faces_scored']} frames={result['frames_sampled']}{note}")

