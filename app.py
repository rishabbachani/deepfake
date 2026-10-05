"""Flask web app: upload an image or video, get Real/Fake with a confidence score.

    python app.py            # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from deepfake_detector.config import (IMAGE_EXTENSIONS, MODEL_PATH, VIDEO_EXTENSIONS,
                                      VIDEO_FRAMES)
from deepfake_detector.faces import decode_image_rgb

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

_detector = None
_detector_error: str | None = None
_lock = threading.Lock()


def get_detector():
    """Load the model once, on first use."""
    global _detector, _detector_error
    with _lock:
        if _detector is None and _detector_error is None:
            try:
                from deepfake_detector.inference import DeepfakeDetector
                _detector = DeepfakeDetector()
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                _detector_error = (f"Could not load the model from {MODEL_PATH}. "
                                   f"Train it first (python train.py). Details: {exc}")
        return _detector


@app.get("/")
def index():
    return render_template(
        "index.html",
        image_types=",".join(sorted(IMAGE_EXTENSIONS)),
        video_types=",".join(sorted(VIDEO_EXTENSIONS)),
        video_frames=VIDEO_FRAMES,
    )


@app.get("/api/health")
def health():
    detector = get_detector()
    return jsonify({"model_loaded": detector is not None, "error": _detector_error,
                    "calibrated": bool(detector and detector.calibration)})


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "No file uploaded."}), 400

    ext = Path(upload.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    detector = get_detector()
    if detector is None:
        return jsonify({"error": _detector_error}), 503

    try:
        with _lock:
            if ext in IMAGE_EXTENSIONS:
                image = decode_image_rgb(upload.read())
                if image is None:
                    return jsonify({"error": "Could not decode the image."}), 400
                result = detector.analyze_image(image)
            else:
                # OpenCV needs a real file path to read videos.
                fd, tmp_path = tempfile.mkstemp(suffix=ext)
                os.close(fd)
                try:
                    upload.save(tmp_path)
                    result = detector.analyze_video(tmp_path)
                finally:
                    os.remove(tmp_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result["filename"] = upload.filename
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File is larger than 200 MB."}), 413


if __name__ == "__main__":
    get_detector()  # load the model before the first request
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
