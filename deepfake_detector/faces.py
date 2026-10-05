"""Face isolation (MTCNN) and media loading helpers.

Every image that reaches the classifier - during training and in the web app -
goes through the same crop: detect the face, expand the box into a square with
a margin, and resize to IMG_SIZE x IMG_SIZE. Keeping one code path for both is
what makes the test-set numbers meaningful for real uploads.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import FACE_MARGIN, IMG_SIZE, MIN_FACE_CONFIDENCE


def load_image_rgb(path: str | Path) -> np.ndarray | None:
    """Read an image file as an RGB uint8 array, or None if it cannot be decoded."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return decode_image_rgb(data.tobytes())


def decode_image_rgb(raw: bytes) -> np.ndarray | None:
    """Decode encoded image bytes (jpg/png/...) to an RGB uint8 array."""
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_video_frames(path: str | Path, num_frames: int) -> list[np.ndarray]:
    """Return up to `num_frames` RGB frames spread evenly across the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []

    if total > 0:
        wanted = np.unique(np.linspace(0, total - 1, num_frames).astype(int))
        for index in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        # Some containers do not report a frame count; read everything, then subsample.
        all_frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            all_frames.append(bgr)
        if all_frames:
            wanted = np.unique(np.linspace(0, len(all_frames) - 1, num_frames).astype(int))
            frames = [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB) for i in wanted]

    cap.release()
    return frames


def square_crop(image: np.ndarray, box: tuple[int, int, int, int],
                margin: float = FACE_MARGIN, size: int = IMG_SIZE) -> np.ndarray:
    """Crop a square around an (x, y, w, h) box, expanded by `margin`, resized to size x size.

    Regions that fall outside the frame are zero-padded so the face stays centred.
    """
    x, y, w, h = box
    side = int(round(max(w, h) * (1.0 + margin)))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side

    img_h, img_w = image.shape[:2]
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - img_w), max(0, y1 - img_h)

    crop = image[max(0, y0):min(img_h, y1), max(0, x0):min(img_w, x1)]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def resize_full(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Fallback when no face is found: centre-crop to a square and resize."""
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return cv2.resize(image[top:top + side, left:left + side], (size, size),
                      interpolation=cv2.INTER_AREA)


class FaceExtractor:
    """Thin wrapper around MTCNN that returns ready-to-classify face crops."""

    name = "MTCNN"

    def __init__(self, min_confidence: float = MIN_FACE_CONFIDENCE,
                 margin: float = FACE_MARGIN, size: int = IMG_SIZE):
        # Imported lazily: MTCNN pulls in TensorFlow, which is slow to import.
        import tensorflow as tf
        from mtcnn import MTCNN

        device = "GPU:0" if tf.config.list_physical_devices("GPU") else "CPU:0"
        try:
            self._detector = MTCNN(device=device)  # mtcnn >= 1.0
        except TypeError:
            self._detector = MTCNN()  # mtcnn 0.1.x (Python 3.9)
        self._supports_batch = True
        self.min_confidence = min_confidence
        self.margin = margin
        self.size = size

    def _largest(self, detections: list[dict]) -> dict | None:
        faces = [f for f in detections if f.get("confidence", 0.0) >= self.min_confidence]
        return max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None

    def largest_face(self, image: np.ndarray) -> np.ndarray | None:
        """Crop of the largest detected face, or None if no face is found."""
        face = self._largest(self._detector.detect_faces(image))
        return None if face is None else square_crop(image, face["box"], self.margin, self.size)

    def largest_faces(self, images: list[np.ndarray]) -> list[np.ndarray | None]:
        """Batched version of largest_face (about 3x faster for same-sized images)."""
        if not self._supports_batch or len(images) < 2 or len({im.shape for im in images}) > 1:
            return [self.largest_face(im) for im in images]
        try:
            batch = self._detector.detect_faces(images)
        except Exception:  # noqa: BLE001 - older mtcnn only accepts one image
            self._supports_batch = False
            return [self.largest_face(im) for im in images]
        crops = []
        for image, detections in zip(images, batch):
            face = self._largest(detections)
            crops.append(None if face is None else
                         square_crop(image, face["box"], self.margin, self.size))
        return crops
