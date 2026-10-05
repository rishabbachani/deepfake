"""Project-wide settings shared by data preparation, training, evaluation and the web app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input resolution expected by EfficientNet-B0.
IMG_SIZE = 224

# Label order matters: index 0 = real, index 1 = fake.
# The sigmoid output of the model is therefore P(fake).
CLASS_NAMES = ("real", "fake")

SEED = 42

# Default locations (all relative to the project root).
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "best_model.keras"
WEIGHTS_PATH = MODEL_DIR / "best_model.weights.h5"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"

# Face cropping: the MTCNN box is expanded by this fraction so the crop keeps
# the blending boundary around the face (where many manipulation artefacts live).
FACE_MARGIN = 0.30
MIN_FACE_CONFIDENCE = 0.90

# Number of evenly spaced frames scored per uploaded video.
VIDEO_FRAMES = 16

# P(fake) above this value is reported as "fake".
DECISION_THRESHOLD = 0.5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
