"""Command-line prediction for one or more images / videos.

    python predict.py path/to/face.jpg
    python predict.py clip.mp4 another.png --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepfake_detector.config import VIDEO_EXTENSIONS
from deepfake_detector.faces import load_image_rgb
from deepfake_detector.inference import DeepfakeDetector, describe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args()

    detector = DeepfakeDetector()
    for path in args.paths:
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            result = detector.analyze_video(path)
        else:
            image = load_image_rgb(path)
            if image is None:
                print(f"{path}: could not read image")
                continue
            result = detector.analyze_image(image)

        if args.json:
            result = {k: v for k, v in result.items() if k != "face_previews"}
            print(json.dumps({"file": str(path), **result}, indent=2))
        else:
            print(f"{path}: {describe(result)}")


if __name__ == "__main__":
    main()
