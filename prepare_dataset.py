"""Build the face-crop dataset used for training and evaluation.

Reads a raw dataset, isolates the face in every image (or video frame) with
MTCNN, resizes it to 224x224 and writes:

    data/processed/{train,val,test}/{real,fake}/*.jpg
    data/processed/summary.json

Supported source layouts (auto-detected):

  presplit  Folders already split, e.g. the Kaggle "140k Real and Fake Faces"
            dataset:  <source>/.../{train,valid|val,test}/{real,fake}/*.jpg
  flat      <source>/{real,fake}/*.jpg - split here into train/val/test.
  dfdc      DFDC-style video folder: <source>/*.mp4 + metadata.json with
            {"video.mp4": {"label": "REAL"|"FAKE"}}. Frames are sampled from
            each video and the split is done per video, so frames from the same
            video never land in two splits (no leakage).

Examples:

    python prepare_dataset.py --source ~/datasets/real_vs_fake
    python prepare_dataset.py --source ~/datasets/real_vs_fake --train 7000 --val 1500 --test 1500
    python prepare_dataset.py --source train_sample_videos --frames-per-video 10
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2

from deepfake_detector.config import (CLASS_NAMES, DATA_DIR, IMAGE_EXTENSIONS, SEED,
                                      VIDEO_EXTENSIONS)
from deepfake_detector.faces import (FaceExtractor, load_image_rgb, resize_full,
                                     sample_video_frames)

SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": ("train", "training"),
                 "val": ("val", "valid", "validation"),
                 "test": ("test", "testing")}


# --------------------------------------------------------------------- discovery
def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _child(folder: Path, names) -> Path | None:
    for child in folder.iterdir() if folder.is_dir() else []:
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def _find_presplit_root(source: Path, max_depth: int = 4) -> Path | None:
    """Find a folder that has train/val/test sub-folders, each with real/ and fake/."""
    candidates = [source] + [p for p in source.rglob("*") if p.is_dir()
                             and len(p.relative_to(source).parts) <= max_depth]
    for folder in candidates:
        splits = [_child(folder, SPLIT_ALIASES[s]) for s in SPLITS]
        if all(splits) and all(_child(s, {"real"}) and _child(s, {"fake"}) for s in splits):
            return folder
    return None


def detect_layout(source: Path) -> str:
    if (source / "metadata.json").exists():
        return "dfdc"
    if _find_presplit_root(source):
        return "presplit"
    if _child(source, {"real"}) and _child(source, {"fake"}):
        return "flat"
    raise SystemExit(
        f"Could not recognise the layout of {source}.\n"
        "Expected {train,val,test}/{real,fake}/, or {real,fake}/, or videos + metadata.json.")


# --------------------------------------------------------------------- planning
Plan = dict  # split -> class -> list of source items


def plan_presplit(source: Path, sizes: dict[str, int], rng: random.Random) -> Plan:
    root = _find_presplit_root(source)
    print(f"Using pre-split dataset at {root}")
    plan: Plan = {}
    for split in SPLITS:
        split_dir = _child(root, SPLIT_ALIASES[split])
        plan[split] = {}
        for cls in CLASS_NAMES:
            files = _images_in(_child(split_dir, {cls}))
            rng.shuffle(files)
            plan[split][cls] = files[:sizes[split]] if sizes[split] > 0 else files
    return plan


def plan_flat(source: Path, sizes: dict[str, int], ratios: tuple[float, float, float],
              rng: random.Random) -> Plan:
    plan: Plan = {s: {} for s in SPLITS}
    for cls in CLASS_NAMES:
        files = _images_in(_child(source, {cls}))
        rng.shuffle(files)
        if all(sizes[s] > 0 for s in SPLITS):
            counts = [sizes[s] for s in SPLITS]
        else:
            n = len(files)
            counts = [int(n * ratios[0]), int(n * ratios[1])]
            counts.append(n - sum(counts))
        start = 0
        for split, count in zip(SPLITS, counts):
            plan[split][cls] = files[start:start + count]
            start += count
    return plan


def plan_dfdc(source: Path, ratios: tuple[float, float, float], rng: random.Random) -> Plan:
    meta = json.loads((source / "metadata.json").read_text())
    by_class: dict[str, list[Path]] = {c: [] for c in CLASS_NAMES}
    for name, info in meta.items():
        path = source / name
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.exists():
            by_class[info["label"].lower()].append(path)

    plan: Plan = {s: {} for s in SPLITS}
    for cls, videos in by_class.items():
        videos.sort()
        rng.shuffle(videos)
        n = len(videos)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        plan["train"][cls] = videos[:n_train]
        plan["val"][cls] = videos[n_train:n_train + n_val]
        plan["test"][cls] = videos[n_train + n_val:]
    return plan


# ------------------------------------------------------------------- processing
def _write(face, out_path: Path) -> None:
    cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def _load_sources(item: Path, layout: str, frames_per_video: int) -> list[tuple[str, object]]:
    """(output name, RGB image) pairs for one dataset item (an image or a video)."""
    if layout == "dfdc":
        return [(f"{item.stem}_f{i:02d}", frame) for i, frame in
                enumerate(sample_video_frames(item, frames_per_video))]
    image = load_image_rgb(item)
    return [] if image is None else [(f"{item.parent.name}_{item.stem}", image)]


def process(plan: Plan, out_dir: Path, layout: str, extractor: FaceExtractor | None,
            frames_per_video: int, batch_size: int = 32) -> dict:
    stats = {s: Counter() for s in SPLITS}
    total = sum(len(items) for split in plan.values() for items in split.values())
    done, started = 0, time.time()

    for split in SPLITS:
        for cls in CLASS_NAMES:
            target = out_dir / split / cls
            target.mkdir(parents=True, exist_ok=True)
            items = plan[split][cls]

            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                sources = []
                for item in chunk:
                    loaded = _load_sources(item, layout, frames_per_video)
                    if not loaded:
                        stats[split]["unreadable"] += 1
                    sources.extend(loaded)

                images = [image for _, image in sources]
                faces = (extractor.largest_faces(images) if extractor
                         else [resize_full(image) for image in images])
                for (name, _), face in zip(sources, faces):
                    if face is None:
                        stats[split]["no_face"] += 1
                        continue
                    _write(face, target / f"{name}.jpg")
                    stats[split][cls] += 1

                previous, done = done, done + len(chunk)
                if done // 500 > previous // 500 or done == total:
                    rate = done / max(time.time() - started, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    print(f"  [{done}/{total}] {rate:.1f} items/s, ~{eta / 60:.1f} min left",
                          flush=True)

    return {split: dict(counter) for split, counter in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Raw dataset folder.")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output folder.")
    parser.add_argument("--layout", choices=["auto", "presplit", "flat", "dfdc"], default="auto")
    parser.add_argument("--train", type=int, default=7000, help="Images per class for train (0 = all).")
    parser.add_argument("--val", type=int, default=1500, help="Images per class for val (0 = all).")
    parser.add_argument("--test", type=int, default=1500, help="Images per class for test (0 = all).")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        help="Split ratios used when sizes are 0 or for dfdc videos.")
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip MTCNN and just resize (only for datasets that are already face crops).")
    parser.add_argument("--overwrite", action="store_true", help="Delete --out first.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source folder not found: {args.source}")
    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out} is not empty. Use --overwrite to rebuild it.")
        shutil.rmtree(args.out)

    rng = random.Random(args.seed)
    layout = detect_layout(args.source) if args.layout == "auto" else args.layout
    sizes = {"train": args.train, "val": args.val, "test": args.test}
    print(f"Layout: {layout}")

    if layout == "presplit":
        plan = plan_presplit(args.source, sizes, rng)
    elif layout == "flat":
        plan = plan_flat(args.source, sizes, tuple(args.ratios), rng)
    else:
        plan = plan_dfdc(args.source, tuple(args.ratios), rng)

    for split in SPLITS:
        print(f"  {split:5s} " + "  ".join(f"{c}={len(plan[split][c])}" for c in CLASS_NAMES))

    extractor = None if args.no_crop else FaceExtractor()
    stats = process(plan, args.out, layout, extractor, args.frames_per_video)

    summary = {
        "source": str(args.source.resolve()),
        "layout": layout,
        "face_cropping": "none" if args.no_crop else "MTCNN",
        "seed": args.seed,
        "counts": stats,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Done. Dataset written to {args.out}")


if __name__ == "__main__":
    main()
