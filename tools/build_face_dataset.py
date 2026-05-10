#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import crop_box_for_face, detect_faces  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
PERSON_CLASSES = {"alex", "artem"}
FIELDNAMES = [
    "label",
    "dataset_path",
    "source_dir",
    "source_path",
    "source_name",
    "score",
    "crop_width",
    "crop_height",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "original_crop_path",
    "split_group_key",
    "dataset_weight",
]


@dataclass(frozen=True)
class SourceImage:
    source_root_label: str
    root: Path
    path: Path

    @property
    def label(self) -> str:
        source_group, person = source_parts(self)
        if self.source_root_label == "positive" and source_group == "us" and person in PERSON_CLASSES:
            return person
        return self.source_root_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PNG face crop dataset for four-class face classification.")
    parser.add_argument(
        "--positive-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Positive originals root. May be repeated, for example: "
            "--positive-dir data/positives_original "
            "--positive-dir data/positives_original_search4faces_top20"
        ),
    )
    parser.add_argument("--negative-dir", type=Path, default=Path("data/negatives_original"))
    parser.add_argument("--out", type=Path, default=Path("data/face_dataset_png"))
    parser.add_argument("--detector-model", type=Path, default=Path("face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--detector-score-threshold", type=float, default=0.65)
    parser.add_argument("--detector-nms-threshold", type=float, default=0.3)
    parser.add_argument("--detector-top-k", type=int, default=5000)
    parser.add_argument("--detector-input-max-side", type=int, default=1024)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--personal-weight", type=float, default=5.0)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def rel_path(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def source_parts(source: SourceImage) -> tuple[str, str]:
    rel = source.path.resolve().relative_to(source.root.resolve())
    if len(rel.parts) < 3:
        raise ValueError(f"Expected image below <source>/<person>/<file>: {source.path}")
    return rel.parts[0], rel.parts[1]


def unique_stem(path: Path | str) -> str:
    rel = Path(path).with_suffix("")
    return "__".join(part.replace(" ", "_") for part in rel.parts)


def open_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image: {path}") from exc


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)


def detect_best_face(
    detector: Any,
    image: Image.Image,
    margin: float,
    detector_input_max_side: int,
) -> tuple[Any | None, Any]:
    bgr = pil_to_bgr(image)
    faces = detect_faces(detector, bgr, max_side=detector_input_max_side)
    if not faces:
        return None, None
    best = max(faces, key=lambda face: (face.detector_score, face.area))
    crop_box = crop_box_for_face(best, image.width, image.height, margin)
    return best, crop_box


def dataset_weight(label: str, args: argparse.Namespace) -> float:
    return args.personal_weight if label in PERSON_CLASSES else args.default_weight


def split_group_key(source_group: str, person: str, source_path: Path) -> str:
    if source_group == "us":
        return f"{source_group}/{person}/{unique_stem(source_path.name)}"
    return f"{source_group}/{person}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_errors(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "source_path", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    positive_dirs = args.positive_dir or [Path("data/positives_original")]
    register_heif()

    if not args.detector_model.exists():
        raise FileNotFoundError(f"Detector model not found: {args.detector_model}")
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    crop_root = args.out / "data"
    crop_root.mkdir(parents=True, exist_ok=True)
    detector = cv.FaceDetectorYN.create(
        model=str(args.detector_model.resolve()),
        config="",
        input_size=(1, 1),
        score_threshold=args.detector_score_threshold,
        nms_threshold=args.detector_nms_threshold,
        top_k=args.detector_top_k,
    )

    sources = [
        *[
            SourceImage("positive", positive_dir, path)
            for positive_dir in positive_dirs
            for path in iter_images(positive_dir)
        ],
        *[SourceImage("negative", args.negative_dir, path) for path in iter_images(args.negative_dir)],
    ]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    face_index_by_group: dict[tuple[str, str, str], int] = defaultdict(int)

    for index, source in enumerate(sources, 1):
        try:
            image = open_rgb(source.path)
            source_group, person = source_parts(source)
            face, crop_box = detect_best_face(detector, image, args.crop_margin, args.detector_input_max_side)
            label = source.label
            if face is None or crop_box is None:
                rejected.append({"label": label, "source_path": rel_path(source.path), "error": "no_face"})
                continue
            if crop_box.width < args.min_face_size or crop_box.height < args.min_face_size:
                rejected.append({"label": label, "source_path": rel_path(source.path), "error": "face_too_small"})
                continue

            key = (label, source_group, person)
            face_index_by_group[key] += 1
            face_name = f"face_{face_index_by_group[key]}.png"
            crop_path = crop_root / label / source_group / person / face_name
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop = image.crop((crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
            crop.save(crop_path, format="PNG")

            dataset_path = rel_path(crop_path, args.out.parent)
            source_rel = rel_path(source.path, source.root)
            rows.append(
                {
                    "label": label,
                    "dataset_path": dataset_path,
                    "source_dir": str(Path(source_rel).parent),
                    "source_path": source_rel,
                    "source_name": source.path.name,
                    "score": f"{face.detector_score:.9f}",
                    "crop_width": crop_box.width,
                    "crop_height": crop_box.height,
                    "bbox_x1": crop_box.x1,
                    "bbox_y1": crop_box.y1,
                    "bbox_x2": crop_box.x2,
                    "bbox_y2": crop_box.y2,
                    "original_crop_path": "",
                    "split_group_key": split_group_key(source_group, person, source.path),
                    "dataset_weight": dataset_weight(label, args),
                }
            )
        except Exception as exc:
            errors.append({"label": source.label, "source_path": rel_path(source.path), "error": str(exc)})

        if args.progress_every and (index % args.progress_every == 0 or index == len(sources)):
            print(
                f"processed {index}/{len(sources)} images, crops={len(rows)}, rejected={len(rejected)}, errors={len(errors)}",
                flush=True,
            )

    positive_rows = [row for row in rows if row["label"] == "positive"]
    negative_rows = [row for row in rows if row["label"] == "negative"]
    alex_rows = [row for row in rows if row["label"] == "alex"]
    artem_rows = [row for row in rows if row["label"] == "artem"]
    write_csv(args.out / "positive_manifest.csv", positive_rows)
    write_csv(args.out / "negative_manifest.csv", negative_rows)
    write_csv(args.out / "alex_manifest.csv", alex_rows)
    write_csv(args.out / "artem_manifest.csv", artem_rows)
    write_csv(args.out / "manifest.csv", rows)
    write_errors(args.out / "rejected.csv", rejected)
    write_errors(args.out / "errors.csv", errors)

    label_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(f"{row['label']}:{Path(row['dataset_path']).parts[3]}" for row in rows)
    summary = {
        "positive_images": len([source for source in sources if source.label == "positive"]),
        "negative_images": len([source for source in sources if source.label == "negative"]),
        "alex_images": len([source for source in sources if source.label == "alex"]),
        "artem_images": len([source for source in sources if source.label == "artem"]),
        "positive_crops": label_counts["positive"],
        "negative_crops": label_counts["negative"],
        "alex_crops": label_counts["alex"],
        "artem_crops": label_counts["artem"],
        "rejected": len(rejected),
        "errors": len(errors),
        "format": "png",
        "detector_model": str(args.detector_model),
        "detector_score_threshold": args.detector_score_threshold,
        "nms_threshold": args.detector_nms_threshold,
        "top_k": args.detector_top_k,
        "detector_input_max_side": args.detector_input_max_side,
        "crop_margin": args.crop_margin,
        "min_face_size": args.min_face_size,
        "positive_dirs": [str(path) for path in positive_dirs],
        "negative_dir": str(args.negative_dir),
        "weights": {
            "default": args.default_weight,
            "alex": args.personal_weight,
            "artem": args.personal_weight,
        },
        "split_policy": {
            "default": "group-safe by <source>/<person>",
            "alex_artem": "image-level via split_group_key, so each person can appear in both train and val",
        },
        "counts_by_label_source": dict(sorted(source_counts.items())),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
