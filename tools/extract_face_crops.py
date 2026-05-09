#!/usr/bin/env python3

import argparse
import csv
import html
import json
import math
import os
import shutil
from pathlib import Path

import cv2 as cv


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def unique_crop_path(out_dir: Path, source_path: Path, face_index: int, score: float) -> Path:
    rel = source_path.with_suffix("")
    safe_parts = [part.replace(" ", "_") for part in rel.parts]
    stem = "__".join(safe_parts)
    return out_dir / f"{stem}__face_{face_index:02d}__score_{score:.4f}.jpg"


def detect_faces_yunet(
    image_path: Path,
    model_path: Path,
    crop_dir: Path,
    margin: float,
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
    best_face_only: bool,
):
    img = cv.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]
    detector = cv.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(img_w, img_h),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )

    _, faces = detector.detect(img)
    if faces is None:
        return []
    if best_face_only and len(faces) > 1:
        best_index = max(range(len(faces)), key=lambda idx: float(faces[idx][14]))
        faces = faces[best_index : best_index + 1]

    rows = []
    for face_index, face in enumerate(faces):
        x, y, w, h = face[:4]
        score = float(face[14])

        pad_x = w * margin
        pad_y = h * margin
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))
        crop_w = x2 - x1
        crop_h = y2 - y1

        crop_path = unique_crop_path(crop_dir, image_path, face_index, score)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(crop_path), img[y1:y2, x1:x2])

        rows.append(
            {
                "source_path": str(image_path),
                "source_dir": str(image_path.parent),
                "source_name": image_path.name,
                "face_index": face_index,
                "score": score,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "image_width": img_w,
                "image_height": img_h,
                "crop_path": str(crop_path),
            }
        )

    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "rank",
        "score",
        "source_path",
        "source_dir",
        "source_name",
        "face_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_width",
        "crop_height",
        "size_bucket",
        "image_width",
        "image_height",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            out = dict(row)
            out["rank"] = rank
            out.setdefault("size_bucket", "")
            writer.writerow(out)


def write_no_face_csv(path: Path, paths):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path"])
        for item in paths:
            writer.writerow([str(item)])


def quantile(sorted_scores, q: float):
    if not sorted_scores:
        return None
    pos = (len(sorted_scores) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_scores[lower]
    return sorted_scores[lower] * (upper - pos) + sorted_scores[upper] * (pos - lower)


def write_summary(path: Path, image_count: int, rows, large_rows, small_rows, no_face_count: int, args):
    scores = sorted(row["score"] for row in rows)
    summary = {
        "input": str(args.input),
        "model": str(args.model),
        "score_threshold": args.score_threshold,
        "best_face_only": args.best_face_only,
        "min_size": args.min_size,
        "nms_threshold": args.nms_threshold,
        "top_k": args.top_k,
        "margin": args.margin,
        "sort": args.sort,
        "images": image_count,
        "images_without_faces": no_face_count,
        "detections": len(rows),
        "detections_at_least_min_size": len(large_rows),
        "detections_below_min_size": len(small_rows),
        "min_score": scores[0] if scores else None,
        "max_score": scores[-1] if scores else None,
        "p01": quantile(scores, 0.01),
        "p05": quantile(scores, 0.05),
        "p10": quantile(scores, 0.10),
        "p25": quantile(scores, 0.25),
        "p50": quantile(scores, 0.50),
        "p75": quantile(scores, 0.75),
        "p90": quantile(scores, 0.90),
        "p95": quantile(scores, 0.95),
        "p99": quantile(scores, 0.99),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel_url(path: Path, base_dir: Path) -> str:
    return html.escape(os.path.relpath(path, base_dir))


def render_cards(rows, base_dir: Path):
    cards = []
    for rank, row in enumerate(rows, 1):
        score = row["score"]
        source = Path(row["source_path"])
        crop = Path(row["crop_path"])
        cards.append(
            f"""
            <article class="card">
              <a href="{rel_url(crop, base_dir)}"><img src="{rel_url(crop, base_dir)}" loading="lazy" alt=""></a>
              <div class="meta">
                <strong>#{rank} score {score:.4f}</strong>
                <a href="{rel_url(source, base_dir)}">{html.escape(str(source))}</a>
                <span>crop {row['crop_width']}x{row['crop_height']}</span>
                <span>bbox {row['bbox_x1']},{row['bbox_y1']} - {row['bbox_x2']},{row['bbox_y2']}</span>
              </div>
            </article>
            """
        )
    return "".join(cards)


def write_html(path: Path, large_rows, small_rows, args):
    base_dir = path.parent
    large_cards = render_cards(large_rows, base_dir)
    small_cards = render_cards(small_rows, base_dir)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face scores</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f6f6f3; color: #202020; }}
    header {{ position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #ddd; padding: 12px 18px; z-index: 1; }}
    h1 {{ font-size: 18px; margin: 0 0 4px; }}
    p {{ margin: 0; color: #555; font-size: 13px; }}
    nav {{ margin-top: 8px; display: flex; gap: 12px; flex-wrap: wrap; font-size: 13px; }}
    nav a {{ color: #23527c; }}
    section {{ padding-top: 10px; }}
    h2 {{ font-size: 16px; margin: 10px 12px 0; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; padding: 12px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; height: 220px; object-fit: contain; background: #ececea; }}
    .meta {{ display: grid; gap: 4px; padding: 9px; font-size: 12px; overflow-wrap: anywhere; }}
    .meta a {{ color: #23527c; }}
    .meta span {{ color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>Face scores</h1>
    <p>{len(large_rows) + len(small_rows)} detections, sorted {html.escape(args.sort)}, threshold {args.score_threshold}, min size {args.min_size}x{args.min_size}</p>
    <nav>
      <a href="#large">At least {args.min_size}x{args.min_size}: {len(large_rows)}</a>
      <a href="#small">Below {args.min_size}x{args.min_size}: {len(small_rows)}</a>
    </nav>
  </header>
  <section id="large">
    <h2>At least {args.min_size}x{args.min_size}</h2>
    <main>{large_cards}</main>
  </section>
  <section id="small">
    <h2>Below {args.min_size}x{args.min_size}</h2>
    <main>{small_cards}</main>
  </section>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_single_section_html(path: Path, rows, title: str, subtitle: str):
    base_dir = path.parent
    cards = render_cards(rows, base_dir)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f6f6f3; color: #202020; }}
    header {{ position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #ddd; padding: 12px 18px; z-index: 1; }}
    h1 {{ font-size: 18px; margin: 0 0 4px; }}
    p {{ margin: 0; color: #555; font-size: 13px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; padding: 12px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; height: 220px; object-fit: contain; background: #ececea; }}
    .meta {{ display: grid; gap: 4px; padding: 9px; font-size: 12px; overflow-wrap: anywhere; }}
    .meta a {{ color: #23527c; }}
    .meta span {{ color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
  </header>
  <main>{cards}</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch YuNet face scoring and crop export.")
    parser.add_argument("input", nargs="?", default="data", type=Path, help="Input image or directory.")
    parser.add_argument("--model", type=Path, default=Path("face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--out", type=Path, default=Path("reports/face_scores"))
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--best-face-only", action="store_true", help="Keep only the highest-score face per source image.")
    parser.add_argument("--min-size", type=int, default=48, help="Split detections by minimum crop width and height.")
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--sort", choices=["asc", "desc"], default="asc")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before writing.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    crop_dir = args.out / "crops"
    args.out.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(args.input))
    rows = []
    no_face_paths = []
    errors = []

    for idx, image_path in enumerate(image_paths, 1):
        try:
            found = detect_faces_yunet(
                image_path=image_path,
                model_path=args.model,
                crop_dir=crop_dir,
                margin=args.margin,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                top_k=args.top_k,
                best_face_only=args.best_face_only,
            )
        except Exception as exc:
            errors.append({"source_path": str(image_path), "error": str(exc)})
            continue

        if found:
            rows.extend(found)
        else:
            no_face_paths.append(image_path)

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"processed {idx}/{len(image_paths)} images, detections={len(rows)}")

    reverse = args.sort == "desc"
    rows.sort(key=lambda row: row["score"], reverse=reverse)
    large_rows = []
    small_rows = []
    for row in rows:
        if args.min_size and (row["crop_width"] < args.min_size or row["crop_height"] < args.min_size):
            row["size_bucket"] = f"below_{args.min_size}"
            small_rows.append(row)
        else:
            row["size_bucket"] = f"at_least_{args.min_size}" if args.min_size else "all"
            large_rows.append(row)

    write_csv(args.out / "faces_by_score.csv", rows)
    write_csv(args.out / f"faces_{args.min_size}plus_by_score.csv", large_rows)
    write_csv(args.out / f"faces_under_{args.min_size}_by_score.csv", small_rows)
    write_no_face_csv(args.out / "no_faces.csv", no_face_paths)
    write_summary(args.out / "summary.json", len(image_paths), rows, large_rows, small_rows, len(no_face_paths), args)
    write_html(args.out / "index.html", large_rows, small_rows, args)
    write_single_section_html(
        args.out / f"index_{args.min_size}plus.html",
        large_rows,
        f"Face scores: at least {args.min_size}x{args.min_size}",
        f"{len(large_rows)} detections, width >= {args.min_size} and height >= {args.min_size}, sorted {args.sort}",
    )
    write_single_section_html(
        args.out / f"index_under_{args.min_size}.html",
        small_rows,
        f"Face scores: below {args.min_size}x{args.min_size}",
        f"{len(small_rows)} detections, width < {args.min_size} or height < {args.min_size}, sorted {args.sort}",
    )

    if errors:
        with (args.out / "errors.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "error"])
            writer.writeheader()
            writer.writerows(errors)

    print(f"images: {len(image_paths)}")
    print(f"detections: {len(rows)}")
    print(f"detections at least {args.min_size}x{args.min_size}: {len(large_rows)}")
    print(f"detections below {args.min_size}x{args.min_size}: {len(small_rows)}")
    print(f"images without detections: {len(no_face_paths)}")
    print(f"output: {args.out}")
    print(f"csv: {args.out / 'faces_by_score.csv'}")
    print(f"html: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
