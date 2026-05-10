#!/usr/bin/env python3

import argparse
import csv
import html
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import Settings, load_inference_service  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIELDNAMES = [
    "true_label",
    "true_label_id",
    "source_group",
    "source_path",
    "image_width",
    "image_height",
    "face_found",
    "reason",
    "pred_label",
    "pred_label_id",
    "correct",
    "score",
    "logit",
    "threshold",
    "detector_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "detection_ms",
    "classification_ms",
    "total_ms",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production face-score pipeline on original images.")
    parser.add_argument("--positive-dir", type=Path, default=Path("data/positives_original"))
    parser.add_argument("--negative-dir", type=Path, default=Path("data/negatives_original"))
    parser.add_argument("--out", type=Path, default=Path("reports/original_pipeline_eval"))
    parser.add_argument("--classifier-model", type=Path, default=Path("runs/face_efficientnet_b0/cpu_export/img224/model_fp32_ts.pt"))
    parser.add_argument("--detector-model", type=Path, default=Path("face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--detector-score-threshold", type=float, default=0.65)
    parser.add_argument("--detector-nms-threshold", type=float, default=0.3)
    parser.add_argument("--detector-top-k", type=int, default=5000)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--clean", action="store_true", help="Remove output directory before running.")
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_group(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return rel.parts[0] if rel.parts else "."


def make_service(args: argparse.Namespace):
    settings = Settings(
        classifier_model_path=args.classifier_model,
        detector_model_path=args.detector_model,
        torch_num_threads=args.threads,
        torch_num_interop_threads=args.interop_threads,
        classifier_threshold=args.threshold,
        detector_score_threshold=args.detector_score_threshold,
        detector_nms_threshold=args.detector_nms_threshold,
        detector_top_k=args.detector_top_k,
        crop_margin=args.crop_margin,
        min_face_size=args.min_face_size,
    )
    return settings, load_inference_service(settings)


def prediction_row(path: Path, label: str, root: Path, service: Any) -> dict[str, Any]:
    label_id = 1 if label == "positive" else 0
    row: dict[str, Any] = {
        "true_label": label,
        "true_label_id": label_id,
        "source_group": source_group(path, root),
        "source_path": rel_path(path),
        "image_width": "",
        "image_height": "",
        "face_found": False,
        "reason": "",
        "pred_label": "negative",
        "pred_label_id": 0,
        "correct": label_id == 0,
        "score": "",
        "logit": "",
        "threshold": service.settings.classifier_threshold,
        "detector_score": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_width": "",
        "bbox_height": "",
        "detection_ms": "",
        "classification_ms": "",
        "total_ms": "",
        "error": "",
    }

    try:
        with Image.open(path) as image:
            row["image_width"] = image.width
            row["image_height"] = image.height
            result = service.predict(image)
    except Exception as exc:
        row["reason"] = "error"
        row["error"] = str(exc)
        return row

    row["face_found"] = bool(result.get("face_found"))
    row["reason"] = result.get("reason") or ""
    row["score"] = result.get("score") if result.get("score") is not None else ""
    row["logit"] = result.get("logit") if result.get("logit") is not None else ""
    row["detector_score"] = result.get("detector_score") if result.get("detector_score") is not None else ""
    timings = result.get("timings_ms") or {}
    row["detection_ms"] = timings.get("detection", "")
    row["classification_ms"] = timings.get("classification", "")
    row["total_ms"] = timings.get("total", "")

    bbox = result.get("bbox")
    if bbox:
        row["bbox_x1"] = bbox["x1"]
        row["bbox_y1"] = bbox["y1"]
        row["bbox_x2"] = bbox["x2"]
        row["bbox_y2"] = bbox["y2"]
        row["bbox_width"] = bbox["width"]
        row["bbox_height"] = bbox["height"]

    if result.get("face_found"):
        pred_label = result["label"]
        row["pred_label"] = pred_label
        row["pred_label_id"] = 1 if pred_label == "positive" else 0
        row["correct"] = int(row["pred_label_id"]) == label_id
    else:
        row["reason"] = row["reason"] or "rejected"
        row["pred_label"] = "negative"
        row["pred_label_id"] = 0
        row["correct"] = label_id == 0

    return row


def load_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def confusion(rows: list[dict[str, Any]], accepted_only: bool = False) -> dict[str, int]:
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        if accepted_only and not parse_bool(row["face_found"]):
            continue
        label = parse_int(row["true_label_id"])
        pred = parse_int(row["pred_label_id"])
        if label == 1 and pred == 1:
            counts["tp"] += 1
        elif label == 0 and pred == 0:
            counts["tn"] += 1
        elif label == 0 and pred == 1:
            counts["fp"] += 1
        elif label == 1 and pred == 0:
            counts["fn"] += 1
    return counts


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in pairs[index:next_index])
        index = next_index
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics_from_confusion(counts: dict[str, int], rows: int, auc: float | None) -> dict[str, Any]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": rows,
        "accuracy": (tp + tn) / rows if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion": counts,
    }


def threshold_metrics(rows: list[dict[str, Any]], threshold: float, accepted_only: bool) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        face_found = parse_bool(row["face_found"])
        if accepted_only and not face_found:
            continue
        score = parse_float(row["score"])
        label = parse_int(row["true_label_id"])
        pred = 1 if face_found and score is not None and score >= threshold else 0
        enriched = dict(row)
        enriched["pred_label_id"] = pred
        evaluated.append(enriched)
        if score is not None:
            labels.append(label)
            scores.append(score)
        elif not accepted_only:
            labels.append(label)
            scores.append(0.0)
    return metrics_from_confusion(confusion(evaluated), len(evaluated), roc_auc(labels, scores))


def best_threshold(rows: list[dict[str, Any]], metric: str, accepted_only: bool) -> dict[str, Any]:
    scores = sorted({score for row in rows if (score := parse_float(row["score"])) is not None})
    if not scores:
        return {}
    candidates = [0.0, *scores, 1.0]
    best: dict[str, Any] | None = None
    for threshold in candidates:
        current = threshold_metrics(rows, threshold, accepted_only)
        current["threshold"] = threshold
        if best is None or current[metric] > best[metric]:
            best = current
    return best or {}


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def score_summary(values: list[float]) -> dict[str, Any]:
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0] if sorted_values else None,
        "p05": quantile(sorted_values, 0.05),
        "p25": quantile(sorted_values, 0.25),
        "p50": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p95": quantile(sorted_values, 0.95),
        "max": sorted_values[-1] if sorted_values else None,
        "mean": (sum(sorted_values) / len(sorted_values) if sorted_values else None),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in ("positive", "negative"):
        label_rows = [row for row in rows if row["true_label"] == label]
        reasons = Counter(row["reason"] or ("accepted" if parse_bool(row["face_found"]) else "unknown") for row in label_rows)
        scores = [score for row in label_rows if (score := parse_float(row["score"])) is not None]
        by_label[label] = {
            "images": len(label_rows),
            "accepted": sum(1 for row in label_rows if parse_bool(row["face_found"])),
            "predicted_positive": sum(1 for row in label_rows if parse_int(row["pred_label_id"]) == 1),
            "reasons": dict(sorted(reasons.items())),
            "scores": score_summary(scores),
        }

    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[f"{row['true_label']}:{row['source_group']}"].append(row)

    by_group = []
    for key, items in sorted(group_rows.items()):
        label, group = key.split(":", 1)
        item_metrics = threshold_metrics(items, args.threshold, accepted_only=False)
        accepted = sum(1 for item in items if parse_bool(item["face_found"]))
        by_group.append(
            {
                "true_label": label,
                "source_group": group,
                "images": len(items),
                "accepted": accepted,
                "accepted_rate": accepted / len(items) if items else 0.0,
                "predicted_positive": item_metrics["confusion"]["tp"] + item_metrics["confusion"]["fp"],
                "accuracy": item_metrics["accuracy"],
                "precision": item_metrics["precision"],
                "recall": item_metrics["recall"],
                "f1": item_metrics["f1"],
            }
        )

    return {
        "config": {
            "positive_dir": str(args.positive_dir),
            "negative_dir": str(args.negative_dir),
            "classifier_model_path": str(settings.classifier_model_path),
            "detector_model_path": str(settings.detector_model_path),
            "threshold": args.threshold,
            "detector_score_threshold": args.detector_score_threshold,
            "detector_nms_threshold": args.detector_nms_threshold,
            "detector_top_k": args.detector_top_k,
            "crop_margin": args.crop_margin,
            "min_face_size": args.min_face_size,
        },
        "dataset": {
            "images": len(rows),
            "labels": by_label,
        },
        "metrics": {
            "all_images_rejections_as_negative": threshold_metrics(rows, args.threshold, accepted_only=False),
            "accepted_faces_only": threshold_metrics(rows, args.threshold, accepted_only=True),
            "best_f1_all_images": best_threshold(rows, "f1", accepted_only=False),
            "best_accuracy_all_images": best_threshold(rows, "accuracy", accepted_only=False),
            "best_f1_accepted_faces": best_threshold(rows, "f1", accepted_only=True),
            "best_accuracy_accepted_faces": best_threshold(rows, "accuracy", accepted_only=True),
        },
        "by_group": by_group,
    }


def write_group_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "true_label",
        "source_group",
        "images",
        "accepted",
        "accepted_rate",
        "predicted_positive",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(groups)


def write_mistakes(rows: list[dict[str, Any]], out_dir: Path) -> None:
    mistakes = [row for row in rows if not parse_bool(row["correct"])]
    false_positives = [row for row in mistakes if row["true_label"] == "negative"]
    false_negatives = [row for row in mistakes if row["true_label"] == "positive"]
    false_positives.sort(key=lambda row: parse_float(row["score"]) or -1.0, reverse=True)
    false_negatives.sort(key=lambda row: parse_float(row["score"]) if parse_float(row["score"]) is not None else -1.0)

    for name, items in (
        ("mistakes.csv", mistakes),
        ("false_positives.csv", false_positives),
        ("false_negatives.csv", false_negatives),
    ):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(items)

    render_mistakes_html(out_dir / "mistakes.html", false_positives[:120], false_negatives[:120], out_dir)


def render_cards(rows: list[dict[str, Any]], out_dir: Path) -> str:
    cards = []
    for row in rows:
        source = REPO_ROOT / row["source_path"]
        image_url = html.escape(os.path.relpath(source, out_dir))
        score = parse_float(row["score"])
        score_text = "-" if score is None else f"{score:.4f}"
        reason = row["reason"] or "accepted"
        cards.append(
            f"""
            <article class="card">
              <a href="{image_url}"><img src="{image_url}" loading="lazy" alt=""></a>
              <div class="meta">
                <strong>{html.escape(row['true_label'])} -> {html.escape(row['pred_label'])}</strong>
                <span>score {score_text}, reason {html.escape(reason)}</span>
                <span>detector {html.escape(str(row['detector_score']))}</span>
                <a href="{image_url}">{html.escape(row['source_path'])}</a>
              </div>
            </article>
            """
        )
    return "".join(cards)


def render_mistakes_html(path: Path, false_positives: list[dict[str, Any]], false_negatives: list[dict[str, Any]], out_dir: Path) -> None:
    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Original pipeline mistakes</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f7f8f5; color: #151817; }}
    header {{ position: sticky; top: 0; z-index: 1; padding: 14px 18px; background: #fff; border-bottom: 1px solid #ddd; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    p {{ margin: 0; color: #606964; }}
    section {{ padding: 16px; }}
    h2 {{ font-size: 18px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ overflow: hidden; border: 1px solid #ddd; border-radius: 8px; background: #fff; }}
    .card img {{ display: block; width: 100%; height: 260px; object-fit: contain; background: #eef1ea; }}
    .meta {{ display: grid; gap: 4px; padding: 10px; font-size: 12px; overflow-wrap: anywhere; }}
    .meta a {{ color: #1f6f4b; }}
  </style>
</head>
<body>
  <header>
    <h1>Original pipeline mistakes</h1>
    <p>Top false positives by score and false negatives by lowest score/rejection.</p>
  </header>
  <section>
    <h2>False positives</h2>
    <main>{render_cards(false_positives, out_dir)}</main>
  </section>
  <section>
    <h2>False negatives</h2>
    <main>{render_cards(false_negatives, out_dir)}</main>
  </section>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    settings, service = make_service(args)
    predictions_path = args.out / "predictions.csv"
    existing_rows = load_existing(predictions_path)
    processed = {row["source_path"] for row in existing_rows}

    positive_paths = iter_images(args.positive_dir)
    negative_paths = iter_images(args.negative_dir)
    labeled_paths = [
        *[("positive", args.positive_dir, path) for path in positive_paths],
        *[("negative", args.negative_dir, path) for path in negative_paths],
    ]

    mode = "a" if predictions_path.exists() and not args.clean else "w"
    with predictions_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        for index, (label, root, path) in enumerate(labeled_paths, 1):
            relative = rel_path(path)
            if relative in processed:
                continue
            row = prediction_row(path, label, root, service)
            writer.writerow(row)
            processed.add(relative)
            if args.progress_every and (index % args.progress_every == 0 or index == len(labeled_paths)):
                file.flush()
                print(f"processed {index}/{len(labeled_paths)} images")

    rows = load_existing(predictions_path)
    summary = summarize(rows, args, settings)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_group_csv(args.out / "by_group.csv", summary["by_group"])
    write_mistakes(rows, args.out)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
