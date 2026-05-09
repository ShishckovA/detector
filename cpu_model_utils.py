#!/usr/bin/env python3

import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from train_face_classifier import binary_roc_auc, build_model, build_transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CsvImageDataset:
    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["resolved_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def import_cpu_dependencies() -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise SystemExit(
            "CPU inference dependencies are not installed. Install them with:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/pip install -r requirements-cpu.txt"
        ) from exc

    return {
        "torch": torch,
        "Image": Image,
        "nn": nn,
        "DataLoader": DataLoader,
        "transforms": transforms,
        "EfficientNet_B0_Weights": EfficientNet_B0_Weights,
        "efficientnet_b0": efficientnet_b0,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    missing = [index for index, row in enumerate(rows) if "resolved_path" not in row or "label_id" not in row]
    if missing:
        raise ValueError(f"Rows missing resolved_path or label_id in {path}: {missing[:10]}")
    for row in rows:
        row["resolved_path"] = str(resolve_row_image_path(row, path))
    return rows


def resolve_row_image_path(row: dict[str, Any], csv_path: Path) -> Path:
    resolved = Path(row["resolved_path"])
    if resolved.exists():
        return resolved

    dataset_path = row.get("dataset_path")
    candidates: list[Path] = []
    if dataset_path:
        rel = Path(dataset_path)
        candidates.extend(
            [
                Path.cwd() / rel,
                csv_path.parent.parent.parent / rel,
                csv_path.parent.parent.parent / "face_dataset" / rel,
            ]
        )
    candidates.append(resolved)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def load_checkpoint_model(checkpoint_path: Path, deps: dict[str, Any], device: Any) -> Any:
    torch = deps["torch"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(deps, "random")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_val_transform(deps: dict[str, Any], image_size: int) -> Any:
    _, val_transform = build_transforms(deps, image_size)
    return val_transform


def evaluate_model(
    model: Any,
    rows: list[dict[str, Any]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    deps: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deps = deps or import_cpu_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    transform = make_val_transform(deps, image_size)
    dataset = CsvImageDataset(rows, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    predictions = [dict(row) for row in rows]
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            logits_list = logits.detach().cpu().tolist()
            for index, logit, prob in zip(indices.tolist(), logits_list, probs):
                predictions[index]["logit"] = float(logit)
                predictions[index]["prob_positive"] = float(prob)

    metrics = summarize_predictions(predictions, threshold=0.5)
    metrics["best_f1_threshold"] = best_f1_threshold(predictions)
    metrics["best_accuracy_threshold"] = best_accuracy_threshold(predictions)
    return predictions, metrics


def summarize_predictions(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["label_id"]) for row in predictions]
    logits = [float(row["logit"]) for row in predictions]
    probs = [float(row["prob_positive"]) for row in predictions]
    pred_ids = [1 if prob >= threshold else 0 for prob in probs]
    counts = Counter((label, pred_id) for label, pred_id in zip(labels, pred_ids))
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(predictions),
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": binary_roc_auc(labels, logits),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def threshold_candidates(predictions: list[dict[str, Any]]) -> list[float]:
    probs = sorted({float(row["prob_positive"]) for row in predictions})
    if not probs:
        return [0.5]
    candidates = [0.0, 1.0]
    candidates.extend(probs)
    candidates.extend((left + right) / 2.0 for left, right in zip(probs, probs[1:]))
    candidates.append(0.5)
    return sorted(set(max(0.0, min(1.0, value)) for value in candidates))


def best_f1_threshold(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in threshold_candidates(predictions):
        metrics = summarize_predictions(predictions, threshold)
        key = (metrics["f1"], metrics["accuracy"], -abs(threshold - 0.5))
        if best is None or key > best["_key"]:
            best = {**metrics, "_key": key}
    assert best is not None
    best.pop("_key", None)
    return best


def best_accuracy_threshold(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in threshold_candidates(predictions):
        metrics = summarize_predictions(predictions, threshold)
        key = (metrics["accuracy"], metrics["f1"], -abs(threshold - 0.5))
        if best is None or key > best["_key"]:
            best = {**metrics, "_key": key}
    assert best is not None
    best.pop("_key", None)
    return best


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(predictions[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values_ms)),
        "mean_ms": statistics.fmean(values_ms) if values_ms else math.nan,
        "median_ms": percentile(values_ms, 0.50),
        "p90_ms": percentile(values_ms, 0.90),
        "p95_ms": percentile(values_ms, 0.95),
        "p99_ms": percentile(values_ms, 0.99),
        "min_ms": min(values_ms) if values_ms else math.nan,
        "max_ms": max(values_ms) if values_ms else math.nan,
    }


def benchmark_callable(fn: Any, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(values)
