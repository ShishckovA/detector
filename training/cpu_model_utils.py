#!/usr/bin/env python3

import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from training.train_face_classifier import ID_TO_LABEL, LABEL_TO_ID, build_model, build_transforms, class_metric_row


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
            "  .venv/bin/pip install -r training/requirements-cpu.txt"
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
                Path.cwd() / "data" / rel,
                csv_path.parent.parent.parent / rel,
                csv_path.parent.parent.parent / "data" / rel,
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
    class_to_idx = checkpoint.get("class_to_idx") or checkpoint.get("config", {}).get("class_to_idx") or LABEL_TO_ID
    id_to_label = {int(index): label for label, index in class_to_idx.items()}
    classifier_weight = checkpoint["model_state_dict"].get("classifier.1.weight")
    num_outputs = int(classifier_weight.shape[0]) if classifier_weight is not None else len(class_to_idx)
    if num_outputs == 1:
        id_to_label = {0: "negative", 1: "positive"}
    model = build_model(deps, "random", num_classes=num_outputs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model.class_to_idx = class_to_idx
    model.id_to_label = id_to_label
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
    id_to_label = getattr(model, "id_to_label", ID_TO_LABEL)

    predictions = [dict(row) for row in rows]
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            logits_list = logits.detach().cpu().tolist()
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0]).detach().cpu().tolist()
                for index, logit_values, prob in zip(indices.tolist(), logits_list, probs):
                    pred_id = 1 if float(prob) >= 0.5 else 0
                    predictions[index]["pred_label_id"] = pred_id
                    predictions[index]["pred_label"] = id_to_label[pred_id]
                    predictions[index]["logit_positive"] = float(logit_values[0])
                    predictions[index]["prob_negative"] = 1.0 - float(prob)
                    predictions[index]["prob_positive"] = float(prob)
                continue

            probs = torch.softmax(logits, dim=1).detach().cpu().tolist()
            for index, logit_values, prob_values in zip(indices.tolist(), logits_list, probs):
                pred_id = max(range(len(prob_values)), key=lambda class_index: prob_values[class_index])
                predictions[index]["pred_label_id"] = pred_id
                predictions[index]["pred_label"] = id_to_label.get(pred_id, str(pred_id))
                for class_id, class_name in id_to_label.items():
                    predictions[index][f"logit_{class_name}"] = float(logit_values[class_id])
                    predictions[index][f"prob_{class_name}"] = float(prob_values[class_id])

    metrics = summarize_predictions(predictions, id_to_label)
    return predictions, metrics


def summarize_predictions(predictions: list[dict[str, Any]], id_to_label: dict[int, str] | None = None) -> dict[str, Any]:
    id_to_label = id_to_label or ID_TO_LABEL
    labels = [int(row["label_id"]) for row in predictions]
    pred_ids = [int(row["pred_label_id"]) for row in predictions]
    counts = Counter((label, pred_id) for label, pred_id in zip(labels, pred_ids))
    correct = sum(1 for label, pred_id in zip(labels, pred_ids) if label == pred_id)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    confusion: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, float]] = {}
    label_counts = Counter(labels)
    for class_id, class_name in id_to_label.items():
        tp = counts[(class_id, class_id)]
        fp = sum(counts[(other, class_id)] for other in id_to_label if other != class_id)
        fn = sum(counts[(class_id, other)] for other in id_to_label if other != class_id)
        predicted_count = sum(1 for pred_id in pred_ids if pred_id == class_id)
        per_class[class_name] = class_metric_row(
            float(tp),
            float(fp),
            float(fn),
            float(label_counts[class_id]),
            float(predicted_count),
        )
        precisions.append(per_class[class_name]["precision"])
        recalls.append(per_class[class_name]["recall"])
        f1s.append(per_class[class_name]["f1"])
        confusion[class_name] = {id_to_label[pred_id]: counts[(class_id, pred_id)] for pred_id in id_to_label}

    return {
        "rows": len(predictions),
        "accuracy": correct / max(1, len(predictions)),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "labels": {id_to_label[class_id]: label_counts[class_id] for class_id in id_to_label},
        "confusion": confusion,
    }


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
