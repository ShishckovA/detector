#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_NAMES = ("loss", "accuracy", "precision", "recall", "f1", "roc_auc")
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
EXTRA_SPLIT_FIELDS = [
    "label_id",
    "group_source",
    "group_person",
    "group_key",
    "dataset_weight",
    "split",
    "resolved_path",
]


@dataclass(frozen=True)
class ManifestData:
    rows: list[dict[str, Any]]
    fieldnames: list[str]
    manifest_path: Path


class FaceImageDataset:
    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, str]:
        import torch
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["resolved_path"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = torch.tensor(float(row["label_id"]), dtype=torch.float32)
        weight = torch.tensor(float(row.get("dataset_weight", 1.0)), dtype=torch.float32)
        return image, label, weight, row["group_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on face_dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("face_dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="face_efficientnet_b0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Fixed output directory. If omitted, uses RUNS_ROOT/RUN_NAME_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--weights", choices=("imagenet", "random"), default="imagenet")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--split-only", action="store_true", help="Create and validate splits without training.")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard log dir. Defaults to OUTPUT_DIR/tensorboard.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Do not write TensorBoard event files.")
    parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
        help="Per-dataset weight, repeatable. Dataset is group_source, e.g. akykla=2.0.",
    )
    parser.add_argument("--default-dataset-weight", type=float, default=1.0)
    return parser.parse_args()


def safe_run_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned.strip("._-") or "run"


def make_unique_run_dir(runs_root: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = runs_root / f"{safe_run_name(run_name)}_{timestamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return make_unique_run_dir(args.runs_root.resolve(), args.run_name).resolve()


def find_group_key(dataset_path: str, expected_label: str) -> tuple[str, str]:
    parts = Path(dataset_path).parts
    for idx, part in enumerate(parts):
        if part != "data":
            continue
        if idx + 4 >= len(parts):
            continue
        label = parts[idx + 1]
        if label not in LABEL_TO_ID:
            continue
        if label != expected_label:
            raise ValueError(
                f"Manifest label {expected_label!r} does not match dataset path label {label!r}: {dataset_path}"
            )
        return parts[idx + 2], parts[idx + 3]
    raise ValueError(
        "Expected dataset_path shaped like "
        "face_dataset/data/<class>/<source>/<person>/face_i.jpg, got: "
        f"{dataset_path}"
    )


def resolve_image_path(dataset_path: str, dataset_dir: Path, manifest_path: Path) -> Path:
    path = Path(dataset_path)
    if path.is_absolute():
        return path

    dataset_dir = dataset_dir.resolve()
    manifest_dir = manifest_path.parent.resolve()
    candidates = [
        Path.cwd() / path,
        dataset_dir.parent / path,
        manifest_dir.parent / path,
        dataset_dir / path,
        manifest_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_manifest(dataset_dir: Path) -> ManifestData:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        missing = {"label", "dataset_path"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            label = row["label"].strip()
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown label {label!r} in row {row_index}")
            source, person = find_group_key(row["dataset_path"], label)
            resolved_path = resolve_image_path(row["dataset_path"], dataset_dir, manifest_path)
            enriched = dict(row)
            enriched["label"] = label
            enriched["label_id"] = LABEL_TO_ID[label]
            enriched["group_source"] = source
            enriched["group_person"] = person
            enriched["group_key"] = f"{source}/{person}"
            enriched["resolved_path"] = str(resolved_path)
            rows.append(enriched)

    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    return ManifestData(rows=rows, fieldnames=list(reader.fieldnames), manifest_path=manifest_path)


def parse_dataset_weight_items(items: list[str], default_weight: float) -> dict[str, float]:
    if default_weight <= 0:
        raise ValueError(f"--default-dataset-weight must be positive, got {default_weight}")

    weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset-weight must look like DATASET=WEIGHT, got {item!r}")
        dataset, raw_weight = item.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise ValueError(f"Empty dataset name in --dataset-weight {item!r}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid weight in --dataset-weight {item!r}") from exc
        if weight <= 0:
            raise ValueError(f"Dataset weight must be positive, got {item!r}")
        weights[dataset] = weight
    return weights


def apply_dataset_weights(
    rows: list[dict[str, Any]],
    explicit_weights: dict[str, float],
    default_weight: float,
) -> dict[str, float]:
    datasets = sorted({row["group_source"] for row in rows})
    unknown = sorted(set(explicit_weights) - set(datasets))
    if unknown:
        raise ValueError(f"Unknown dataset weights for {unknown}; known datasets are {datasets}")

    effective = {dataset: explicit_weights.get(dataset, default_weight) for dataset in datasets}
    for row in rows:
        row["dataset_weight"] = effective[row["group_source"]]
    return effective


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        group_key = row["group_key"]
        if group_key not in grouped:
            grouped[group_key] = {
                "group_key": group_key,
                "counts": Counter(),
                "indices": [],
            }
        grouped[group_key]["counts"][int(row["label_id"])] += 1
        grouped[group_key]["indices"].append(index)
    return grouped


def split_pure_groups(
    groups: dict[str, dict[str, Any]],
    total_counts: Counter,
    val_size: float,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    val_groups: set[str] = set()

    for label_id in sorted(LABEL_TO_ID.values()):
        candidates = [
            group
            for group in groups.values()
            if group["counts"][label_id] > 0 and sum(group["counts"].values()) == group["counts"][label_id]
        ]
        rng.shuffle(candidates)
        selected_count = 0
        target_count = total_counts[label_id] * val_size
        for group in candidates:
            if selected_count >= target_count:
                break
            val_groups.add(group["group_key"])
            selected_count += group["counts"][label_id]

    return val_groups


def split_mixed_groups(
    groups: dict[str, dict[str, Any]],
    total_counts: Counter,
    val_size: float,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    target = {label_id: total_counts[label_id] * val_size for label_id in LABEL_TO_ID.values()}
    target_total = sum(target.values())

    def score(counts: Counter) -> float:
        value = 0.0
        for label_id, target_count in target.items():
            denom = max(1.0, target_count)
            value += ((counts[label_id] - target_count) / denom) ** 2
            if target_count > 0 and counts[label_id] == 0:
                value += 10.0
        total_denom = max(1.0, target_total)
        value += 0.25 * ((sum(counts.values()) - target_total) / total_denom) ** 2
        return value

    items = list(groups.values())
    keyed_items = [(rng.random(), group) for group in items]
    keyed_items.sort(key=lambda item: (-max(item[1]["counts"].values()), -sum(item[1]["counts"].values()), item[0]))

    val_groups: set[str] = set()
    current = Counter()
    current_score = score(current)
    for _, group in keyed_items:
        candidate = current + group["counts"]
        candidate_score = score(candidate)
        if candidate_score < current_score:
            val_groups.add(group["group_key"])
            current = candidate
            current_score = candidate_score

    for label_id in LABEL_TO_ID.values():
        if current[label_id] > 0:
            continue
        candidates = [group for group in items if group["counts"][label_id] > 0 and group["group_key"] not in val_groups]
        if candidates:
            smallest = min(candidates, key=lambda group: sum(group["counts"].values()))
            val_groups.add(smallest["group_key"])
            current += smallest["counts"]

    return val_groups


def make_group_safe_split(
    rows: list[dict[str, Any]],
    val_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"--val-size must be between 0 and 1, got {val_size}")

    groups = group_rows(rows)
    total_counts = Counter(int(row["label_id"]) for row in rows)
    mixed_group_exists = any(sum(1 for count in group["counts"].values() if count > 0) > 1 for group in groups.values())
    if mixed_group_exists:
        val_groups = split_mixed_groups(groups, total_counts, val_size, seed)
    else:
        val_groups = split_pure_groups(groups, total_counts, val_size, seed)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["group_key"] in val_groups:
            out = dict(row)
            out["split"] = "val"
            val_rows.append(out)
        else:
            out = dict(row)
            out["split"] = "train"
            train_rows.append(out)

    validate_split(train_rows, val_rows)
    return train_rows, val_rows


def validate_split(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_groups = {row["group_key"] for row in train_rows}
    val_groups = {row["group_key"] for row in val_rows}
    leakage = sorted(train_groups & val_groups)
    if leakage:
        raise ValueError(f"Group leakage between train and val: {leakage[:10]}")

    for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
        label_counts = Counter(row["label"] for row in split_rows)
        missing = sorted(set(LABEL_TO_ID) - set(label_counts))
        if missing:
            raise ValueError(f"{split_name} split is missing labels: {missing}")


def validate_image_paths(rows: list[dict[str, Any]]) -> None:
    missing = [row["resolved_path"] for row in rows if not Path(row["resolved_path"]).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First missing paths:\n{preview}")


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(row["label"] for row in rows)
    return {
        "rows": len(rows),
        "groups": len({row["group_key"] for row in rows}),
        "labels": {label: label_counts.get(label, 0) for label in LABEL_TO_ID},
    }


def split_summary(
    all_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    val_size: float,
    seed: int,
) -> dict[str, Any]:
    total_label_counts = Counter(row["label"] for row in all_rows)
    val_label_counts = Counter(row["label"] for row in val_rows)
    train_groups = {row["group_key"] for row in train_rows}
    val_groups = {row["group_key"] for row in val_rows}
    return {
        "val_size_requested": val_size,
        "seed": seed,
        "total": split_counts(all_rows),
        "train": split_counts(train_rows),
        "val": split_counts(val_rows),
        "val_share_by_label": {
            label: (val_label_counts[label] / total_label_counts[label] if total_label_counts[label] else None)
            for label in LABEL_TO_ID
        },
        "group_leakage": sorted(train_groups & val_groups),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_split_csv(path: Path, rows: list[dict[str, Any]], manifest_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_fields)
    for field in EXTRA_SPLIT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_split_artifacts(
    output_dir: Path,
    manifest_data: ManifestData,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_dir = output_dir / "splits"
    write_split_csv(split_dir / "train.csv", train_rows, manifest_data.fieldnames)
    write_split_csv(split_dir / "val.csv", val_rows, manifest_data.fieldnames)
    summary = split_summary(manifest_data.rows, train_rows, val_rows, args.val_size, args.seed)
    write_json(split_dir / "summary.json", summary)
    return summary


def config_from_args(args: argparse.Namespace, split: dict[str, Any], dataset_weights: dict[str, float]) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["class_to_idx"] = LABEL_TO_ID
    config["dataset_weights"] = dataset_weights
    config["split"] = split
    return config


def select_device(torch_module: Any, choice: str) -> Any:
    if choice == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch_module.device("cuda")
    if choice == "mps":
        if not hasattr(torch_module.backends, "mps") or not torch_module.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch_module.device("mps")
    if choice == "cpu":
        return torch_module.device("cpu")

    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def binary_roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

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


def weighted_binary_roc_auc(labels: list[int], scores: list[float], weights: list[float]) -> float:
    positive_weight = sum(weight for label, weight in zip(labels, weights) if label == 1)
    negative_weight = sum(weight for label, weight in zip(labels, weights) if label == 0)
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return math.nan

    pairs = sorted(zip(scores, labels, weights), key=lambda item: item[0])
    auc_numerator = 0.0
    cumulative_negative_weight = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        tied_positive_weight = sum(weight for _, label, weight in pairs[index:next_index] if label == 1)
        tied_negative_weight = sum(weight for _, label, weight in pairs[index:next_index] if label == 0)
        auc_numerator += tied_positive_weight * (cumulative_negative_weight + 0.5 * tied_negative_weight)
        cumulative_negative_weight += tied_negative_weight
        index = next_index

    return auc_numerator / (positive_weight * negative_weight)


def binary_metrics(labels: list[int], logits: list[float], loss: float) -> dict[str, float]:
    predictions = [1 if logit >= 0.0 else 0 for logit in logits]
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)

    total = max(1, len(labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": loss,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": binary_roc_auc(labels, logits),
    }


def weighted_binary_metrics(
    labels: list[int],
    logits: list[float],
    losses: list[float],
    weights: list[float],
) -> dict[str, float]:
    predictions = [1 if logit >= 0.0 else 0 for logit in logits]
    total_weight = max(1e-12, sum(weights))
    tp = sum(weight for pred, label, weight in zip(predictions, labels, weights) if pred == 1 and label == 1)
    tn = sum(weight for pred, label, weight in zip(predictions, labels, weights) if pred == 0 and label == 0)
    fp = sum(weight for pred, label, weight in zip(predictions, labels, weights) if pred == 1 and label == 0)
    fn = sum(weight for pred, label, weight in zip(predictions, labels, weights) if pred == 0 and label == 1)
    weighted_loss = sum(loss * weight for loss, weight in zip(losses, weights)) / total_weight
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": weighted_loss,
        "accuracy": (tp + tn) / total_weight,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": weighted_binary_roc_auc(labels, logits, weights),
    }


def collect_epoch_metrics(
    labels: list[int],
    logits: list[float],
    losses: list[float],
    weights: list[float],
    datasets: list[str],
) -> dict[str, Any]:
    overall_loss = sum(losses) / max(1, len(losses))
    by_dataset: dict[str, dict[str, float]] = {}
    for dataset in sorted(set(datasets)):
        indices = [index for index, item in enumerate(datasets) if item == dataset]
        dataset_labels = [labels[index] for index in indices]
        dataset_logits = [logits[index] for index in indices]
        dataset_losses = [losses[index] for index in indices]
        by_dataset[dataset] = binary_metrics(dataset_labels, dataset_logits, sum(dataset_losses) / max(1, len(dataset_losses)))
        by_dataset[dataset]["count"] = float(len(indices))
        by_dataset[dataset]["weight"] = weights[indices[0]]

    return {
        "overall": binary_metrics(labels, logits, overall_loss),
        "weighted": weighted_binary_metrics(labels, logits, losses, weights),
        "by_dataset": by_dataset,
    }


def flatten_split_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric_name in METRIC_NAMES:
        row[f"{prefix}_{metric_name}"] = metrics["overall"][metric_name]
        row[f"{prefix}_weighted_{metric_name}"] = metrics["weighted"][metric_name]

    for dataset, dataset_metrics in metrics["by_dataset"].items():
        dataset_key = safe_metric_key(dataset)
        row[f"{prefix}_dataset_{dataset_key}_count"] = int(dataset_metrics["count"])
        row[f"{prefix}_dataset_{dataset_key}_weight"] = dataset_metrics["weight"]
        row[f"{prefix}_dataset_{dataset_key}_accuracy"] = dataset_metrics["accuracy"]
    return row


def safe_metric_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def metric_row_fieldnames(row: dict[str, Any]) -> list[str]:
    base = [
        "epoch",
        *[f"train_{metric}" for metric in METRIC_NAMES],
        *[f"train_weighted_{metric}" for metric in METRIC_NAMES],
        *[f"val_{metric}" for metric in METRIC_NAMES],
        *[f"val_weighted_{metric}" for metric in METRIC_NAMES],
        "best_val_weighted_roc_auc",
        "best_val_roc_auc",
    ]
    return base + sorted(key for key in row if key not in base)


def append_metrics(path: Path, row: dict[str, Any], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    existing_fieldnames: list[str] = []
    if append and path.exists():
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fieldnames = metric_row_fieldnames(row)
    for fieldname in existing_fieldnames:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)


def write_tensorboard_metrics(
    writer: Any,
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_auc: float,
    lr: float,
) -> None:
    for metric_name in METRIC_NAMES:
        writer.add_scalar(f"{metric_name}/train", train_metrics["overall"][metric_name], epoch)
        writer.add_scalar(f"{metric_name}/val", val_metrics["overall"][metric_name], epoch)
        writer.add_scalar(f"{metric_name}/weighted/train", train_metrics["weighted"][metric_name], epoch)
        writer.add_scalar(f"{metric_name}/weighted/val", val_metrics["weighted"][metric_name], epoch)

    for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
        for dataset, metrics in split_metrics["by_dataset"].items():
            dataset_key = safe_metric_key(dataset)
            writer.add_scalar(f"accuracy/dataset/{dataset_key}/{split_name}", metrics["accuracy"], epoch)
            writer.add_scalar(f"dataset_count/{dataset_key}/{split_name}", metrics["count"], epoch)
            writer.add_scalar(f"dataset_weight/{dataset_key}", metrics["weight"], epoch)

    writer.add_scalar("roc_auc/best_val_weighted", best_auc, epoch)
    writer.add_scalar("lr", lr, epoch)
    writer.flush()


def import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.tensorboard import SummaryWriter
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are not installed. Install them with:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/pip install -r requirements-train.txt"
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    return {
        "torch": torch,
        "Image": Image,
        "nn": nn,
        "SummaryWriter": SummaryWriter,
        "DataLoader": DataLoader,
        "transforms": transforms,
        "EfficientNet_B0_Weights": EfficientNet_B0_Weights,
        "efficientnet_b0": efficientnet_b0,
        "tqdm": tqdm,
    }


def seed_training(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def build_transforms(deps: dict[str, Any], image_size: int) -> tuple[Any, Any]:
    transforms = deps["transforms"]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def build_model(deps: dict[str, Any], weights_name: str) -> Any:
    nn = deps["nn"]
    efficientnet_b0 = deps["efficientnet_b0"]
    weights_enum = deps["EfficientNet_B0_Weights"]
    weights = weights_enum.IMAGENET1K_V1 if weights_name == "imagenet" else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def run_epoch(
    deps: dict[str, Any],
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    train: bool,
    epoch: int,
) -> dict[str, float]:
    torch = deps["torch"]
    tqdm = deps["tqdm"]

    model.train(train)
    all_labels: list[int] = []
    all_logits: list[float] = []
    all_losses: list[float] = []
    all_weights: list[float] = []
    all_datasets: list[str] = []
    description = f"{'train' if train else 'val'} epoch {epoch}"
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=description, leave=False)

    for images, labels, sample_weights, dataset_names in iterator:
        images = images.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images).squeeze(1)
            per_sample_loss = criterion(logits, labels)
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            if train:
                loss.backward()
                optimizer.step()

        all_labels.extend(int(value) for value in labels.detach().cpu().tolist())
        all_logits.extend(float(value) for value in logits.detach().cpu().tolist())
        all_losses.extend(float(value) for value in per_sample_loss.detach().cpu().tolist())
        all_weights.extend(float(value) for value in sample_weights.detach().cpu().tolist())
        all_datasets.extend(str(value) for value in dataset_names)

    return collect_epoch_metrics(all_labels, all_logits, all_losses, all_weights, all_datasets)


def train_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    deps = import_training_dependencies()
    torch = deps["torch"]
    nn = deps["nn"]
    DataLoader = deps["DataLoader"]

    seed_training(torch, args.seed)
    device = select_device(torch, args.device)

    train_transform, val_transform = build_transforms(deps, args.image_size)
    train_dataset = FaceImageDataset(train_rows, train_transform)
    val_dataset = FaceImageDataset(val_rows, val_transform)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(deps, args.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_counts = Counter(int(row["label_id"]) for row in train_rows)
    pos_weight_value = train_counts[0] / max(1, train_counts[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device), reduction="none")

    start_epoch = 1
    best_auc = -math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_auc", best_auc))

    metrics_path = args.output_dir / "metrics.csv"
    append_metric_rows = args.resume is not None
    writer = None
    if not args.no_tensorboard:
        tensorboard_dir = args.tensorboard_dir if args.tensorboard_dir is not None else args.output_dir / "tensorboard"
        writer = deps["SummaryWriter"](log_dir=str(tensorboard_dir))
        writer.add_text("config", f"```json\n{json.dumps(config, ensure_ascii=False, indent=2)}\n```", 0)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(deps, model, train_loader, criterion, optimizer, device, True, epoch)
            val_metrics = run_epoch(deps, model, val_loader, criterion, optimizer, device, False, epoch)
            val_auc = val_metrics["overall"]["roc_auc"]
            val_weighted_auc = val_metrics["weighted"]["roc_auc"]
            selection_auc = val_weighted_auc if not math.isnan(val_weighted_auc) else val_auc
            if not math.isnan(selection_auc) and selection_auc > best_auc:
                best_auc = selection_auc
                save_checkpoint(args.output_dir / "best.pt", epoch, model, optimizer, best_auc, config, val_metrics, torch)

            save_checkpoint(args.output_dir / "last.pt", epoch, model, optimizer, best_auc, config, val_metrics, torch)
            metric_row = {
                "epoch": epoch,
                **flatten_split_metrics("train", train_metrics),
                **flatten_split_metrics("val", val_metrics),
                "val_roc_auc": val_auc,
                "best_val_weighted_roc_auc": best_auc,
                "best_val_roc_auc": best_auc,
            }
            append_metrics(metrics_path, metric_row, append=append_metric_rows)
            if writer is not None:
                write_tensorboard_metrics(writer, epoch, train_metrics, val_metrics, best_auc, args.lr)
            append_metric_rows = True
            print(
                "epoch "
                f"{epoch}/{args.epochs} "
                f"train_loss={train_metrics['overall']['loss']:.4f} "
                f"train_weighted_loss={train_metrics['weighted']['loss']:.4f} "
                f"val_loss={val_metrics['overall']['loss']:.4f} "
                f"val_weighted_loss={val_metrics['weighted']['loss']:.4f} "
                f"val_auc={val_auc:.4f} "
                f"val_weighted_auc={val_weighted_auc:.4f} "
                f"best_weighted_auc={best_auc:.4f}"
            )
    finally:
        if writer is not None:
            writer.close()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: Any,
    optimizer: Any,
    best_auc: float,
    config: dict[str, Any],
    metrics: dict[str, float],
    torch_module: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "config": config,
            "class_to_idx": LABEL_TO_ID,
            "metrics": metrics,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.runs_root = args.runs_root.resolve()
    args.output_dir = resolve_output_dir(args)

    manifest_data = read_manifest(args.dataset_dir)
    validate_image_paths(manifest_data.rows)
    explicit_dataset_weights = parse_dataset_weight_items(args.dataset_weight, args.default_dataset_weight)
    dataset_weights = apply_dataset_weights(manifest_data.rows, explicit_dataset_weights, args.default_dataset_weight)
    train_rows, val_rows = make_group_safe_split(manifest_data.rows, args.val_size, args.seed)
    split = write_split_artifacts(args.output_dir, manifest_data, train_rows, val_rows, args)
    config = config_from_args(args, split, dataset_weights)
    write_json(args.output_dir / "config.json", config)

    print(f"Output directory: {args.output_dir}")
    print(json.dumps(split, ensure_ascii=False, indent=2))
    if args.split_only:
        return 0

    train_model(args, config, train_rows, val_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
