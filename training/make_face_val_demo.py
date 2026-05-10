#!/usr/bin/env python3

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training.train_face_classifier import (
    LABEL_TO_ID,
    build_model,
    build_transforms,
    class_metric_row,
    import_training_dependencies,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML validation demo for the face classifier.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/face_efficientnet_b0"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thumbnail-size", type=int, default=224)
    parser.add_argument("--force-thumbnails", action="store_true")
    return parser.parse_args()


class ValDataset:
    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import torch
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["resolved_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, index


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in validation CSV: {path}")
    for index, row in enumerate(rows):
        if "resolved_path" not in row or "label_id" not in row:
            raise ValueError(f"Validation row {index} is missing resolved_path or label_id")
    return rows


def make_thumbnail(src: Path, dst: Path, size: int, force: bool) -> None:
    if dst.exists() and not force:
        return
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (245, 246, 248))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        canvas.save(dst, quality=90)


def run_inference(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    image_size: int,
    device_choice: str,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    deps = import_training_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    device = select_device(torch, device_choice)

    _, val_transform = build_transforms(deps, image_size)
    dataset = ValDataset(rows, val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx") or checkpoint.get("config", {}).get("class_to_idx") or LABEL_TO_ID
    id_to_label = {int(index): label for label, index in class_to_idx.items()}
    classifier_weight = checkpoint["model_state_dict"].get("classifier.1.weight")
    num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else len(class_to_idx)
    model = build_model(deps, "random", num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    predictions = [dict(row) for row in rows]
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            logits_list = logits.detach().cpu().tolist()
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0]).detach().cpu().tolist()
                for index, logit_values, prob in zip(indices.tolist(), logits_list, probs):
                    label_id = int(predictions[index]["label_id"])
                    pred_id = 1 if float(prob) >= 0.5 else 0
                    predictions[index]["logit_positive"] = float(logit_values[0])
                    predictions[index]["prob_negative"] = 1.0 - float(prob)
                    predictions[index]["prob_positive"] = float(prob)
                    predictions[index]["score"] = float(prob) if pred_id == 1 else 1.0 - float(prob)
                    predictions[index]["pred_label_id"] = pred_id
                    predictions[index]["pred_label"] = "positive" if pred_id == 1 else "negative"
                    predictions[index]["correct"] = pred_id == label_id
                continue

            probs = torch.softmax(logits, dim=1).detach().cpu().tolist()
            for index, logit_values, prob_values in zip(indices.tolist(), logits_list, probs):
                label_id = int(predictions[index]["label_id"])
                pred_id = max(range(len(prob_values)), key=lambda class_index: prob_values[class_index])
                predictions[index]["pred_label_id"] = pred_id
                predictions[index]["pred_label"] = id_to_label.get(pred_id, str(pred_id))
                predictions[index]["score"] = float(prob_values[pred_id])
                predictions[index]["correct"] = pred_id == label_id
                for class_id, class_name in id_to_label.items():
                    predictions[index][f"logit_{class_name}"] = float(logit_values[class_id])
                    predictions[index][f"prob_{class_name}"] = float(prob_values[class_id])

    return predictions


def summarize(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if predictions and "prob_positive" in predictions[0] and len({row["pred_label"] for row in predictions}) <= 2:
        for row in predictions:
            pred_id = 1 if float(row["prob_positive"]) >= threshold else 0
            row["pred_label_id"] = pred_id
            row["pred_label"] = "positive" if pred_id == 1 else "negative"
            row["score"] = float(row["prob_positive"]) if pred_id == 1 else 1.0 - float(row["prob_positive"])
            row["correct"] = pred_id == int(row["label_id"])

    labels = [int(row["label_id"]) for row in predictions]
    pred_ids = [int(row["pred_label_id"]) for row in predictions]
    label_names: dict[int, str] = {}
    for row in predictions:
        label_names[int(row["label_id"])] = row["label"]
        label_names[int(row["pred_label_id"])] = row["pred_label"]

    counts = Counter((label, pred_id) for label, pred_id in zip(labels, pred_ids))
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    confusion: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, float]] = {}
    for class_id, class_name in sorted(label_names.items()):
        tp = counts[(class_id, class_id)]
        fp = sum(counts[(other, class_id)] for other in label_names if other != class_id)
        fn = sum(counts[(class_id, other)] for other in label_names if other != class_id)
        support = sum(1 for label in labels if label == class_id)
        predicted = sum(1 for pred_id in pred_ids if pred_id == class_id)
        per_class[class_name] = class_metric_row(float(tp), float(fp), float(fn), float(support), float(predicted))
        precisions.append(per_class[class_name]["precision"])
        recalls.append(per_class[class_name]["recall"])
        f1s.append(per_class[class_name]["f1"])
        confusion[class_name] = {
            label_names[pred_id]: counts[(class_id, pred_id)]
            for pred_id in sorted(label_names)
        }

    return {
        "rows": len(predictions),
        "threshold": threshold,
        "accuracy": sum(1 for row in predictions if row["correct"]) / max(1, len(predictions)),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "confusion": confusion,
        "labels": dict(Counter(row["label"] for row in predictions)),
        "mistakes": sum(1 for row in predictions if not row["correct"]),
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


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def row_card(row: dict[str, Any]) -> str:
    correct = bool(row["correct"])
    true_label = row["label"]
    pred_label = row["pred_label"]
    outcome = "correct" if correct else "mistake"
    confidence = float(row.get("score", 0.0))
    status = "correct" if correct else "mistake"
    title = html.escape(row.get("dataset_path", ""))
    group = html.escape(row.get("group_key", ""))
    source = html.escape(row.get("source_path", ""))
    image = html.escape(row["thumbnail_path"])
    return f"""
      <article class="card {status}" data-status="{status}" data-outcome="{outcome}" data-true="{html.escape(true_label)}" data-pred="{html.escape(pred_label)}" data-group="{group}" data-path="{title}">
        <a class="thumb" href="{image}"><img src="{image}" loading="lazy" alt=""></a>
        <div class="card-body">
          <div class="row">
            <span class="badge {status}">{outcome.upper()}</span>
            <span class="prob">score {confidence:.3f}</span>
          </div>
          <div class="labels">
            <span>true <b>{html.escape(true_label)}</b></span>
            <span>pred <b>{html.escape(pred_label)}</b></span>
            <span>conf <b>{confidence:.3f}</b></span>
          </div>
          <div class="meta" title="{title}">{title}</div>
          <div class="meta" title="{source}">{source}</div>
          <div class="group">{group}</div>
        </div>
      </article>
    """


def render_section(title: str, rows: list[dict[str, Any]], empty_text: str) -> str:
    if not rows:
        body = f'<p class="empty">{html.escape(empty_text)}</p>'
    else:
        body = '<div class="grid">' + "\n".join(row_card(row) for row in rows) + "</div>"
    return f"<section><h2>{html.escape(title)}</h2>{body}</section>"


def write_html(path: Path, predictions: list[dict[str, Any]], summary: dict[str, Any], checkpoint_path: Path) -> None:
    def row_score(row: dict[str, Any]) -> float:
        return float(row.get("score", 0.0))

    sorted_rows = sorted(
        predictions,
        key=lambda row: (
            bool(row["correct"]),
            -row_score(row),
        ),
    )
    mistakes = [row for row in sorted_rows if not row["correct"]]
    uncertain = sorted(predictions, key=row_score)[:48]
    confident_correct = sorted(
        [row for row in predictions if row["correct"]],
        key=row_score,
        reverse=True,
    )[:48]

    confusion = summary["confusion"]
    confusion_labels = list(confusion)
    confusion_header = "".join(f"<th>{html.escape(label)}</th>" for label in confusion_labels)
    confusion_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(true_label)}</th>"
        + "".join(f"<td>{confusion[true_label].get(pred_label, 0)}</td>" for pred_label in confusion_labels)
        + "</tr>"
        for true_label in confusion_labels
    )
    label_buttons = "\n".join(
        f'<button data-filter="{html.escape(label)}">True {html.escape(label)} ({count})</button>'
        for label, count in sorted(summary["labels"].items())
    )
    per_class_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(class_name)}</th>"
        f"<td>{fmt_float(metrics['precision'])}</td>"
        f"<td>{fmt_float(metrics['recall'])}</td>"
        f"<td>{fmt_float(metrics['f1'])}</td>"
        f"<td>{int(metrics['support'])}</td>"
        f"<td>{int(metrics['predicted'])}</td>"
        "</tr>"
        for class_name, metrics in summary["per_class"].items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face EfficientNet-B0 Validation Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f6f8;
      --text: #15171a;
      --muted: #626a73;
      --line: #d8dde3;
      --ok: #16784f;
      --bad: #bd2f2f;
      --panel: #ffffff;
      --accent: #2454a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 24px 28px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .sub {{ color: var(--muted); max-width: 1100px; overflow-wrap: anywhere; }}
    main {{ padding: 0 28px 32px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ color: var(--muted); }}
    .confusion-table {{ border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); overflow: hidden; }}
    .confusion-table th, .confusion-table td {{ border: 1px solid var(--line); padding: 9px 12px; text-align: right; }}
    .confusion-table th:first-child {{ text-align: left; }}
    .per-class-table {{ border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); margin-top: 12px; }}
    .per-class-table th, .per-class-table td {{ border: 1px solid var(--line); padding: 9px 12px; text-align: right; }}
    .per-class-table th:first-child {{ text-align: left; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 18px 0; }}
    button {{ border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 8px 10px; cursor: pointer; }}
    button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    input {{ width: min(520px, 100%); border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .card.mistake {{ border-color: color-mix(in srgb, var(--bad) 55%, var(--line)); }}
    .thumb {{ display: block; background: #eef1f4; aspect-ratio: 1 / 1; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    .card-body {{ padding: 10px; }}
    .row {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; }}
    .badge {{ border-radius: 999px; padding: 2px 8px; font-weight: 700; font-size: 12px; }}
    .badge.correct {{ color: var(--ok); background: #e8f4ef; }}
    .badge.mistake {{ color: var(--bad); background: #fdecec; }}
    .prob {{ font-variant-numeric: tabular-nums; color: var(--muted); }}
    .labels {{ display: grid; gap: 3px; margin-bottom: 8px; }}
    .meta, .group {{ color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .group {{ color: var(--text); margin-top: 4px; }}
    .empty {{ color: var(--muted); }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>Face EfficientNet-B0 Validation Demo</h1>
    <div class="sub">Checkpoint: {html.escape(str(checkpoint_path))}</div>
    <div class="sub">Validation rows: {summary["rows"]}; labels: {html.escape(json.dumps(summary["labels"], ensure_ascii=False))}</div>
    <div class="metrics">
      <div class="metric"><strong>{fmt_float(summary["accuracy"])}</strong><span>accuracy</span></div>
      <div class="metric"><strong>{fmt_float(summary["precision"])}</strong><span>macro precision</span></div>
      <div class="metric"><strong>{fmt_float(summary["recall"])}</strong><span>macro recall</span></div>
      <div class="metric"><strong>{fmt_float(summary["f1"])}</strong><span>F1</span></div>
      <div class="metric"><strong>{summary["mistakes"]}</strong><span>mistakes</span></div>
    </div>
  </header>
  <main>
    <section>
      <h2>Confusion Matrix</h2>
      <table class="confusion-table">
        <thead><tr><th>true \\ pred</th>{confusion_header}</tr></thead>
        <tbody>{confusion_rows}</tbody>
      </table>
      <h2>Per-Class Metrics</h2>
      <table class="per-class-table">
        <thead><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th><th>predicted</th></tr></thead>
        <tbody>{per_class_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>All Validation Examples</h2>
      <div class="controls">
        <button class="active" data-filter="all">All ({summary["rows"]})</button>
        <button data-filter="mistake">Mistakes ({summary["mistakes"]})</button>
        <button data-filter="correct">Correct ({summary["rows"] - summary["mistakes"]})</button>
        {label_buttons}
        <input id="search" type="search" placeholder="Search path or group">
      </div>
      <div id="cards" class="grid">
        {"".join(row_card(row) for row in sorted_rows)}
      </div>
    </section>
    <div id="derived-sections">
      {render_section("Mistakes", mistakes[:96], "No mistakes.")}
      {render_section("Lowest Confidence", uncertain, "No examples.")}
      {render_section("Most Confident Correct", confident_correct, "No correct examples.")}
    </div>
  </main>
  <script>
    const buttons = [...document.querySelectorAll('button[data-filter]')];
    const cards = [...document.querySelectorAll('#cards .card')];
    const search = document.querySelector('#search');
    const derivedSections = document.querySelector('#derived-sections');
    let active = 'all';
    function applyFilter() {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const statusOk = active === 'all' || card.dataset.outcome === active || card.dataset.status === active || card.dataset.true === active;
        const textOk = !query || (card.dataset.path + ' ' + card.dataset.group).toLowerCase().includes(query);
        card.classList.toggle('hidden', !(statusOk && textOk));
      }}
      derivedSections.classList.toggle('hidden', active !== 'all' || Boolean(query));
    }}
    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        active = button.dataset.filter;
        buttons.forEach(item => item.classList.toggle('active', item === button));
        applyFilter();
      }});
    }}
    search.addEventListener('input', applyFilter);
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def prepare_demo(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    checkpoint_path = (args.checkpoint or run_dir / "best.pt").resolve()
    val_csv = (args.val_csv or run_dir / "splits" / "val.csv").resolve()
    output_dir = (args.output_dir or run_dir / "val_demo").resolve()
    image_dir = output_dir / "images"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    rows = load_rows(val_csv)
    config_path = run_dir / "config.json"
    image_size = 224
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        image_size = int(config.get("image_size", image_size))

    predictions = run_inference(rows, checkpoint_path, image_size, args.device, args.batch_size, args.num_workers)
    summary = summarize(predictions, args.threshold)
    for index, row in enumerate(predictions):
        thumb_name = f"{index:04d}_{row['label']}_{row['pred_label']}.jpg"
        thumb_path = image_dir / thumb_name
        make_thumbnail(Path(row["resolved_path"]), thumb_path, args.thumbnail_size, args.force_thumbnails)
        row["thumbnail_path"] = str(thumb_path.relative_to(output_dir))

    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_json(output_dir / "summary.json", summary)
    write_html(output_dir / "index.html", predictions, summary, checkpoint_path)
    return output_dir / "index.html"


def main() -> int:
    args = parse_args()
    html_path = prepare_demo(args)
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
