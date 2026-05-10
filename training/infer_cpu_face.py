#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from training.cpu_model_utils import import_cpu_dependencies, make_val_transform

DEFAULT_CLASS_LABELS = ("positive", "negative", "alex", "artem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU inference for one cropped face image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("runs/face_efficientnet_b0/cpu_export/img224/model_fp32_ts.pt"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class-labels", default=",".join(DEFAULT_CLASS_LABELS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def parse_class_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise ValueError("--class-labels must contain at least one label")
    return labels


def main() -> int:
    args = parse_args()
    deps = import_cpu_dependencies()
    torch = deps["torch"]
    Image = deps["Image"]
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(max(1, args.interop_threads))
    except RuntimeError:
        pass

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = torch.jit.load(str(args.model.resolve()), map_location="cpu")
    model.eval()
    class_labels = parse_class_labels(args.class_labels)
    transform = make_val_transform(deps, args.image_size)
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        output = model(tensor).reshape(-1)
        if output.numel() == 1:
            logit = float(output.item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred_label = "positive" if prob >= args.threshold else "negative"
            scores = {"negative": 1.0 - prob, "positive": prob}
            logits = {"positive": logit}
        else:
            if output.numel() != len(class_labels):
                raise ValueError(f"Model returned {output.numel()} logits, but --class-labels has {len(class_labels)} labels")
            probabilities = torch.softmax(output, dim=0)
            pred_id = int(torch.argmax(probabilities).item())
            pred_label = class_labels[pred_id]
            prob = float(probabilities[pred_id].item())
            logit = float(output[pred_id].item())
            scores = {label: float(probabilities[index].item()) for index, label in enumerate(class_labels)}
            logits = {label: float(output[index].item()) for index, label in enumerate(class_labels)}

    result = {
        "image": str(args.image.resolve()),
        "model": str(args.model.resolve()),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "logit": logit,
        "score": prob,
        "scores": scores,
        "logits": logits,
        "pred_label": pred_label,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
