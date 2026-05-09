#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from training.cpu_model_utils import import_cpu_dependencies, make_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU inference for one cropped face image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("runs/face_efficientnet_b0/cpu_export/img192/model_fp32_ts.pt"))
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


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
    transform = make_val_transform(deps, args.image_size)
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        logit = float(model(tensor).squeeze().item())
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

    result = {
        "image": str(args.image.resolve()),
        "model": str(args.model.resolve()),
        "image_size": args.image_size,
        "threshold": args.threshold,
        "logit": logit,
        "prob_positive": prob,
        "pred_label": "positive" if prob >= args.threshold else "negative",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
