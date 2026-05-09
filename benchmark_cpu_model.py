#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cpu_model_utils import (
    benchmark_callable,
    import_cpu_dependencies,
    latency_summary,
    load_checkpoint_model,
    load_rows,
    make_val_transform,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-image CPU latency for an exported TorchScript model.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, help="Exported TorchScript model.")
    model_group.add_argument("--checkpoint", type=Path, help="Training checkpoint with model_state_dict.")
    parser.add_argument("--compile", action="store_true", help="Compile checkpoint model with torch.compile at startup.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--val-csv", type=Path, default=Path("runs/face_efficientnet_b0/splits/val.csv"))
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def set_cpu_threads(torch: Any, threads: int, interop_threads: int) -> None:
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        pass


def load_tensor(row: dict[str, Any], image_size: int, deps: dict[str, Any]) -> Any:
    from PIL import Image

    transform = make_val_transform(deps, image_size)
    with Image.open(row["resolved_path"]) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def main() -> int:
    args = parse_args()
    deps = import_cpu_dependencies()
    torch = deps["torch"]
    set_cpu_threads(torch, args.threads, args.interop_threads)

    if args.model is not None and not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    rows = load_rows(args.val_csv.resolve())
    row = rows[args.sample_index % len(rows)]
    tensor = load_tensor(row, args.image_size, deps)
    compile_startup_ms = None
    if args.model is not None:
        model = torch.jit.load(str(args.model.resolve()), map_location="cpu")
    else:
        model = load_checkpoint_model(args.checkpoint.resolve(), deps, torch.device("cpu"))
        if args.compile:
            import time

            start = time.perf_counter()
            model = torch.compile(model, mode=args.compile_mode)
            with torch.inference_mode():
                model(tensor)
            compile_startup_ms = (time.perf_counter() - start) * 1000.0
    model.eval()

    def model_only() -> None:
        with torch.inference_mode():
            model(tensor)

    def end_to_end() -> None:
        image_tensor = load_tensor(row, args.image_size, deps)
        with torch.inference_mode():
            model(image_tensor)

    if args.concurrency == 1:
        model_latency = benchmark_callable(model_only, args.warmup, args.iterations)
        e2e_latency = benchmark_callable(end_to_end, max(1, args.warmup // 4), max(1, args.iterations // 4))
    else:
        def timed(fn: Any) -> float:
            import time

            start = time.perf_counter()
            fn()
            return (time.perf_counter() - start) * 1000.0

        def run_parallel(fn: Any) -> list[float]:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(timed, fn) for _ in range(2)]
                return [future.result() for future in futures]

        for _ in range(args.warmup):
            run_parallel(model_only)
        values_ms: list[float] = []
        for _ in range(args.iterations):
            values_ms.extend(run_parallel(model_only))
        model_latency = latency_summary(values_ms)

        for _ in range(max(1, args.warmup // 4)):
            run_parallel(end_to_end)
        e2e_values_ms: list[float] = []
        for _ in range(max(1, args.iterations // 4)):
            e2e_values_ms.extend(run_parallel(end_to_end))
        e2e_latency = latency_summary(e2e_values_ms)

    result = {
        "model": str(args.model.resolve()) if args.model is not None else None,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "compiled": bool(args.compile),
        "compile_mode": args.compile_mode if args.compile else None,
        "compile_startup_ms": compile_startup_ms,
        "val_csv": str(args.val_csv.resolve()),
        "sample_path": row["resolved_path"],
        "image_size": args.image_size,
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "model_only": model_latency,
        "end_to_end": e2e_latency,
    }

    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
