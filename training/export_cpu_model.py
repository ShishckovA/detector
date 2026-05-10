#!/usr/bin/env python3

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from training.cpu_model_utils import (
    evaluate_model,
    import_cpu_dependencies,
    load_checkpoint_model,
    load_rows,
    write_json,
    write_predictions_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CPU inference artifacts from a face classifier checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/face_efficientnet_b0"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--skip-fx-int8", action="store_true")
    parser.add_argument("--skip-dynamic-int8", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def set_cpu_threads(torch: Any, threads: int, interop_threads: int) -> None:
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        pass


def select_quantized_engine(torch: Any) -> str:
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("fbgemm", "x86", "onednn", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    return torch.backends.quantized.engine


def export_torchscript(model: Any, example_input: Any, output_path: Path, torch: Any) -> None:
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, example_input, strict=False)
        exported = torch.jit.freeze(traced.eval())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(output_path))


def export_dynamic_int8(model: Any, example_input: Any, output_path: Path, torch: Any) -> tuple[Any, str | None]:
    try:
        select_quantized_engine(torch)
        quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)
        export_torchscript(quantized, example_input, output_path, torch)
        return quantized, None
    except Exception as exc:
        return None, str(exc)


def calibration_loader(rows: list[dict[str, Any]], image_size: int, calibration_size: int, deps: dict[str, Any]) -> Any:
    from training.cpu_model_utils import CsvImageDataset, make_val_transform

    DataLoader = deps["DataLoader"]
    transform = make_val_transform(deps, image_size)
    dataset = CsvImageDataset(rows[:calibration_size], transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def export_fx_static_int8(
    model: Any,
    rows: list[dict[str, Any]],
    image_size: int,
    calibration_size: int,
    output_path: Path,
    deps: dict[str, Any],
) -> tuple[Any, str | None]:
    torch = deps["torch"]
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        engine = select_quantized_engine(torch)
        example_input = (torch.randn(1, 3, image_size, image_size),)
        qconfig_mapping = get_default_qconfig_mapping(engine)
        prepared = prepare_fx(copy.deepcopy(model).cpu().eval(), qconfig_mapping, example_input)
        loader = calibration_loader(rows, image_size, calibration_size, deps)
        with torch.inference_mode():
            for images, _indices in loader:
                prepared(images)
        quantized = convert_fx(prepared).eval()
        export_torchscript(quantized, example_input[0], output_path, torch)
        return quantized, None
    except Exception as exc:
        return None, str(exc)


def artifact_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def prepare_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_dir = args.run_dir.resolve()
    checkpoint_path = (args.checkpoint or run_dir / "best.pt").resolve()
    val_csv = (args.val_csv or run_dir / "splits" / "val.csv").resolve()
    output_dir = (args.output_dir or run_dir / "cpu_export" / f"img{args.image_size}").resolve()
    return run_dir, checkpoint_path, val_csv, output_dir


def main() -> int:
    args = parse_args()
    _run_dir, checkpoint_path, val_csv, output_dir = prepare_paths(args)
    deps = import_cpu_dependencies()
    torch = deps["torch"]
    set_cpu_threads(torch, args.threads, args.interop_threads)
    device = torch.device("cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(val_csv)
    model = load_checkpoint_model(checkpoint_path, deps, device)
    example_input = torch.randn(1, 3, args.image_size, args.image_size)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "val_csv": str(val_csv),
        "image_size": args.image_size,
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "calibration_size": min(args.calibration_size, len(rows)),
        "artifacts": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    preprocess = {"image_size": args.image_size, "resize": 256, "center_crop": args.image_size}
    write_json(output_dir / "preprocess.json", preprocess)

    fp32_path = output_dir / "model_fp32_ts.pt"
    export_torchscript(model, example_input, fp32_path, torch)
    fp32_predictions, fp32_metrics = evaluate_model(model, rows, args.image_size, args.eval_batch_size, args.num_workers, deps)
    write_predictions_csv(output_dir / "predictions_fp32.csv", fp32_predictions)
    summary["artifacts"]["fp32_ts"] = {
        "path": str(fp32_path),
        "bytes": artifact_size(fp32_path),
        "metrics": fp32_metrics,
    }

    if not args.skip_dynamic_int8:
        dynamic_path = output_dir / "model_dynamic_int8_ts.pt"
        dynamic_model, error = export_dynamic_int8(model, example_input, dynamic_path, torch)
        if dynamic_model is None:
            summary["artifacts"]["dynamic_int8_ts"] = {"path": str(dynamic_path), "error": error}
        else:
            predictions, metrics = evaluate_model(dynamic_model, rows, args.image_size, args.eval_batch_size, args.num_workers, deps)
            write_predictions_csv(output_dir / "predictions_dynamic_int8.csv", predictions)
            summary["artifacts"]["dynamic_int8_ts"] = {
                "path": str(dynamic_path),
                "bytes": artifact_size(dynamic_path),
                "metrics": metrics,
            }

    if not args.skip_fx_int8:
        fx_path = output_dir / "model_fx_static_int8_ts.pt"
        fx_model, error = export_fx_static_int8(model, rows, args.image_size, args.calibration_size, fx_path, deps)
        if fx_model is None:
            summary["artifacts"]["fx_static_int8_ts"] = {"path": str(fx_path), "error": error}
        else:
            predictions, metrics = evaluate_model(fx_model, rows, args.image_size, args.eval_batch_size, args.num_workers, deps)
            write_predictions_csv(output_dir / "predictions_fx_static_int8.csv", predictions)
            summary["artifacts"]["fx_static_int8_ts"] = {
                "path": str(fx_path),
                "bytes": artifact_size(fx_path),
                "metrics": metrics,
            }

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
