#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CPU benchmark JSON files.")
    parser.add_argument("bench_dir", type=Path, nargs="?", default=Path("runs/face_efficientnet_b0/cpu_bench"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(args.bench_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": path.stem.rsplit("_t", 1)[0],
                "threads": data["threads"],
                "concurrency": data["concurrency"],
                "image_size": data["image_size"],
                "model_p50": data["model_only"]["median_ms"],
                "model_p95": data["model_only"]["p95_ms"],
                "e2e_p50": data["end_to_end"]["median_ms"],
                "e2e_p95": data["end_to_end"]["p95_ms"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No benchmark JSON files found in {args.bench_dir}")

    rows.sort(key=lambda row: (row["concurrency"], row["name"], row["threads"]))
    print("name,threads,conc,img,model_p50,model_p95,e2e_p50,e2e_p95")
    for row in rows:
        print(
            f"{row['name']},{row['threads']},{row['concurrency']},{row['image_size']},"
            f"{row['model_p50']:.2f},{row['model_p95']:.2f},{row['e2e_p50']:.2f},{row['e2e_p95']:.2f}"
        )

    print("\nBEST_BY_CONC_MODEL_P95")
    for concurrency in (1, 2):
        candidates = [row for row in rows if row["concurrency"] == concurrency]
        best = min(candidates, key=lambda row: row["model_p95"])
        print(
            f"conc={concurrency}: {best['name']} threads={best['threads']} img={best['image_size']} "
            f"model_p50={best['model_p50']:.2f} model_p95={best['model_p95']:.2f} e2e_p95={best['e2e_p95']:.2f}"
        )

    print("\nBEST_BY_CONC_E2E_P95")
    for concurrency in (1, 2):
        candidates = [row for row in rows if row["concurrency"] == concurrency]
        best = min(candidates, key=lambda row: row["e2e_p95"])
        print(
            f"conc={concurrency}: {best['name']} threads={best['threads']} img={best['image_size']} "
            f"model_p50={best['model_p50']:.2f} model_p95={best['model_p95']:.2f} e2e_p95={best['e2e_p95']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
