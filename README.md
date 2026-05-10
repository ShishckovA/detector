# Детектор Общительных

Production layout:

- `backend/` - FastAPI inference API.
- `frontend/` - public landing page static assets.
- `nginx/` - public reverse proxy.
- `training/` - model training, validation demos, CPU export, and CPU benchmarks.
- `tools/` - data collection and dataset preparation scripts.
- `data/` - dataset manifests and source metadata. Image files are intentionally git-ignored.
- `reports/` - generated CSV/HTML analysis reports that are useful to keep.
- `docs/` - project notes.
- `docker-compose.yml` - production composition. Only nginx is published outside Docker.

## Run

The model artifacts must exist on the host:

- `runs/face_efficientnet_b0/cpu_export/img224/model_fp32_ts.pt`
- `face_detection_yunet_2023mar.onnx`

```bash
docker compose up -d --build
```

By default nginx is published on host port `8000`:

```bash
curl http://127.0.0.1:8000/health
```

To use a different public port:

```bash
PUBLIC_PORT=80 docker compose up -d --build
```

Backend is not exposed on a host port. It is reachable only from the internal Docker network as `backend:8000`; nginx proxies `/health` and `/api/*` to it.

## ML Utilities

Training dependencies:

```bash
.venv/bin/pip install -r training/requirements-train.txt
```

Training entry point:

```bash
python tools/build_face_dataset.py --clean --out data/face_dataset_png
python -m training.train_face_classifier --dataset-dir data/face_dataset_png
```

The current classifier uses four classes: `positive`, `negative`, `alex`, `artem`.
Images under `data/positives_original/us/alex` and `data/positives_original/us/artem`
are exported as separate classes with higher sample weight by default.
Face detection applies EXIF orientation and downsizes large images to max side `1024`
before YuNet, then maps the detected box back to original coordinates.

CPU export dependencies:

```bash
.venv/bin/pip install -r training/requirements-cpu.txt
```

CPU export entry point:

```bash
python -m training.export_cpu_model --run-dir runs/face_efficientnet_b0
```

Face crop/report generation:

```bash
python tools/extract_face_crops.py data/raw --out reports/face_scores
```
