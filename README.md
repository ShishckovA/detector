# Детектор Общительных

Production layout:

- `backend/` - FastAPI inference API.
- `frontend/` - public landing page static assets.
- `nginx/` - public reverse proxy.
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
