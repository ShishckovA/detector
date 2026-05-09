#!/usr/bin/env python3

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_CLASSIFIER_PATH = Path("runs/face_efficientnet_b0/cpu_export/img224/model_fp32_ts.pt")
DEFAULT_DETECTOR_PATH = Path("face_detection_yunet_2023mar.onnx")
IMAGE_SIZE = 224
RESIZE_SIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Settings:
    classifier_model_path: Path
    detector_model_path: Path
    torch_num_threads: int = 1
    torch_num_interop_threads: int = 1
    max_concurrency: int = 2
    classifier_threshold: float = 0.65
    detector_score_threshold: float = 0.65
    detector_nms_threshold: float = 0.3
    detector_top_k: int = 5000
    crop_margin: float = 0.25
    min_face_size: int = 48


@dataclass(frozen=True)
class FaceCandidate:
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detector_score: float

    @property
    def area(self) -> float:
        return max(0.0, self.bbox_w) * max(0.0, self.bbox_h)


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class InferenceService:
    def __init__(
        self,
        settings: Settings,
        detector: Any,
        classifier: Any,
        preprocess: Any,
        torch_module: Any = torch,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.classifier = classifier
        self.preprocess = preprocess
        self.torch = torch_module
        self.detector_lock = threading.Lock()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        total_start = time.perf_counter()
        rgb_image = image.convert("RGB")
        bgr_image = pil_to_bgr(rgb_image)

        detect_start = time.perf_counter()
        with self.detector_lock:
            faces = detect_faces(self.detector, bgr_image)
        detection_ms = elapsed_ms(detect_start)

        face = choose_best_face(faces)
        if face is None:
            return {
                "face_found": False,
                "reason": "no_face",
                "threshold": self.settings.classifier_threshold,
                "label": None,
                "score": None,
                "bbox": None,
                "detector_score": None,
                "timings_ms": {
                    "detection": detection_ms,
                    "classification": 0.0,
                    "total": elapsed_ms(total_start),
                },
            }

        crop_box = crop_box_for_face(
            face=face,
            image_width=rgb_image.width,
            image_height=rgb_image.height,
            margin=self.settings.crop_margin,
        )
        if crop_box.width < self.settings.min_face_size or crop_box.height < self.settings.min_face_size:
            return {
                "face_found": False,
                "reason": "face_too_small",
                "threshold": self.settings.classifier_threshold,
                "label": None,
                "score": None,
                "bbox": crop_box_to_payload(crop_box),
                "detector_score": face.detector_score,
                "timings_ms": {
                    "detection": detection_ms,
                    "classification": 0.0,
                    "total": elapsed_ms(total_start),
                },
            }

        crop = rgb_image.crop((crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
        tensor = self.preprocess(crop).unsqueeze(0)

        classify_start = time.perf_counter()
        with self.torch.inference_mode():
            logit = float(self.classifier(tensor).squeeze().item())
            score = float(self.torch.sigmoid(self.torch.tensor(logit)).item())
        classification_ms = elapsed_ms(classify_start)
        label = "positive" if score >= self.settings.classifier_threshold else "negative"

        return {
            "face_found": True,
            "bbox": crop_box_to_payload(crop_box),
            "detector_score": face.detector_score,
            "score": score,
            "logit": logit,
            "threshold": self.settings.classifier_threshold,
            "label": label,
            "timings_ms": {
                "detection": detection_ms,
                "classification": classification_ms,
                "total": elapsed_ms(total_start),
            },
        }


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def load_settings() -> Settings:
    return Settings(
        classifier_model_path=env_path("FACE_CLASSIFIER_MODEL_PATH", DEFAULT_CLASSIFIER_PATH),
        detector_model_path=env_path("FACE_DETECTOR_MODEL_PATH", DEFAULT_DETECTOR_PATH),
        torch_num_threads=env_int("TORCH_NUM_THREADS", 1),
        torch_num_interop_threads=env_int("TORCH_NUM_INTEROP_THREADS", 1),
        max_concurrency=env_int("INFERENCE_MAX_CONCURRENCY", 2),
        classifier_threshold=env_float("FACE_CLASSIFIER_THRESHOLD", 0.65),
        detector_score_threshold=env_float("FACE_DETECTOR_SCORE_THRESHOLD", 0.65),
        detector_nms_threshold=env_float("FACE_DETECTOR_NMS_THRESHOLD", 0.3),
        detector_top_k=env_int("FACE_DETECTOR_TOP_K", 5000),
        crop_margin=env_float("FACE_CROP_MARGIN", 0.25),
        min_face_size=env_int("FACE_MIN_SIZE", 48),
    )


def configure_torch_threads(settings: Settings) -> None:
    torch.set_num_threads(max(1, settings.torch_num_threads))
    try:
        torch.set_num_interop_threads(max(1, settings.torch_num_interop_threads))
    except RuntimeError:
        pass


def make_preprocess() -> Any:
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def load_inference_service(settings: Settings) -> InferenceService:
    if not settings.classifier_model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {settings.classifier_model_path}")
    if not settings.detector_model_path.exists():
        raise FileNotFoundError(f"YuNet detector model not found: {settings.detector_model_path}")

    configure_torch_threads(settings)
    classifier = torch.jit.load(str(settings.classifier_model_path.resolve()), map_location="cpu")
    classifier.eval()
    detector = cv.FaceDetectorYN.create(
        model=str(settings.detector_model_path.resolve()),
        config="",
        input_size=(1, 1),
        score_threshold=settings.detector_score_threshold,
        nms_threshold=settings.detector_nms_threshold,
        top_k=settings.detector_top_k,
    )
    return InferenceService(settings, detector, classifier, make_preprocess())


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)


def detect_faces(detector: Any, bgr_image: np.ndarray) -> list[FaceCandidate]:
    image_height, image_width = bgr_image.shape[:2]
    detector.setInputSize((image_width, image_height))
    _, raw_faces = detector.detect(bgr_image)
    if raw_faces is None:
        return []
    return [
        FaceCandidate(
            bbox_x=float(face[0]),
            bbox_y=float(face[1]),
            bbox_w=float(face[2]),
            bbox_h=float(face[3]),
            detector_score=float(face[14]),
        )
        for face in raw_faces
    ]


def choose_best_face(faces: list[FaceCandidate]) -> FaceCandidate | None:
    if not faces:
        return None
    return max(faces, key=lambda face: (face.detector_score, face.area))


def crop_box_for_face(face: FaceCandidate, image_width: int, image_height: int, margin: float) -> CropBox:
    pad_x = face.bbox_w * margin
    pad_y = face.bbox_h * margin
    x1 = max(0, int(face.bbox_x - pad_x))
    y1 = max(0, int(face.bbox_y - pad_y))
    x2 = min(image_width, int(face.bbox_x + face.bbox_w + pad_x))
    y2 = min(image_height, int(face.bbox_y + face.bbox_h + pad_y))
    return CropBox(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_box_to_payload(crop_box: CropBox) -> dict[str, int]:
    return {
        "x1": crop_box.x1,
        "y1": crop_box.y1,
        "x2": crop_box.x2,
        "y2": crop_box.y2,
        "width": crop_box.width,
        "height": crop_box.height,
    }


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def decode_image(data: bytes) -> Image.Image:
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.service = load_inference_service(settings)
    app.state.inference_semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
    yield


app = FastAPI(title="CPU Face Inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    settings: Settings = app.state.settings
    return {
        "status": "ok",
        "classifier_model_path": str(settings.classifier_model_path),
        "detector_model_path": str(settings.detector_model_path),
        "torch_num_threads": settings.torch_num_threads,
        "torch_num_interop_threads": settings.torch_num_interop_threads,
        "max_concurrency": settings.max_concurrency,
    }


@app.post("/api/face-score")
async def face_score(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image = decode_image(data)
    service: InferenceService = app.state.service
    semaphore: asyncio.Semaphore = app.state.inference_semaphore
    async with semaphore:
        result = await asyncio.to_thread(service.predict, image)
    return JSONResponse(result)
