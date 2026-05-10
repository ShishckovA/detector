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
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms


DEFAULT_CLASSIFIER_PATH = Path("runs/face_efficientnet_b0/cpu_export/img224/model_fp32_ts.pt")
DEFAULT_DETECTOR_PATH = Path("face_detection_yunet_2023mar.onnx")
DEFAULT_CLASS_LABELS = ("positive", "negative", "alex", "artem")
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
    class_labels: tuple[str, ...] = DEFAULT_CLASS_LABELS
    detector_score_threshold: float = 0.65
    detector_nms_threshold: float = 0.3
    detector_top_k: int = 5000
    detector_input_max_side: int = 1024
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
            faces = detect_faces(self.detector, bgr_image, self.settings.detector_input_max_side)
        detection_ms = elapsed_ms(detect_start)

        face = choose_best_face(faces)
        if face is None:
            return {
                "face_found": False,
                "reason": "no_face",
                "threshold": self.settings.classifier_threshold,
                "label": None,
                "score": None,
                "scores": None,
                "logits": None,
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
                "scores": None,
                "logits": None,
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
            classifier_output = self.classifier(tensor).reshape(-1)
            if classifier_output.numel() == 1:
                logit = float(classifier_output.item())
                score = float(self.torch.sigmoid(self.torch.tensor(logit)).item())
                label = "positive" if score >= self.settings.classifier_threshold else "negative"
                scores = {"negative": 1.0 - score, "positive": score}
                logits = {"positive": logit}
            else:
                if classifier_output.numel() != len(self.settings.class_labels):
                    raise ValueError(
                        "Classifier output size "
                        f"{classifier_output.numel()} does not match labels {self.settings.class_labels}"
                    )
                probabilities = self.torch.softmax(classifier_output, dim=0)
                best_index = int(self.torch.argmax(probabilities).item())
                label = self.settings.class_labels[best_index]
                score = float(probabilities[best_index].item())
                logit = float(classifier_output[best_index].item())
                scores = {
                    class_label: float(probabilities[index].item())
                    for index, class_label in enumerate(self.settings.class_labels)
                }
                logits = {
                    class_label: float(classifier_output[index].item())
                    for index, class_label in enumerate(self.settings.class_labels)
                }
        classification_ms = elapsed_ms(classify_start)

        return {
            "face_found": True,
            "bbox": crop_box_to_payload(crop_box),
            "detector_score": face.detector_score,
            "score": score,
            "logit": logit,
            "scores": scores,
            "logits": logits,
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


def env_class_labels(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise ValueError(f"{name} must contain at least one label")
    return labels


def load_settings() -> Settings:
    return Settings(
        classifier_model_path=env_path("FACE_CLASSIFIER_MODEL_PATH", DEFAULT_CLASSIFIER_PATH),
        detector_model_path=env_path("FACE_DETECTOR_MODEL_PATH", DEFAULT_DETECTOR_PATH),
        torch_num_threads=env_int("TORCH_NUM_THREADS", 1),
        torch_num_interop_threads=env_int("TORCH_NUM_INTEROP_THREADS", 1),
        max_concurrency=env_int("INFERENCE_MAX_CONCURRENCY", 2),
        classifier_threshold=env_float("FACE_CLASSIFIER_THRESHOLD", 0.65),
        class_labels=env_class_labels("FACE_CLASS_LABELS", DEFAULT_CLASS_LABELS),
        detector_score_threshold=env_float("FACE_DETECTOR_SCORE_THRESHOLD", 0.65),
        detector_nms_threshold=env_float("FACE_DETECTOR_NMS_THRESHOLD", 0.3),
        detector_top_k=env_int("FACE_DETECTOR_TOP_K", 5000),
        detector_input_max_side=env_int("FACE_DETECTOR_INPUT_MAX_SIDE", 1024),
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


def resize_for_detection(bgr_image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    image_height, image_width = bgr_image.shape[:2]
    longest_side = max(image_width, image_height)
    if max_side <= 0 or longest_side <= max_side:
        return bgr_image, 1.0

    scale = max_side / longest_side
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv.resize(bgr_image, (resized_width, resized_height), interpolation=cv.INTER_AREA)
    return resized, scale


def detect_faces(detector: Any, bgr_image: np.ndarray, max_side: int = 1024) -> list[FaceCandidate]:
    detector_image, scale = resize_for_detection(bgr_image, max_side)
    image_height, image_width = detector_image.shape[:2]
    detector.setInputSize((image_width, image_height))
    _, raw_faces = detector.detect(detector_image)
    if raw_faces is None:
        return []
    return [
        FaceCandidate(
            bbox_x=float(face[0]) / scale,
            bbox_y=float(face[1]) / scale,
            bbox_w=float(face[2]) / scale,
            bbox_h=float(face[3]) / scale,
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
            return ImageOps.exif_transpose(image).convert("RGB")
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
        "class_labels": list(settings.class_labels),
        "detector_input_max_side": settings.detector_input_max_side,
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
