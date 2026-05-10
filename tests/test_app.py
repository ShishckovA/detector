import asyncio
from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import (
    CropBox,
    FaceCandidate,
    InferenceService,
    Settings,
    app,
    choose_best_face,
    crop_box_for_face,
    detect_faces,
    make_preprocess,
)


class NoFaceDetector:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        return 1, None


class OneFaceDetector:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        faces = torch.zeros((1, 15), dtype=torch.float32).numpy()
        faces[0, 0:4] = [40, 40, 80, 80]
        faces[0, 14] = 0.9
        return 1, faces


class SmallFaceDetector:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        faces = torch.zeros((1, 15), dtype=torch.float32).numpy()
        faces[0, 0:4] = [20, 20, 20, 20]
        faces[0, 14] = 0.92
        return 1, faces


class ScaledFaceDetector:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        faces = torch.zeros((1, 15), dtype=torch.float32).numpy()
        faces[0, 0:4] = [10, 20, 30, 40]
        faces[0, 14] = 0.8
        return 1, faces


class ConstantClassifier:
    def __init__(self, logit):
        self.logit = logit

    def __call__(self, tensor):
        self.last_shape = tuple(tensor.shape)
        return torch.tensor([[self.logit]], dtype=torch.float32)


class MultiClassClassifier:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, tensor):
        self.last_shape = tuple(tensor.shape)
        return torch.tensor([self.logits], dtype=torch.float32)


def make_png_bytes(size=(180, 160), color=(128, 96, 64)):
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_service(detector, classifier=None):
    settings = Settings(
        classifier_model_path=Path("classifier.pt"),
        detector_model_path=Path("detector.onnx"),
        classifier_threshold=0.65,
        min_face_size=48,
    )
    classifier = classifier or ConstantClassifier(logit=1.0)
    return InferenceService(settings, detector, classifier, make_preprocess()), classifier


def test_choose_best_face_prefers_score_then_area():
    faces = [
        FaceCandidate(0, 0, 120, 120, 0.8),
        FaceCandidate(0, 0, 40, 40, 0.9),
        FaceCandidate(0, 0, 60, 80, 0.9),
    ]

    best = choose_best_face(faces)

    assert best == faces[2]


def test_crop_box_matches_margin_and_clamps_to_image():
    face = FaceCandidate(bbox_x=5, bbox_y=10, bbox_w=100, bbox_h=80, detector_score=0.9)

    box = crop_box_for_face(face, image_width=120, image_height=100, margin=0.25)

    assert box == CropBox(x1=0, y1=0, x2=120, y2=100)


def test_preprocess_outputs_classifier_shape():
    preprocess = make_preprocess()
    tensor = preprocess(Image.new("RGB", (96, 128), color=(20, 40, 80))).unsqueeze(0)

    assert tuple(tensor.shape) == (1, 3, 224, 224)


def test_preprocess_is_deterministic_for_fixed_crop():
    preprocess = make_preprocess()
    image = Image.new("RGB", (96, 128), color=(20, 40, 80))

    first = preprocess(image)
    second = preprocess(image)

    assert torch.equal(first, second)


def test_detect_faces_resizes_large_image_and_scales_bbox_back():
    detector = ScaledFaceDetector()
    image = torch.zeros((1000, 2000, 3), dtype=torch.uint8).numpy()

    faces = detect_faces(detector, image, max_side=1000)

    assert detector.size == (1000, 500)
    assert len(faces) == 1
    assert faces[0].bbox_x == 20
    assert faces[0].bbox_y == 40
    assert faces[0].bbox_w == 60
    assert faces[0].bbox_h == 80
    assert abs(faces[0].detector_score - 0.8) < 1e-6


def test_service_returns_no_face_response():
    service, _ = make_service(NoFaceDetector())

    result = service.predict(Image.open(BytesIO(make_png_bytes())))

    assert result["face_found"] is False
    assert result["reason"] == "no_face"
    assert result["score"] is None
    assert result["threshold"] == 0.65


def test_service_classifies_one_face_and_returns_score():
    service, classifier = make_service(OneFaceDetector())

    result = service.predict(Image.open(BytesIO(make_png_bytes())))

    assert result["face_found"] is True
    assert result["label"] == "positive"
    assert result["threshold"] == 0.65
    assert result["score"] > 0.65
    assert classifier.last_shape == (1, 3, 224, 224)
    assert result["bbox"]["width"] >= 48
    assert result["bbox"]["height"] >= 48


def test_service_classifies_multiclass_output():
    classifier = MultiClassClassifier([-1.0, 0.5, 4.0, 1.0])
    service, _ = make_service(OneFaceDetector(), classifier)

    result = service.predict(Image.open(BytesIO(make_png_bytes())))

    assert result["face_found"] is True
    assert result["label"] == "alex"
    assert result["score"] == result["scores"]["alex"]
    assert result["logit"] == result["logits"]["alex"]
    assert set(result["scores"]) == {"negative", "positive", "alex", "artem"}
    assert classifier.last_shape == (1, 3, 224, 224)


def test_service_rejects_small_face_without_classification():
    service, classifier = make_service(SmallFaceDetector())

    result = service.predict(Image.open(BytesIO(make_png_bytes())))

    assert result["face_found"] is False
    assert result["reason"] == "face_too_small"
    assert not hasattr(classifier, "last_shape")


def test_endpoint_handles_broken_upload_with_injected_service():
    service, _ = make_service(NoFaceDetector())
    app.state.settings = service.settings
    app.state.service = service
    app.state.inference_semaphore = asyncio.Semaphore(2)

    client = TestClient(app)
    response = client.post(
        "/api/face-score",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a readable image"


def test_endpoint_returns_score_payload_with_injected_service():
    service, _ = make_service(OneFaceDetector())
    app.state.settings = service.settings
    app.state.service = service
    app.state.inference_semaphore = asyncio.Semaphore(2)

    client = TestClient(app)
    response = client.post(
        "/api/face-score",
        files={"file": ("face.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["face_found"] is True
    assert data["score"] > 0.65
    assert data["threshold"] == 0.65
    assert data["bbox"]["width"] >= 48
