const API_BASE =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : "";

const fileInput = document.querySelector("#fileInput");
const pickButton = document.querySelector("#pickButton");
const submitButton = document.querySelector("#submitButton");
const resetButton = document.querySelector("#resetButton");
const dropZone = document.querySelector("#dropZone");
const emptyState = document.querySelector("#emptyState");
const previewCanvas = document.querySelector("#previewCanvas");
const fileMeta = document.querySelector("#fileMeta");
const resultEmpty = document.querySelector("#resultEmpty");
const resultContent = document.querySelector("#resultContent");
const messageBox = document.querySelector("#messageBox");
const labelText = document.querySelector("#labelText");
const scoreRing = document.querySelector("#scoreRing");
const scoreValue = document.querySelector("#scoreValue");
const thresholdValue = document.querySelector("#thresholdValue");
const detectorValue = document.querySelector("#detectorValue");
const totalTiming = document.querySelector("#totalTiming");

let selectedFile = null;
let currentImage = null;
let currentImageUrl = null;
let latestResult = null;

const DISPLAY_LABELS = {
  positive: "Открытый",
  negative: "Спокойный",
  alex: "Алекс",
  artem: "Артём",
};

const RESULT_MESSAGES = {
  positive: "Фото производит открытое и контактное впечатление.",
  negative: "Фото выглядит спокойнее. Можно попробовать другой ракурс или выражение.",
  alex: "Фото больше всего похоже на класс Алекс.",
  artem: "Фото больше всего похоже на класс Артём.",
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatNumber(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatMs(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value < 1000 ? "мгновенно" : `${(value / 1000).toFixed(1)} с`;
}

function formatPercent(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "-";
}

function formatPhotoQuality(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 0.9) return "отличное";
  if (value >= 0.75) return "хорошее";
  return "среднее";
}

function getImpressionLabel(payload) {
  if (!payload.face_found) {
    return payload.reason === "face_too_small" ? "Нужно ближе" : "Не получилось";
  }
  return DISPLAY_LABELS[payload.label] || payload.label || "-";
}

function getUploadErrorMessage(message) {
  if (message === "Uploaded file is empty") return "Файл пустой. Выберите другое фото.";
  if (message === "Uploaded file is not a readable image") {
    return "Не получилось прочитать изображение. Попробуйте JPG, PNG или WEBP.";
  }
  return message;
}

function setMessage(text, type = "error") {
  if (!text) {
    messageBox.hidden = true;
    messageBox.textContent = "";
    messageBox.classList.remove("is-ok");
    return;
  }

  messageBox.hidden = false;
  messageBox.textContent = text;
  messageBox.classList.toggle("is-ok", type === "ok");
}

function resetResult() {
  latestResult = null;
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  setMessage("");
}

function resetAll() {
  selectedFile = null;
  latestResult = null;
  currentImage = null;

  if (currentImageUrl) {
    URL.revokeObjectURL(currentImageUrl);
    currentImageUrl = null;
  }

  fileInput.value = "";
  fileMeta.textContent = "Файл не выбран";
  submitButton.disabled = true;
  resetButton.disabled = true;
  previewCanvas.hidden = true;
  emptyState.hidden = false;
  resetResult();
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
    }

    const image = new Image();
    currentImageUrl = URL.createObjectURL(file);
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Файл не похож на читаемое изображение"));
    image.src = currentImageUrl;
  });
}

function drawPreview(result = latestResult) {
  if (!currentImage || previewCanvas.hidden) return;

  const rect = dropZone.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(260, Math.floor(rect.height * dpr));
  previewCanvas.width = width;
  previewCanvas.height = height;

  const ctx = previewCanvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#edf1ea";
  ctx.fillRect(0, 0, width, height);

  const scale = Math.min(width / currentImage.naturalWidth, height / currentImage.naturalHeight);
  const drawWidth = currentImage.naturalWidth * scale;
  const drawHeight = currentImage.naturalHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;

  ctx.drawImage(currentImage, offsetX, offsetY, drawWidth, drawHeight);

  if (result?.bbox) {
    const { x1, y1, width: boxWidth, height: boxHeight } = result.bbox;
    const left = offsetX + x1 * scale;
    const top = offsetY + y1 * scale;
    const scaledWidth = boxWidth * scale;
    const scaledHeight = boxHeight * scale;

    ctx.save();
    ctx.lineWidth = Math.max(3, 4 * dpr);
    ctx.strokeStyle = result.face_found ? "#1f9d66" : "#ef704b";
    ctx.fillStyle = "rgba(21, 24, 23, 0.16)";
    ctx.fillRect(left, top, scaledWidth, scaledHeight);
    ctx.strokeRect(left, top, scaledWidth, scaledHeight);
    ctx.restore();
  }
}

async function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setMessage("Нужен файл изображения: JPG, PNG или WEBP.");
    return;
  }
  if (file.size === 0) {
    setMessage("Файл пустой. Выберите другое фото.");
    return;
  }

  try {
    currentImage = await loadImage(file);
    selectedFile = file;
    latestResult = null;
    fileMeta.textContent = `${file.name} · ${formatBytes(file.size)} · ${currentImage.naturalWidth}x${currentImage.naturalHeight}`;
    submitButton.disabled = false;
    resetButton.disabled = false;
    previewCanvas.hidden = false;
    emptyState.hidden = true;
    resetResult();
    drawPreview();
  } catch (error) {
    resetAll();
    setMessage(error.message);
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function requestScore(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/face-score`, {
    method: "POST",
    body: formData,
  });

  const payload = await readJson(response);
  if (!response.ok) {
    const message = payload?.detail || `HTTP ${response.status}`;
    const uploadError = response.status === 400 && payload?.detail;
    if (uploadError) {
      const error = new Error(message);
      error.uploadError = true;
      throw error;
    }
    throw new Error(message);
  }

  return payload;
}

function renderResult(payload) {
  latestResult = payload;
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  const score = typeof payload.score === "number" ? payload.score : 0;
  const label = getImpressionLabel(payload);
  labelText.textContent = label;
  labelText.classList.toggle("is-negative", payload.label === "negative" || !payload.face_found);
  scoreRing.style.setProperty("--score", String(Math.max(0, Math.min(1, score))));
  scoreValue.textContent = payload.score === null ? "-" : formatPercent(score);
  thresholdValue.textContent = payload.face_found ? "Фото принято" : "Нужно другое фото";
  detectorValue.textContent = formatPhotoQuality(payload.detector_score);
  totalTiming.textContent = formatMs(payload.timings_ms?.total);

  if (payload.face_found) {
    setMessage(RESULT_MESSAGES[payload.label] || "Оценка готова.", "ok");
  } else if (payload.reason === "face_too_small") {
    setMessage("Лицо на фото слишком маленькое. Попробуйте портрет крупнее.");
  } else {
    setMessage("Не удалось уверенно найти лицо. Выберите более четкий портрет.");
  }

  drawPreview(payload);
}

async function submitPhoto() {
  if (!selectedFile) return;

  submitButton.disabled = true;
  submitButton.textContent = "Проверка";
  setMessage("");

  try {
    const payload = await requestScore(selectedFile);
    renderResult(payload);
  } catch (error) {
    if (error.uploadError) {
      setMessage(getUploadErrorMessage(error.message));
      return;
    }

    setMessage("Сервис временно недоступен. Попробуйте еще раз позже.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Узнать оценку";
  }
}

fileInput.addEventListener("change", (event) => {
  setFile(event.target.files?.[0]);
});

pickButton.addEventListener("click", () => {
  fileInput.click();
});

submitButton.addEventListener("click", submitPhoto);
resetButton.addEventListener("click", resetAll);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files?.[0]);
});

window.addEventListener("resize", () => {
  drawPreview();
});
