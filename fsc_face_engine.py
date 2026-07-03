from __future__ import annotations

import io
import os
import site
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_DIR / "model" / "insightface"
MODEL_NAME = "buffalo_l"
FORCE_CPU_ENV = "FSC_FORCE_CPU"
RUNTIME_MODE_ENV = "FSC_RUNTIME_MODE"
DEFAULT_CANVAS_IMAGE_SIZE = 396


class FaceEngineError(RuntimeError):
    pass


@dataclass
class EngineInfo:
    providers: list[str]
    requested_gpu: bool
    using_gpu: bool
    model_root: str
    notes: list[str]

    @property
    def status_text(self) -> str:
        provider_text = "GPU" if self.using_gpu else "CPU"
        if self.notes:
            return f"InsightFace {MODEL_NAME} ready on {provider_text}. " + " ".join(self.notes)
        return f"InsightFace {MODEL_NAME} ready on {provider_text}."


@dataclass
class AnalyzedFace:
    image_bgr: np.ndarray
    embedding: np.ndarray
    bbox: list[float]
    kps: list[list[float]]
    landmarks: list[list[float]] | None
    landmarks3d: list[list[float]] | None
    det_score: float
    quality_score: float
    quality_details: dict[str, float]
    source_path: str = ""
    file_name: str = ""


class FaceEngine:
    def __init__(self, prefer_gpu: bool = True) -> None:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        prefer_gpu = should_prefer_gpu(prefer_gpu)
        self.info = self._load_app(prefer_gpu)

    def _load_app(self, prefer_gpu: bool) -> EngineInfo:
        notes: list[str] = []
        notes.extend(_add_nvidia_dll_directories())
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - import failure is environment-specific
            raise FaceEngineError(
                "Failed to import onnxruntime. Install the project with the "
                "cpu or gpu extra and start FSC Studio with the same Python environment. "
                f"Original error: {exc}"
            ) from exc

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
            except Exception as exc:
                notes.append(f"CUDA DLL preload warning: {exc}")

        available = list(ort.get_available_providers())
        requested_providers = ["CPUExecutionProvider"]
        if prefer_gpu and "CUDAExecutionProvider" in available:
            requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif prefer_gpu:
            notes.append("CUDAExecutionProvider is not available; using CPU.")

        try:
            return self._create_app(requested_providers, prefer_gpu, notes)
        except Exception as gpu_exc:
            if requested_providers[0] != "CUDAExecutionProvider":
                raise
            notes.append(f"CUDA initialization failed; using CPU. {gpu_exc}")
            return self._create_app(["CPUExecutionProvider"], prefer_gpu, notes)

    def _create_app(
        self,
        providers: list[str],
        prefer_gpu: bool,
        notes: list[str],
    ) -> EngineInfo:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            root=str(MODEL_ROOT),
            providers=providers,
        )
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(ctx_id=ctx_id)
        actual_providers = _actual_session_providers(self.app)
        if providers[0] == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in actual_providers:
            notes.append("CUDAExecutionProvider was requested but sessions are using CPU.")
        provider_list = actual_providers or providers
        return EngineInfo(
            providers=provider_list,
            requested_gpu=prefer_gpu,
            using_gpu="CUDAExecutionProvider" in provider_list,
            model_root=str(MODEL_ROOT),
            notes=notes,
        )

    def extract_faces_from_bgr(
        self,
        image_bgr: np.ndarray,
        source_path: str = "",
    ) -> list[AnalyzedFace]:
        faces = self.app.get(image_bgr)
        analyzed = [
            self._face_to_analysis(face, image_bgr, source_path)
            for face in faces
        ]
        analyzed.sort(key=lambda item: item.det_score, reverse=True)
        return analyzed

    def extract_faces_from_path(self, image_path: str | os.PathLike[str]) -> list[AnalyzedFace]:
        image_bgr = read_image_bgr(image_path)
        return self.extract_faces_from_bgr(image_bgr, str(image_path))

    def extract_single_face_from_path(self, image_path: str | os.PathLike[str]) -> AnalyzedFace:
        faces = self.extract_faces_from_path(image_path)
        if len(faces) != 1:
            raise FaceEngineError(f"Exactly one face is required; detected {len(faces)}.")
        return faces[0]

    def _face_to_analysis(
        self,
        face: object,
        image_bgr: np.ndarray,
        source_path: str,
    ) -> AnalyzedFace:
        embedding = _as_float_array(_face_value(face, "normed_embedding"))
        if embedding is None:
            embedding = _as_float_array(_face_value(face, "embedding"))
            if embedding is None:
                raise FaceEngineError("InsightFace did not return a face embedding.")
            norm = float(np.linalg.norm(embedding))
            if norm:
                embedding = embedding / norm

        bbox = _as_float_array(_face_value(face, "bbox"))
        kps = _as_float_array(_face_value(face, "kps"))
        landmarks = _as_float_array(_face_value(face, "landmark_2d_106"))
        landmarks3d = _as_float_array(_face_value(face, "landmark_3d_68"))
        det_score = _face_value(face, "det_score", 0.0)
        quality_score, quality_details = face_quality_score(image_bgr, bbox, float(det_score))

        path_obj = Path(source_path) if source_path else None
        return AnalyzedFace(
            image_bgr=image_bgr,
            embedding=np.asarray(embedding, dtype=np.float32).reshape(-1),
            bbox=_array_to_nested_list(bbox),
            kps=_array_to_nested_list(kps),
            landmarks=_array_to_nested_list(landmarks) if landmarks is not None else None,
            landmarks3d=_array_to_nested_list(landmarks3d) if landmarks3d is not None else None,
            det_score=float(det_score),
            quality_score=quality_score,
            quality_details=quality_details,
            source_path=source_path,
            file_name=path_obj.name if path_obj else "",
        )


_ENGINE: FaceEngine | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []


def get_engine(prefer_gpu: bool = True, force_reload: bool = False) -> FaceEngine:
    global _ENGINE
    if force_reload or _ENGINE is None:
        _ENGINE = FaceEngine(prefer_gpu=prefer_gpu)
    return _ENGINE


def should_prefer_gpu(prefer_gpu: bool = True) -> bool:
    mode = os.environ.get(RUNTIME_MODE_ENV, "").strip().lower()
    force_cpu = os.environ.get(FORCE_CPU_ENV, "").strip().lower()
    if force_cpu in {"1", "true", "yes", "on"}:
        return False
    if mode in {"cpu", "force-cpu"}:
        return False
    if mode in {"gpu", "cuda", "auto-gpu"}:
        return True
    return bool(prefer_gpu)


def _add_nvidia_dll_directories() -> list[str]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    notes: list[str] = []
    candidate_roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(Path(user_site))

    seen: set[Path] = set()
    for root in candidate_roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for dll_dir in nvidia_root.rglob("*"):
            if not dll_dir.is_dir() or dll_dir in seen:
                continue
            if not any(dll_dir.glob("*.dll")):
                continue
            seen.add(dll_dir)
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            except OSError as exc:
                notes.append(f"Failed to add DLL directory {dll_dir}: {exc}")
    return notes


def _actual_session_providers(app: object) -> list[str]:
    providers: list[str] = []
    models = getattr(app, "models", {})
    iterable = models.values() if isinstance(models, dict) else []
    for model in iterable:
        session = getattr(model, "session", None)
        if session is None:
            session = getattr(model, "sess", None)
        get_providers = getattr(session, "get_providers", None)
        if not callable(get_providers):
            continue
        for provider in get_providers():
            if provider not in providers:
                providers.append(provider)
    return providers


def read_image_bgr(image_path: str | os.PathLike[str]) -> np.ndarray:
    path = str(image_path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise FaceEngineError(f"Failed to read image: {exc}") from exc
    if data.size == 0:
        raise FaceEngineError("Image file is empty.")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceEngineError("Failed to decode image.")
    return image


def render_plain_preview(
    image_bgr: np.ndarray,
    max_size: int = DEFAULT_CANVAS_IMAGE_SIZE,
) -> Image.Image:
    return _resize_pil(_bgr_to_pil(image_bgr), max_size)


def render_face_overlay(
    analysis: AnalyzedFace,
    max_size: int = DEFAULT_CANVAS_IMAGE_SIZE,
) -> Image.Image:
    return render_faces_overlay([analysis], selected_index=0, max_size=max_size)


def render_faces_overlay(
    faces: list[AnalyzedFace],
    selected_index: int = 0,
    max_size: int = DEFAULT_CANVAS_IMAGE_SIZE,
) -> Image.Image:
    if not faces:
        raise FaceEngineError("No face was detected in the selected image.")
    selected_index = max(0, min(int(selected_index), len(faces) - 1))
    return _resize_pil(_draw_faces_overlay(faces, selected_index), max_size)


def render_focused_faces_overlay(
    faces: list[AnalyzedFace],
    selected_index: int = 0,
    max_size: int = DEFAULT_CANVAS_IMAGE_SIZE,
    margin: float = 0.75,
) -> Image.Image:
    if not faces:
        raise FaceEngineError("No face was detected in the selected image.")
    selected_index = max(0, min(int(selected_index), len(faces) - 1))
    analysis = faces[selected_index]
    pil_image = _bgr_to_pil(faces[0].image_bgr)
    crop_box = _expanded_crop_box(_safe_bbox(analysis.bbox), pil_image.size, margin)
    return _render_fixed_marker_focus(
        pil_image,
        crop_box,
        analysis.bbox,
        analysis.landmarks,
        analysis.kps,
        max_size,
    )


def preview_png_from_analysis(analysis: AnalyzedFace) -> bytes:
    image = render_face_overlay(analysis)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preview_png_to_pil(preview_png: bytes) -> Image.Image:
    with Image.open(io.BytesIO(preview_png)) as image:
        return image.copy()


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
    b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise FaceEngineError(f"Embedding dimensions differ: {a.size} vs {b.size}.")
    return float(np.dot(a, b))


def similarity_percent(cosine: float) -> float:
    return max(0.0, min(1.0, cosine)) * 100.0


def face_quality_score(
    image_bgr: np.ndarray,
    bbox: np.ndarray | Iterable[float] | None,
    det_score: float,
) -> tuple[float, dict[str, float]]:
    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clamped_bbox(bbox, image_width, image_height)
    face_width = max(0, x2 - x1)
    face_height = max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    face_area_ratio = (face_width * face_height) / image_area

    if face_width <= 0 or face_height <= 0:
        return 0.0, {
            "det_score": float(det_score),
            "area_ratio": 0.0,
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
        }

    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    det_component = _clamp01((float(det_score) - 0.45) / 0.5)
    area_component = _clamp01(face_area_ratio / 0.08)
    sharpness_component = _clamp01(sharpness / 160.0)
    brightness_component = _clamp01(1.0 - abs(brightness - 128.0) / 128.0)
    contrast_component = _clamp01(contrast / 64.0)

    score = (
        0.35 * det_component
        + 0.20 * area_component
        + 0.25 * sharpness_component
        + 0.10 * brightness_component
        + 0.10 * contrast_component
    )
    details = {
        "det_score": float(det_score),
        "area_ratio": float(face_area_ratio),
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "det_component": det_component,
        "area_component": area_component,
        "sharpness_component": sharpness_component,
        "brightness_component": brightness_component,
        "contrast_component": contrast_component,
    }
    return float(round(_clamp01(score), 6)), details


def analysis_to_database_record(analysis: AnalyzedFace) -> dict[str, object]:
    return {
        "file_name": analysis.file_name,
        "source_path": analysis.source_path,
        "embedding": analysis.embedding,
        "bbox": analysis.bbox,
        "kps": analysis.kps,
        "landmarks": analysis.landmarks,
        "landmarks3d": analysis.landmarks3d,
        "det_score": analysis.det_score,
        "quality_score": analysis.quality_score,
        "quality": analysis.quality_details,
        "preview_png": preview_png_from_analysis(analysis),
    }


def _face_value(face: object, key: str, default: object = None) -> object:
    value = getattr(face, key, None)
    if value is None and hasattr(face, "get"):
        try:
            value = face.get(key, default)  # type: ignore[attr-defined]
        except Exception:
            value = default
    return default if value is None else value


def _as_float_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _array_to_nested_list(value: np.ndarray | None) -> list:
    if value is None:
        return []
    return np.asarray(value, dtype=np.float32).tolist()


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def _resize_pil(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image.copy()
    scale = max_size / max(width, height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _draw_faces_overlay(faces: list[AnalyzedFace], selected_index: int) -> Image.Image:
    pil_image = _bgr_to_pil(faces[0].image_bgr)
    draw = ImageDraw.Draw(pil_image)

    for index, face in enumerate(faces):
        x1, y1, x2, y2 = _safe_bbox(face.bbox)
        selected = index == selected_index
        color = (0, 255, 0) if selected else (255, 180, 0)
        width = 4 if selected else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = str(index + 1)
        draw.rectangle((x1, max(0, y1 - 20), x1 + 24, y1), fill=color)
        draw.text((x1 + 7, max(0, y1 - 18)), label, fill=(0, 0, 0))

    analysis = faces[selected_index]
    if analysis.landmarks:
        radius = max(1, int(round(max(pil_image.size) / 800)))
        for x, y in _points(analysis.landmarks):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 180, 255))

    keypoint_colors = [
        (0, 128, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for index, (x, y) in enumerate(_points(analysis.kps)):
        color = keypoint_colors[index % len(keypoint_colors)]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    return pil_image


def _render_fixed_marker_focus(
    pil_image: Image.Image,
    crop_box: tuple[int, int, int, int],
    bbox: Iterable[float],
    landmarks: Iterable[Iterable[float]] | None,
    kps: Iterable[Iterable[float]],
    max_size: int,
) -> Image.Image:
    left, top, right, bottom = crop_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    focused = _resize_pil(pil_image.crop(crop_box), max_size)
    scale_x = focused.width / crop_width
    scale_y = focused.height / crop_height
    draw = ImageDraw.Draw(focused)

    def map_point(x: float, y: float) -> tuple[float, float]:
        return (x - left) * scale_x, (y - top) * scale_y

    x1, y1, x2, y2 = _safe_bbox(bbox)
    rx1, ry1 = map_point(x1, y1)
    rx2, ry2 = map_point(x2, y2)
    draw.rectangle((rx1, ry1, rx2, ry2), outline=(0, 255, 0), width=2)

    if landmarks:
        for x, y in _points(landmarks):
            if left <= x <= right and top <= y <= bottom:
                px, py = map_point(x, y)
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(0, 180, 255))

    keypoint_colors = [
        (0, 128, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for index, (x, y) in enumerate(_points(kps)):
        if left <= x <= right and top <= y <= bottom:
            px, py = map_point(x, y)
            color = keypoint_colors[index % len(keypoint_colors)]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    return focused


def _expanded_crop_box(
    bbox: Iterable[float],
    image_size: tuple[int, int],
    margin: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = _safe_bbox(bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    expand_x = width * max(0.0, float(margin))
    expand_y = height * max(0.0, float(margin))
    left = max(0, int(round(x1 - expand_x)))
    top = max(0, int(round(y1 - expand_y)))
    right = min(image_width, int(round(x2 + expand_x)))
    bottom = min(image_height, int(round(y2 + expand_y)))
    if right <= left or bottom <= top:
        return 0, 0, image_width, image_height
    return left, top, right, bottom


def _safe_bbox(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    values = list(bbox)
    if len(values) < 4:
        return 0.0, 0.0, 0.0, 0.0
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _clamped_bbox(
    bbox: np.ndarray | Iterable[float] | None,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, 0, 0
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return 0, 0, 0, 0
    x1 = int(max(0, min(image_width - 1, round(float(values[0])))))
    y1 = int(max(0, min(image_height - 1, round(float(values[1])))))
    x2 = int(max(0, min(image_width, round(float(values[2])))))
    y2 = int(max(0, min(image_height, round(float(values[3])))))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _points(points: Iterable[Iterable[float]]) -> Iterable[tuple[float, float]]:
    for point in points:
        values = list(point)
        if len(values) >= 2:
            yield float(values[0]), float(values[1])
