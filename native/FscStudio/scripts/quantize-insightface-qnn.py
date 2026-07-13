#!/usr/bin/env python3
"""Create QNN HTP QDQ InsightFace models from representative local images.

This is a build-time tool. FSC Studio never imports Python at runtime, and the
calibration images are not copied into the model output directory.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MODEL_NAMES = ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx")


class TensorDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, tensors: list[np.ndarray]) -> None:
        self._input_name = input_name
        self._tensors = tensors
        self._iterator: Iterable[dict[str, np.ndarray]] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = iter({self._input_name: value} for value in self._tensors)
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = None


def buffalo_root(path: Path) -> Path:
    return path if path.name == "buffalo_l" else path / "buffalo_l"


def database_images(database_path: Path) -> list[Path]:
    candidates: list[Path] = []
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT source_path FROM faces WHERE source_path IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    for (source_path,) in rows:
        path = Path(source_path)
        if not path.is_absolute():
            path = database_path.parent / path
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(path)
    return candidates


def calibration_images(path: Path, maximum: int) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".fscdb":
        candidates = database_images(path)
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        candidates = [path]
    elif path.is_dir():
        candidates = [
            candidate
            for candidate in sorted(path.rglob("*"))
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        ]
    else:
        raise FileNotFoundError(f"Calibration source does not exist: {path}")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= maximum:
            break
    if not unique:
        raise RuntimeError("No readable calibration images were found.")
    return unique


def read_bgr(path: Path) -> np.ndarray | None:
    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def detector_tensor(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(640.0 / width, 640.0 / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None])


def recognition_tensor(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    aligned = norm_crop(image, keypoints, image_size=112)
    rgb = aligned[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None])


def landmark_tensor(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    scale = 192.0 / (max(width, height) * 1.5)
    transform = np.array(
        [[scale, 0.0, 96.0 - center_x * scale], [0.0, scale, 96.0 - center_y * scale]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        image,
        transform,
        (192, 192),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rgb = crop[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def collect_tensors(model_root: Path, paths: list[Path]) -> dict[str, list[np.ndarray]]:
    detector = get_model(str(model_root / "det_10g.onnx"), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640))
    tensors: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    for index, path in enumerate(paths, start=1):
        image = read_bgr(path)
        if image is None:
            print(f"[{index}/{len(paths)}] skipped unreadable image: {path}")
            continue
        tensors["det_10g.onnx"].append(detector_tensor(image))
        boxes, keypoint_sets = detector.detect(image, max_num=1, metric="default")
        if boxes.shape[0] == 0 or keypoint_sets is None:
            print(f"[{index}/{len(paths)}] no face: {path}")
            continue
        tensors["w600k_r50.onnx"].append(recognition_tensor(image, keypoint_sets[0]))
        crop = landmark_tensor(image, boxes[0])
        tensors["2d106det.onnx"].append(crop)
        tensors["1k3d68.onnx"].append(crop)
        print(f"[{index}/{len(paths)}] calibrated: {path.name}")

    if not tensors["det_10g.onnx"]:
        raise RuntimeError("No detector calibration tensors were produced.")
    for name in MODEL_NAMES[1:]:
        if not tensors[name]:
            raise RuntimeError(f"No face calibration tensors were produced for {name}.")
    return tensors


def quantize_model(source: Path, destination: Path, tensors: list[np.ndarray], overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"Keeping existing model: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = destination.with_suffix(".preprocessed.onnx")
    if preprocessed.exists():
        preprocessed.unlink()
    changed = qnn_preprocess_model(str(source), str(preprocessed))
    quantization_source = preprocessed if changed else source
    input_name = ort.InferenceSession(str(quantization_source), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    reader = TensorDataReader(input_name, tensors)
    config = get_qnn_qdq_config(
        str(quantization_source),
        reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
    )
    quantize(str(quantization_source), str(destination), config)
    if preprocessed.exists():
        preprocessed.unlink()
    print(f"Wrote {destination} ({destination.stat().st_size / (1024 * 1024):.1f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="InsightFace models root or buffalo_l directory")
    parser.add_argument("--calibration", type=Path, required=True, help="Image, image directory, or local .fscdb")
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = buffalo_root(args.models.resolve())
    missing = [model_root / name for name in MODEL_NAMES if not (model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing InsightFace model: {missing[0]}")
    paths = calibration_images(args.calibration.resolve(), max(1, args.max_images))
    print(f"Using {len(paths)} local calibration image(s).")
    tensors = collect_tensors(model_root, paths)
    output_root = model_root / "qnn_htp"
    for name in MODEL_NAMES:
        quantize_model(model_root / name, output_root / name, tensors[name], args.overwrite)
    print("QNN HTP model set is ready. Calibration images were not copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
