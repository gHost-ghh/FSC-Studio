#!/usr/bin/env python3
"""Convert the official MediaPipe face landmark model to a fixed-shape ONNX asset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import onnx


DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsc-face-mesh-") as temporary:
        temporary_root = Path(temporary)
        task_path = args.task.resolve() if args.task else temporary_root / "face_landmarker.task"
        if args.task:
            if not task_path.is_file():
                raise FileNotFoundError(task_path)
        else:
            print(f"Downloading {args.url}")
            urllib.request.urlretrieve(args.url, task_path)

        with zipfile.ZipFile(task_path) as archive:
            member = next(
                (name for name in archive.namelist() if name.endswith("face_landmarks_detector.tflite")),
                None,
            )
            if member is None:
                raise RuntimeError("The MediaPipe task does not contain face_landmarks_detector.tflite")
            tflite_path = temporary_root / "face_landmarks_detector.tflite"
            tflite_path.write_bytes(archive.read(member))

        converted_path = temporary_root / "face_landmarks_detector.onnx"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--tflite",
                str(tflite_path),
                "--output",
                str(converted_path),
                "--opset",
                "17",
            ],
            check=True,
        )
        model = onnx.load(converted_path)
        input_shape = model.graph.input[0].type.tensor_type.shape
        first_dimension = input_shape.dim[0]
        first_dimension.ClearField("dim_param")
        first_dimension.dim_value = 1
        output_names = {output.name for output in model.graph.output}
        if model.graph.input[0].name != "input_12" or not {"Identity", "Identity_1"}.issubset(output_names):
            raise RuntimeError("Converted model does not expose the expected MediaPipe tensors")
        onnx.checker.check_model(model)
        onnx.save(model, args.output)

    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
