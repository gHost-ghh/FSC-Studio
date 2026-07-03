from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


CUDA_DLL_PATTERNS = ("cudart64*.dll", "cublas64*.dll", "cudnn64*.dll")


def _path_dirs() -> list[Path]:
    dirs: list[Path] = []
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw.strip():
            dirs.append(Path(raw.strip()))
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        dirs.append(Path(cuda_path) / "bin")
    return dirs


def detect_cuda() -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_name = ""
    driver = ""
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            ).strip()
            if output:
                first = output.splitlines()[0]
                parts = [part.strip() for part in first.split(",", 1)]
                gpu_name = parts[0]
                driver = parts[1] if len(parts) > 1 else ""
        except Exception:
            pass

    dll_hits: list[str] = []
    for folder in _path_dirs():
        if not folder.exists():
            continue
        for pattern in CUDA_DLL_PATTERNS:
            if any(folder.glob(pattern)):
                dll_hits.append(f"{folder}\\{pattern}")

    return {
        "nvidia_smi": bool(nvidia_smi),
        "gpu_name": gpu_name,
        "driver": driver,
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "cuda_dll_signals": dll_hits,
        "recommended_mode": "gpu" if nvidia_smi and dll_hits else "cpu",
    }


if __name__ == "__main__":
    print(json.dumps(detect_cuda(), indent=2))
