from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PYTHON_DIR = APP_DIR / "python"
SITE_PACKAGES = PYTHON_DIR / "Lib" / "site-packages"
MODE_FILE = APP_DIR / "runtime_mode.txt"


def _read_runtime_mode() -> str:
    env_mode = os.environ.get("FSC_RUNTIME_MODE", "").strip().lower()
    if env_mode:
        return env_mode
    try:
        return MODE_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "auto"


def _add_dll_directory(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory") or not path.exists():
        return
    try:
        os.add_dll_directory(str(path))
    except OSError:
        pass


def _prepare_runtime() -> None:
    app_path = str(APP_DIR)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)

    mode = _read_runtime_mode()
    if mode:
        os.environ["FSC_RUNTIME_MODE"] = mode
    if mode == "cpu":
        os.environ["FSC_FORCE_CPU"] = "1"

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("INSIGHTFACE_HOME", str(APP_DIR / "model" / "insightface"))

    for relative in [
        Path("PySide6"),
        Path("cv2"),
        Path("onnxruntime") / "capi",
        Path("nvidia"),
    ]:
        path = SITE_PACKAGES / relative
        _add_dll_directory(path)
        if relative == Path("nvidia") and path.exists():
            for child in path.rglob("*"):
                if child.is_dir() and child.name.lower() in {"bin", "x86_64"}:
                    _add_dll_directory(child)


def main() -> int:
    _prepare_runtime()
    target = APP_DIR / "fsc_studio.py"
    if not target.exists():
        print(f"Missing application entry point: {target}", file=sys.stderr)
        return 2
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
