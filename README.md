# FSC Studio

FSC Studio is a desktop face-similarity and face-database application built with PySide6, InsightFace, and ONNX Runtime.

The public repository is intentionally source-only. It does not include private face databases, face photos, model weights, generated installers, virtual environments, or mobile-port prototypes.

## Current Scope

- Multi-face image import into SQLite `.fscdb` databases.
- Face similarity search using normalized InsightFace embeddings.
- One-to-one face comparison.
- Live webcam search against the current database.
- Library browsing, people assignment, tags, review state, ignored records, notes, duplicate detection, and CSV export.
- Similarity clustering for likely same-person groups.
- Runtime/database maintenance tools such as integrity check, backup, checkpoint, and vacuum.

## Repository Layout

```text
fsc_studio.py           Main PySide6 desktop UI
fsc_studio_services.py  Application service layer
fsc_face_engine.py      InsightFace / ONNX Runtime integration
fsc_face_database.py    SQLite .fscdb database layer
pyproject.toml          Python dependency metadata
```

## Setup

Use Python 3.12 on Windows.

CPU runtime:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[cpu]"
```

GPU runtime with CUDA-capable ONNX Runtime:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[gpu]"
```

## Model Files

FSC Studio uses InsightFace `buffalo_l`. Model weights are not stored in Git.

Place the model files locally under:

```text
model/insightface/models/buffalo_l/
```

Expected files:

```text
1k3d68.onnx
2d106det.onnx
det_10g.onnx
genderage.onnx
w600k_r50.onnx
```

## Run

```powershell
.\.venv312\Scripts\python.exe fsc_studio.py
```

## Data Privacy

Face databases and photos can contain sensitive biometric data. Keep `.fscdb`, `.dtb`, image folders, and generated previews out of public commits.
