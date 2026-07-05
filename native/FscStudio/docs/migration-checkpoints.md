# Native C++ Migration Checkpoints

## Branches

- `codex/native-winui-prototype` preserves the abandoned WinUI/C# prototype.
- `codex/native-cpp-qt-restart` starts from `origin/main` and contains the C++/Qt restart.

## Checkpoint 1: Native Database And Identity Probe

Status: started.

Acceptance:

- Build `fsc_native_probe`.
- `fsc_native_probe D:\FSC\new_full.fscdb stats` reads `.fscdb` v8 metadata and counts.
- `fsc_native_probe D:\FSC\new_full.fscdb search <face_id>` returns cosine-ranked faces.
- `fsc_native_probe D:\FSC\new_full.fscdb identify <face_id>` returns Identity Gallery candidates from existing profiles.

## Checkpoint 2: Native InsightFace Inference

Status: next.

Acceptance:

- Load `det_10g.onnx`, `w600k_r50.onnx`, `2d106det.onnx`, and `1k3d68.onnx` through ONNX Runtime C++.
- Match Python InsightFace detection, alignment, embedding normalization, and top-k search ranking on a fixed test set.

## Checkpoint 3: Dense Mesh And Camera

Status: pending.

Acceptance:

- Read and display existing `landmarks3d_json` and `face_mesh3d_json` without Python.
- Generate Dense Mesh natively or through a clearly documented native dependency.
- Camera uses native capture, native inference, cached identity profiles, and short-term smoothing.

## Checkpoint 4: Qt Desktop App And Installer

Status: pending.

Acceptance:

- Qt Widgets app covers Overview, Library, People, Search, Camera, Review, Clusters, Compare, and Runtime.
- Installer contains Qt runtime, ONNX Runtime, models, and app files, but no Python runtime and no user data.
