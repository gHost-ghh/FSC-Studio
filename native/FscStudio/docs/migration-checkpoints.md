# Native C++ Migration Checkpoints

## Branches

- `codex/native-winui-prototype` preserves the abandoned WinUI/C# prototype.
- `codex/native-cpp-qt-restart` starts from `origin/main` and contains the C++/Qt restart.

## Checkpoint 1: Native Database And Identity Probe

Status: verified for local `new_full.fscdb`, including native Identity Gallery rebuild on a copied database.

Acceptance:

- Build `fsc_native_probe`.
- Build `fsc_vision_probe`.
- `fsc_native_probe D:\FSC\new_full.fscdb stats` reads `.fscdb` v8 metadata and counts.
- `fsc_native_probe D:\FSC\new_full.fscdb search <face_id>` returns cosine-ranked faces.
- `fsc_native_probe D:\FSC\new_full.fscdb identify <face_id>` returns Identity Gallery candidates from existing profiles.
- `fsc_native_probe <copy.fscdb> train-profiles` rebuilds local Identity Gallery profiles without Python.
- `fsc_native_probe <new.fscdb> create-db` creates a Python-compatible v8 `.fscdb` without Python.
- `fsc_native_probe <copy.fscdb> import-image <model_root> <image_path>` analyzes a JPEG/PNG/BMP/PPM image through native ONNX and inserts detected faces into `.fscdb`.
- `fsc_native_probe <database.fscdb> add-person`, `assign-person`, `people`, and `train-profiles` cover the minimal native People -> assigned faces -> Identity Gallery loop.
- `fsc_vision_probe models D:\FSC\model\insightface\models` validates all local buffalo_l model paths before ONNX inference is wired.
- `fsc_vision_probe onnx <model>` inspects model I/O after `FSC_ENABLE_ONNX=ON` is configured.
- ONNX-enabled probe must copy the matching `onnxruntime.dll` beside the executable; relying on PATH is not enough because Windows can load an older `System32\onnxruntime.dll`.

Observed local database probe:

- `fsc_native_probe D:\FSC\new_full.fscdb stats`: format `8`, metric `cosine_normed_embedding`, model `buffalo_l`, faces `130`, people `119`, review `3`, average quality `0.8839`.
- `fsc_native_probe D:\FSC\new_full.fscdb search 1 5`: returned cosine-ranked candidates from stored float32 embeddings.
- `fsc_native_probe D:\FSC\new_full.fscdb identify 1 strict`: returned `review`, best profile for face id `1`, score `1.0000`, weak-profile message.
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_train.fscdb train-profiles 0.35 12`: rebuilt `119` profiles from `127` usable samples on a copied database; `117` profiles were weak because most people have fewer than 3 confirmed faces.
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_train.fscdb identify 1 strict`: returned `review`, best candidate face id `1`, score `1.0000`, weak-profile message after native rebuild.
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_import.fscdb import-image D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg 0.50`: inserted face id `131` on a copied database, stored SHA-256 `6b3d861f077ee31884a7a37adb547b86095bc0bf5042cc5c2ea4b35949fb628d`, and marked the import as duplicate because the same source image hash already existed.
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_import.fscdb search 131 5`: found original face id `1` as the best non-self hit with cosine `0.9801`.
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_created.fscdb create-db`: created an empty v8 database with model `buffalo_l`.
- Importing `baiyh.jpg` into that new database inserted face id `1`; importing it a second time inserted face id `2` with `duplicate=true`, and searching face id `2` returned face id `1` with cosine `1.0000`.
- On `native_people.fscdb`, the native probe created a database, imported `baiyh.jpg`, added person `NativeTest`, assigned face id `1`, rebuilt identity profiles, and identified face id `1` as `NativeTest` with score `1.0000` in `review` because the profile is weak.

## Checkpoint 2: Native InsightFace Inference

Status: first native image-to-database path verified on one sample.

Acceptance:

- Load `det_10g.onnx`, `w600k_r50.onnx`, `2d106det.onnx`, and `1k3d68.onnx` through ONNX Runtime C++.
- Match Python InsightFace detection, alignment, embedding normalization, and top-k search ranking on a fixed test set.

Observed local buffalo_l I/O through `fsc_vision_probe` with ONNX Runtime 1.27.0 CPU:

- `det_10g.onnx`: `input.1` float32 `[1,3,-1,-1]`, outputs scores `[12800,1] [3200,1] [800,1]`, boxes `[12800,4] [3200,4] [800,4]`, landmarks `[12800,10] [3200,10] [800,10]`.
- `w600k_r50.onnx`: `input.1` float32 `[-1,3,112,112]`, output `[1,512]`.
- `2d106det.onnx`: `data` float32 `[-1,3,192,192]`, output `fc1` `[1,212]`.
- `1k3d68.onnx`: `data` float32 `[-1,3,192,192]`, output `fc1` `[1,3309]`.

First native SCRFD + ArcFace parity sample:

- Source image: `D:\FSC\test_img\123s2\baiyh.jpg`, converted to temporary P6 PPM for the native probe.
- The same source JPEG now loads directly through Windows Imaging Component; no Python conversion is needed for native probe runs.
- C++ native ONNX CPU: 1 face, score `0.8198`, box `[112.2487,145.7734,367.3689,459.4835]`, embedding dim `512`, norm `1.0000`.
- Python InsightFace CPU reference: 1 face, score `0.8825`, box `[115.1451,141.5938,367.8138,455.8132]`, embedding dim `512`, norm `1.0000`.
- C++ embedding vs Python embedding cosine: `0.9801`.

First native image search sample:

- Command: `fsc_native_probe D:\FSC\new_full.fscdb image-search D:\FSC\model\insightface\models D:\FSC\native\FscStudio\out\probe\baiyh.ppm 5 0.50 strict`.
- Result: 1 detected face, native embedding norm `1.0000`, `106` 2D landmarks, `68` 3D landmarks.
- Native quality score: `0.8542`.
- Identity Gallery result: `review` because the best profile is weak; best candidate was face id `1` with score `0.9801`.
- Similar-face search Top 5: face ids `1`, `41`, `52`, `51`, `60`; top hit was the expected source identity.

First native landmark parity sample:

- C++ 2D landmark first point: `[236.1922,461.5391]`.
- Python InsightFace 2D landmark first point with the same box: `[236.1581,461.5482]`.
- C++ 3D landmark first point: `[107.2578,267.3733,150.0926]`.
- Python InsightFace 3D landmark first point with the same box: `[107.2689,267.3744,150.3109]`.
- C++ quality score: `0.8542`; Python quality score with the same box and detection score: `0.854203`.

## Checkpoint 3: Dense Mesh And Camera

Status: pending.

Acceptance:

- Read and display existing `landmarks3d_json` and `face_mesh3d_json` without Python.
- Generate Dense Mesh natively or through a clearly documented native dependency.
- Camera uses native capture, native inference, cached identity profiles, and short-term smoothing.

## Checkpoint 4: Qt Desktop App And Installer

Status: minimal Qt desktop shell verified; full page parity and installer pending.

Acceptance:

- Qt Widgets app covers Overview, Library, People, Search, Camera, Review, Clusters, Compare, and Runtime.
- Installer contains Qt runtime, ONNX Runtime, models, and app files, but no Python runtime and no user data.

Current Qt shell:

- Builds with preset `msvc-vs-qt-debug` after vcpkg installs the `qt-app` feature.
- Includes Overview, Library, People, Search, and Import tabs backed by native `FscCore` / `FscVision`.
- Can create/open `.fscdb`, list faces and people, search by face id, identify by face id, and import images through native ONNX.
- `FscStudioQt.exe --smoke D:\FSC\new_full.fscdb`: exit code `0`.
- Launch check: `FscStudioQt.exe D:\FSC\new_full.fscdb` stayed running for 3 seconds with the Qt runtime DLLs copied beside the executable.

Dependency notes:

- `qtbase` was installed through vcpkg feature `qt-app`.
- vcpkg downloads for `mity-md4c-release-0.5.3.tar.gz` and `strawberry-perl-5.42.2.1-64bit-portable.zip` needed manual caching because GitHub downloads intermittently failed through the current proxy.
