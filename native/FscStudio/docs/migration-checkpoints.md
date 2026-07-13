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
- `fsc_native_probe <database.fscdb> add-person`, `assign-person`, `people`, `train-profiles`, and `update-review` cover the minimal native People/Review -> assigned faces -> Identity Gallery loop.
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
- `fsc_native_probe D:\FSC\native\FscStudio\out\probe\native_review.fscdb update-review 1 reviewed 0 native-review-smoke`: updated face id `1` to `reviewed` with `ignored=false` on a copied database.

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

Status: cached 3D data display, native Points/Textured Dense Mesh rendering, ONNX-native 478-point Dense Mesh generation, and native Camera capture are implemented.

Acceptance:

- Read and display existing `landmarks3d_json` and `face_mesh3d_json` without Python.
- Generate Dense Mesh from the official MediaPipe landmark network directly through ONNX Runtime using the same point convention as Python.
- Camera uses native capture, native inference, cached identity profiles, and short-term smoothing.

Current Dense Mesh shell:

- `loadFace()` now parses cached `landmarks3d_json` and `face_mesh3d_json`; `loadFaces()` remains lightweight and does not pull dense mesh JSON into search/list paths.
- Qt Dense Mesh panels render only validated cached dense mesh data, with native Points/Textured modes, image texture sampling, per-pixel depth buffering, back-facing triangle darkening, drag/zoom/reset, and an optional 68-point landmark overlay in Textured mode. The Textured path augments the base 852-triangle topology with local eyelid/iris Delaunay triangles, so MediaPipe iris points 468--477 render their original eyeball texture.
- Qt Dense Mesh tabs generate and cache a validated 478-point MediaPipe mesh from the original source image, never from the 68-point landmark cache.
- `fsc_native_probe <database.fscdb> build-mesh <face_id> <face_landmarks_detector.onnx>` writes `face_mesh3d_json` without Python. `repair-invalid-meshes` removes/rebuilds old non-478-point caches and uses the selected face bbox/keypoints instead of a largest-face fallback.
- `FscStudioQt.exe --mesh-smoke D:\FSC\new_full.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --mesh-render-smoke D:\FSC\new_full.fscdb 1 out\probe\mesh_render.png [yaw pitch]`: renders a nonblank textured mesh screenshot without opening a user window; optional angles exercise back-surface darkening.

Current Camera shell:

- OpenCV is optional through vcpkg feature `opencv` with default OpenCV features disabled; only `core`, `dshow`, `msmf`, and `thread` are enabled.
- Qt Camera tab captures frames natively through OpenCV on a 33 ms UI timer. Configurable-size ONNX recognition runs on a background worker, so model initialization and inference no longer pause live preview or box refresh.
- Database face embeddings and Identity Gallery profiles are immutable snapshots rebuilt only when the opened database changes. Recognition results carry camera-session, task-token, and database-path guards before UI application; short vote windows smooth names without overriding `unknown` safety decisions.
- The page mirrors Python controls and layout: Open Database / Use Library DB, threshold, top-k, 300 ms default interval, process size, start/stop, live boxes, focusable best-match preview, and the 11-column identity/similar-hit table.
- `FscStudioQt.exe --camera-smoke`: exit code `0`.
- `FscStudioQt.exe --camera-open-smoke 0`: exit code `0` on this machine; OpenCV selected the MSMF backend and captured a frame from camera index `0`.
- `FscStudioQt.exe --camera-result-smoke D:\FSC\model\insightface\models D:\FSC\new_full.fscdb D:\FSC\test_img\123s2\baiyh.jpg directml`: exit code `0`.
- `FscStudioQt.exe --camera-ui-smoke D:\FSC\new_full.fscdb D:\FSC\model\insightface\models D:\FSC\test_img\test\baiyh.jpg cpu|directml`: both modes exit `0` after populating native Camera UI results.
- `FscStudioQt.exe --camera-live-smoke D:\FSC\new_full.fscdb D:\FSC\model\insightface\models 0 cpu`: exit code `0` after an 8-second physical-camera run proved at least 20 captured frames and one completed background recognition.

Current accelerator shell:

- `fetch-onnxruntime-directml.ps1`: downloads and extracts `Microsoft.ML.OnnxRuntime.DirectML` to `.deps\onnxruntime-directml-1.24.4-nuget`.
- DirectML builds use `dml_provider_factory.h`, disable memory pattern optimization, force sequential execution mode, and append `SessionOptionsAppendExecutionProvider_DML`.
- `fsc_vision_probe.exe onnx D:\FSC\model\insightface\models\buffalo_l\det_10g.onnx directml`: provider `DmlExecutionProvider`, exit code `0`.
- `FscStudioQt.exe --compare-smoke D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg directml`: exit code `0`.
- `FscStudioQt.exe --camera-result-smoke D:\FSC\model\insightface\models D:\FSC\new_full.fscdb D:\FSC\test_img\123s2\baiyh.jpg directml`: exit code `0`.
- x64 CUDA builds load `CUDAExecutionProvider` for all four InsightFace sessions and distribute the tested CUDA 13/cuDNN 9 runtime subset.
- Native ARM64 builds load Qualcomm QNN. Auto tries quantized HTP/NPU models, then float Adreno GPU, then CPU. Dense Mesh uses Adreno GPU or CPU because the official float display model is not quantized for HTP.

## Checkpoint 4: Qt Desktop App And Installer

Status: expanded Qt desktop shell and Python-free Inno Setup installer verified; the remaining migration work is tracked as a page-by-page functional and UX parity audit.

Acceptance:

- Qt Widgets app covers Overview, Library, People, Search, Camera, Review, Clusters, Compare, and Runtime.
- Installer contains Qt runtime, ONNX Runtime, models, and app files, but no Python runtime and no user data.

Current Qt shell:

- Builds with preset `msvc-vs-qt-debug` after vcpkg installs the `qt-app` feature.
- Includes Overview, Library, People, Search, Camera, Review, Clusters, Compare, and Runtime pages backed by native `FscCore` / `FscVision`; Dense Mesh is part of the selected-face Library workflow as in Python.
- Uses the Python app's left navigation order with lower-left language and identity-mode controls.
- Can create/open `.fscdb`, list faces and people, add people, assign selected faces to people, train identity profiles, search by face id, identify by face id, import images or folders from the Library page through native ONNX, and export the filtered Library table to CSV.
- Library has right-side Image / 3D Landmarks / Dense Mesh visual tabs for the selected face. Image overlays cached bbox / 2D landmarks and toggles full-image versus face-focused viewing; Dense Mesh uses only validated MediaPipe caches and explicitly asks the user to generate or repair missing/invalid data.
- Library now matches the Python page structure with a full-width Database/filter band, fixed 430--500 px right visual/detail column, compact overlaid focus control, text/person/tag/review/quality filters, Selected and Batch metadata tabs, and an Activity tab. Multi-image and recursive-folder imports run on a background worker, batch SQLite writes in groups of 50, and coalesce preview rendering at 50 ms so visual activity does not throttle inference. Missing or incompatible Dense Mesh data is generated on a reusable background MediaPipe worker and cached without blocking selection changes.
- The main tab container ignores hidden-page size hints. A native `--page-render-smoke` capture verified Library at exactly 1180x760 instead of the previous forced 2265x985 minimum, while the face table remains horizontally scrollable and the detail column stays fully visible.
- Overview now matches the Python title, spacing, workspace command wording, database/attention summaries, Top People, and Top Tags layout. Native database statistics include minimum/maximum quality, metric labels are translated when the language changes, and 1180x760 plus 1600x1000 captures show no overlap.
- People now mirrors the Python person-management workflow: equal-width person/member/detail columns with horizontal table scrolling, identity health/scorer fields, selected-member preview with compact focus overlay, name/notes editing, merge into target, and assignment clearing. The earlier extra manual ID assignment controls were removed from this page; identity-profile training now runs on a background database connection and refreshes all dependent cached views on completion.
- Review now mirrors the Python queue workflow with a responsive 2:1 queue/detail split, fixed-width scrollable columns, compact selected-face focus overlay, person/tag/review/ignored/notes editor, automatic identity suggestion, confirm AI person, and reject AI suggestion. Suggestions run against immutable cached profiles on background workers with face/generation guards; confirmation, review mutation, and profile rebuilding run on a worker database connection and preserve existing notes.
- Search can now analyze a standalone query image natively, choose among detected query faces, preview boxes, filter by threshold/min-quality/ignored/person/tag state, show identity candidates, compare the filtered database set in time-sliced batches with a throttled live preview, end on the best match, and assign/confirm people from result actions.
- Compare now mirrors the Python multi-face workflow more closely: each side analyzes after image selection, lists detected faces, supports preview-box click selection, focus/full-image toggles, and compares the selected pair.
- Review now shows a selected-face detail preview, can run the native identity suggestion scorer, and can confirm the suggested person while retraining identity profiles.
- Review can update `review_state`, `ignored`, and notes on selected faces.
- Compare can analyze two image files through native ONNX and report embedding cosine plus detection/quality/landmark counts.
- Clusters now matches the Python grouped controls and responsive three-column layout at 1180x760 and 1600x1000. It supports cosine threshold, minimum size, max faces, minimum quality, unassigned-only, and ignored-face options, plus fixed-width scrollable tables, member preview, and a compact focus toggle. OpenCV block matrix multiplication accelerates pair scoring; build, transactional batch assignment, and Identity Gallery rebuilding run on workers with database/token guards.
- Camera result selection updates the evidence/best-hit preview locally; person assignment and review mutations remain on the People and Review pages, matching the Python workflow.
- Runtime exposes the provider modes compiled into each release: Auto / CPU / DirectML on standard x64, Auto / CPU / CUDA on NVIDIA x64, and Auto / CPU / QNN NPU / QNN GPU on native ARM64. Auto validates all four InsightFace sessions before accepting a provider and then falls back as a group.
- Runtime shows current database stats and runs SQLite integrity check, backup, WAL checkpoint, and VACUUM on separate worker connections with guarded completion and operation logging.
- Runtime converts trusted legacy `.dtb` files without Python on a worker: it parses only the known FSC tuple/NumPy-image format, re-extracts embeddings with native ONNX, writes `<output>_legacy_images` PPM previews, and opens the converted v8 database. Progress crosses a mutex-protected queue instead of touching Qt widgets from the worker or re-entering the event loop. `fsc_native_probe <output.fscdb> convert-legacy-dtb <source.dtb> <model_root> [auto|cpu|directml|cuda|qnn-npu|qnn-gpu] [limit]` remains the reproducible non-UI path.
- Packaged builds prefer `models/insightface/models` next to the executable before falling back to the source checkout model path.
- `FscStudioQt.exe --smoke D:\FSC\new_full.fscdb`: exit code `0`.
- `FscStudioQt.exe --review-smoke D:\FSC\native\FscStudio\out\probe\native_review_qt.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --cluster-smoke D:\FSC\new_full.fscdb`: exit code `0`.
- `FscStudioQt.exe --cluster-action-smoke D:\FSC\native\FscStudio\out\probe\native_cluster_action_smoke.fscdb NativeClusterSmoke`: exit code `0`.
- `FscStudioQt.exe --mesh-smoke D:\FSC\new_full.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --library-visual-smoke D:\FSC\new_full.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --metadata-smoke D:\FSC\native\FscStudio\out\probe\native_metadata_smoke.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --search-action-smoke D:\FSC\native\FscStudio\out\probe\native_search_action_smoke.fscdb 1 1 NativeSearchSmoke`: exit code `0`.
- `FscStudioQt.exe --camera-action-smoke D:\FSC\native\FscStudio\out\probe\native_camera_action_smoke.fscdb 1 NativeCameraSmoke`: exit code `0`.
- `FscStudioQt.exe --search-filter-smoke D:\FSC\native\FscStudio\out\probe\native_search_filter_smoke.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --library-export-smoke D:\FSC\new_full.fscdb D:\FSC\native\FscStudio\out\probe\native_library_export_smoke.csv`: exit code `0`.
- `FscStudioQt.exe --library-import-ui-smoke D:\FSC\native\FscStudio\out\probe\library-import-smoke.fscdb D:\FSC\model\insightface\models D:\FSC\test_img\test\baiyh.jpg cpu`: exit code `0`, completed in about 2.46 seconds, and committed one detected face to a newly created v8 database.
- `FscStudioQt.exe --library-mesh-ui-smoke D:\FSC\native\FscStudio\out\probe\library-mesh-ui-smoke.fscdb 1`: exit code `0`; after clearing the cache, the asynchronous UI path regenerated and committed a valid 478-point mesh in about 2.14 seconds.
- `FscStudioQt.exe --people-training-ui-smoke D:\FSC\native\FscStudio\out\probe\people-training-ui-smoke.fscdb`: exit code `0`; rebuilt 119 local identity profiles through the asynchronous People UI path in about 0.58 seconds.
- `FscStudioQt.exe --review-ai-ui-smoke D:\FSC\native\FscStudio\out\probe\review-ai-ui-smoke.fscdb 1`: exit code `0`; suggested the known identity, confirmed it, preserved person id `3`, changed the queue state from `open` to `reviewed`, and rebuilt profiles in about 0.62 seconds.
- `FscStudioQt.exe --review-suggestion-switch-ui-smoke D:\FSC\native\FscStudio\out\probe\review-switch-ui-smoke.fscdb 1 2`: exit code `0`; rapid selection switched from person `3` to person `4`, and the final suggestion remained person `4` after both workers completed.
- `FscStudioQt.exe --clusters-ui-smoke D:\FSC\new_full.fscdb`: exit code `0`; the native asynchronous path built real clusters in about 0.55 seconds without blocking the event loop.
- `FscStudioQt.exe --clusters-assign-ui-smoke D:\FSC\native\FscStudio\out\probe\clusters-assign-ui-smoke.fscdb NativeClusterUiSmoke`: exit code `0`; three faces were atomically assigned, tagged `cluster-suggested`, marked reviewed without overwriting notes, and rebuilt into a `gallery_v2` identity profile in about 0.79 seconds.
- `FscStudioQt.exe --runtime-probe-ui-smoke D:\FSC\model\insightface\models cpu|directml|auto [expected-provider]`: all modes exit `0` after loading all four InsightFace sessions; CPU reports `CPUExecutionProvider`, while DirectML and Auto report `DmlExecutionProvider` on this machine (about 1.38-2.09 seconds per full probe). The CPU-only flavor's Auto smoke explicitly expects `CPUExecutionProvider` and also exits `0`.
- `FscStudioQt.exe --runtime-maintenance-ui-smoke <copy.fscdb> integrity|checkpoint|backup|vacuum [backup-output]`: all four worker paths exit `0`; the verified backup passes `PRAGMA integrity_check` and contains all 130 source faces.
- `FscStudioQt.exe --runtime-legacy-ui-smoke <source.dtb> <output.fscdb> <model-root> cpu|directml|auto`: exit code `0` with the restricted one-row fixture; conversion completed asynchronously in about 1.76 seconds, produced an integrity-clean v8 database, and safely skipped the intentionally faceless 3x2 image.
- `FscStudioQt.exe --review-action-smoke D:\FSC\native\FscStudio\out\probe\native_review_action_smoke.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --people-action-smoke D:\FSC\native\FscStudio\out\probe\native_people_action_smoke.fscdb 1`: exit code `0`.
- `FscStudioQt.exe --maintenance-smoke D:\FSC\native\FscStudio\out\probe\native_maintenance_smoke.fscdb D:\FSC\native\FscStudio\out\probe\native_maintenance_smoke_backup.fscdb`: exit code `0`.
- `FscStudioQt.exe --compare-smoke D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg`: exit code `0`.
- Launch check: `FscStudioQt.exe D:\FSC\new_full.fscdb` stayed running for 3 seconds with the Qt runtime DLLs copied beside the executable.

Current release packages:

- x64 DirectML, x64 CUDA, and native ARM64 QNN builds all compile from the same source tree; CTest passes for both runnable x64 flavors and the ARM64 cross-build completes with Qt, OpenCV, SQLite, ONNX Runtime QNN, and VC runtime staging.
- `package-qt-portable.ps1` validates PE architecture and stages only the selected provider, required models, Qt/OpenCV/SQLite runtime, and VC++ redistributable. Its 0.2.0 manifest records provider and model hashes and states that no Python runtime or user database is present.
- Packaged DirectML and CUDA builds pass full provider-group probing, two-image face comparison, and 478-point Dense Mesh generation against copied database files. Auto reports `DmlExecutionProvider` and `CUDAExecutionProvider` respectively.
- `FSC-Studio-Setup-x64.exe` is the 312.2 MiB DirectML installer; `FSC-Studio-CUDA-Setup-x64.exe` is the 1226.0 MiB CUDA installer with the tested CUDA 13/cuDNN 9 subset; `FSC-Studio-Setup-arm64.exe` is the 391.6 MiB QNN installer.
- The ARM64 executable and all staged native dependencies have ARM64 PE architecture. It cannot be executed on the current x64 build machine, so final HTP/NPU and Adreno acceptance must run on the Surface Pro 11 hardware.
- The standard release path is an Inno Setup executable rather than MSIX, so it does not depend on a locally trusted test certificate. Public distribution still benefits from Authenticode signing.

Dependency notes:

- `qtbase` was installed through vcpkg feature `qt-app`.
- vcpkg downloads for `mity-md4c-release-0.5.3.tar.gz` and `strawberry-perl-5.42.2.1-64bit-portable.zip` needed manual caching because GitHub downloads intermittently failed through the current proxy.
