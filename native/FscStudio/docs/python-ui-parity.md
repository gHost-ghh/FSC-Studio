# Python UI Parity Checklist

The native Qt application must match the current Python `fsc_studio.py` layout,
workflows, and user experience before the migration can be considered complete.

## Global Shell

- Page order: Overview, Library, People, Search, Camera, Review, Clusters, Compare, Runtime.
- Left navigation must provide language switching for English, Chinese, Japanese, and Korean.
- Status bar, page titles, shortcut buttons, and tab labels must use the same translated text keys as the Python version.
- The native application must not expose migration-only pages such as a separate Import tab in the final UI; importing belongs in Library.

## Overview

- Summary cards: Faces, People, Tags, Review, Ignored, Avg Quality.
- Review Queue and Top People tables.
- Shortcut buttons to Review Queue, People, Search, and Clusters.

## Library

- Database controls: create/open/import images/import folder/reload/export.
- Filter controls: text, person, tag, include ignored, review state, minimum quality.
- Main table columns: ID, Name, Person, Tags, Review, Ignored, Dupes, Quality, Source.
- Right visual panel: Image, 3D Landmarks, Dense Mesh.
- Image preview must support focus-on-face behavior.
- Dense Mesh must support Points/Textured modes and optional 3D landmark overlay.
- Metadata tabs: Selected, Batch, Activity.
- Selected metadata must edit person, tags, review state, ignored, and notes.
- Batch metadata must edit person, tags, review, ignored, and notes across selected faces.
- Import progress must show the currently processed image preview.

## People

- Person add/update controls with notes.
- People table with identity status/health, sample count, exemplar count, thresholds.
- Member table for selected person.
- Train Identity Profiles workflow.
- Confirmed face assignment and profile rebuild must match Python semantics.

## Search

- Query image controls with Use Library DB, file picker, threshold, top K, min quality, and identity mode.
- Multi-face selection in query image.
- Similar-face result list plus identity result panel above it.
- Preview should animate through compared database faces and end on the best match.
- Result actions must support assigning/confirming person as in Python.

## Camera

- Camera index, threshold, top K, min quality, identity mode, start/stop, and Use Library DB.
- Live preview with boxes and identity display.
- Similar hits and identity candidates beside the preview.
- Short-term identity smoothing must not override unknown/review safety decisions.

## Review

- Review queue filters and table.
- Detail preview and metadata editor.
- Mark reviewed, ignore, assign/confirm suggested person, and train/update profiles.

## Clusters

- Threshold/min-size controls.
- Cluster table columns: Cluster, Faces, Mean, Max, Avg Quality, Known People.
- Member table columns: ID, Name, Person, Tags, Quality, Review.
- Cluster actions must allow assigning a person to selected cluster members.

## Compare

- Two image selectors with previews.
- Multi-face selection for both images.
- Similarity result and landmark/quality details.

## Runtime

- Runtime provider display and model status.
- Maintenance actions: integrity check, backup, checkpoint, vacuum, and operation log.
- Trusted legacy `.dtb` conversion with progress, an output `.fscdb`, and retained local preview images.

## Current Native Gaps

- Core file handling now uses explicit UTF-8/native-wide path conversion rather than the active Windows code page. Database create/open/backup, CSV export, image hashing, model diagnostics, legacy conversion, and smoke-command arguments are covered by CJK mixed-path regression tests.
- Static interface translations now retain stable English keys instead of reconstructing keys from translated text. English, Chinese, Japanese, and Korean can be cycled repeatedly without collisions; labels, controls, tabs, table headers, placeholders, and tooltips have a dedicated coverage smoke.
- Native shell uses the Python page order with lower-left language / identity-mode controls; Overview now matches the Python title/margins, workspace actions, quality range, attention queue, Top People, and Top Tags panels. Fixed columns and scrollable tables keep the complete layout usable at 1180x760, and dynamic metric labels follow language changes.
- Native Library now has create/open, single-image import, recursive folder import, reload, filtered CSV export, text/person/tag/review/quality filters, selected-face Image / 3D Landmarks / Dense Mesh visual tabs, focus toggle, selected/batch metadata tabs, Activity logging, progress, and import preview.
- Native People now has database/filter controls, the Python-style identity health table, member table, selected-member preview with focus toggle, person name/notes editing, merge into target, clear assignment, manual face assignment, and profile training.
- Native Search now mirrors the Python page layout and operation flow: it exposes Open Database / Use Library DB controls, automatically detects faces on a worker thread after image selection, supports list and preview-box face selection, and provides independent query/result focus toggles. Search filtering, identity candidates, throttled progressive result preview, and final best-match selection remain native and responsive.
- Native Compare now mirrors the Python Images panel, paired previews/lists, compact focus controls, and bottom result metric. Image A/B detection runs asynchronously with stale-result generation guards; both slots reuse the native ONNX session cache, while list and preview-box selection remain synchronized.
- Native Review now has queue filter/limit controls, Python-style reason/duplicate columns, selected-face preview with focus toggle, metadata editing for person/tags/review/ignored/notes, automatic AI suggestion, confirm AI person, and reject AI suggestion.
- Native Clusters now matches the Python title, grouped parameter band, dual-table layout, selected-member preview, compact face-focus control, and batch assignment panel. Cluster construction uses block matrix multiplication and runs on a worker connection; assignment is one SQLite transaction and profile rebuilding also stays off the UI thread.
- Native Runtime now matches the Python title/group layout, creates a real ONNX session asynchronously, reports the actual CPU/DirectML provider, and runs integrity check, backup, WAL checkpoint, VACUUM, and legacy conversion on worker connections. Legacy progress is queued back to the UI without `processEvents` re-entry.
- Native Runtime now matches the Python legacy conversion entry point: opening a `.dtb` routes to `Convert Legacy DTB`, while the restricted native pickle reader extracts embedded RGB images without embedding Python. It re-analyzes the images through native ONNX and writes a sibling preview directory for the converted database.
- Native Camera now mirrors the Python controls and two-column layout, including Open Database / Use Library DB, 300 ms default recognition interval, live preview, focusable match preview, status, and the 11-column identity/similar-hit table. Frame capture stays on a 33 ms UI timer while ONNX recognition and cached-gallery search run in a guarded background task using immutable database snapshots; stale results are discarded across stop/restart or database changes.
- Dense Mesh uses the exact 478-point MediaPipe tessellation in native Points or Textured modes. Textured mode uses depth-tested image sampling, darkens only back-facing triangle surfaces, supports drag/zoom/reset, and overlays the cached 68-point landmarks on the dense surface only when enabled. Generation directly calls the MediaPipe native C Task API with the same `face_landmarker.task`, thresholds, coordinate conversion, and face-selection logic as the Python application. Invalid old caches (including former synthetic 760-point meshes) are rejected and can be repaired with `fsc_native_probe <db> repair-invalid-meshes <face_landmarker.task>`; source images on which MediaPipe detects no mesh remain uncached rather than receiving fabricated geometry.
- Native Inno Setup installer now packages the Qt runtime, ONNX Runtime, MediaPipe runtime, and models without Python or user data. It runs a Windows-platform Qt smoke test before producing Setup; formal release signing remains a distribution task, not a Python-parity gap.
