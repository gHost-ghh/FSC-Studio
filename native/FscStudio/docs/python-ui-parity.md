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

## Current Native Gaps

- Native shell now uses left navigation and lower-left language / identity-mode controls, but some page internals remain structurally simplified.
- Native Library now has create/open, single-image import, recursive folder import, reload, filtered CSV export, text/person/tag/review/quality filters, selected-face Image / 3D Landmarks / Dense Mesh visual tabs, focus toggle, selected/batch metadata tabs, Activity logging, progress, and import preview.
- Native People now has database/filter controls, the Python-style identity health table, member table, selected-member preview with focus toggle, person name/notes editing, merge into target, clear assignment, manual face assignment, and profile training.
- Native Search can analyze a standalone query image, choose among detected faces, filter by threshold/min-quality/ignored/person/tag state, show identity candidates plus a separate result preview, cycle quickly through top result previews, and assign/confirm people from results. It still lacks Python's true background progressive comparison status for every database face.
- Native Compare now supports two previews, detected-face lists, click-to-select boxes, focus toggles, and selected-pair comparison.
- Native Review now has queue filter/limit controls, Python-style reason/duplicate columns, selected-face preview with focus toggle, metadata editing for person/tags/review/ignored/notes, automatic AI suggestion, confirm AI person, and reject AI suggestion.
- Native Clusters now includes known people, member tags, max-face/min-quality/unassigned/ignored filters, selected-member preview, and batch cluster assignment.
- Native Runtime now includes current database stats and maintenance actions for integrity check, backup, WAL checkpoint, VACUUM, and operation logging.
- Native Camera now includes threshold/top-k/interval/process-size controls, current database status, live boxes, best-match preview, per-face identity smoothing, an identity-plus-similar-hit table, and selected-result actions for confirming identity, assigning a matched face, and marking a match reviewed.
- Dense Mesh generation directly calls the MediaPipe native C Task API with the same `face_landmarker.task`, thresholds, coordinate conversion, and face-selection logic as the Python application. Invalid old caches (including former synthetic 760-point meshes) are rejected and can be repaired with `fsc_native_probe <db> repair-invalid-meshes <face_landmarker.task>`; source images on which MediaPipe detects no mesh remain uncached rather than receiving fabricated geometry.
- Final installer should be MSI/WiX or QtIFW style; the current package includes portable files plus install/uninstall scripts.
