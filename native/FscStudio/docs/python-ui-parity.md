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

- Native shell now uses left navigation and lower-left language / identity-mode controls, but the page internals remain structurally simplified.
- Native Library now has a selected-face preview with bbox/landmarks and focus toggle, but still lacks the full Python right visual/metadata panel.
- Native Search can analyze a standalone query image and choose among detected faces, but still lacks Python's full result action panel and animated comparison preview.
- Native UI does not yet match the Python Review, Camera, and Runtime workflows.
- Dense Mesh native generation currently has a deterministic synthetic fallback from 3D landmarks. It is not a MediaPipe-equivalent textured mesh and cannot be treated as final parity.
- Final installer should be MSI/WiX or QtIFW style; the current package includes portable files plus install/uninstall scripts.
