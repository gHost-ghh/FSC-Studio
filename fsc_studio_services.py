from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import cv2

from fsc_face_database import (
    FaceRecord,
    PersonRecord,
    TagRecord,
    assign_person as db_assign_person,
    connect_database,
    create_identity_profile_schema,
    database_statistics,
    ensure_person,
    existing_image_hashes,
    initialize_database,
    insert_face,
    load_face_preview as db_load_face_preview,
    list_people as db_list_people,
    list_tags as db_list_tags,
    load_face_records,
    load_review_records,
    normalize_database_path,
    set_face_tags as db_set_face_tags,
    set_face_tags_on_connection,
    update_face_mesh3d as db_update_face_mesh3d,
    update_face_landmarks3d as db_update_face_landmarks3d,
    update_face_review as db_update_face_review,
)
from fsc_face_engine import (
    MODEL_NAME,
    AnalyzedFace,
    FaceEngineError,
    analysis_to_database_record,
    cosine_similarity,
    get_engine,
    read_image_bgr,
    similarity_percent,
)


ProgressCallback = Callable[[str, int, int], None]
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
MEDIAPIPE_MODEL_DIR = Path(__file__).resolve().parent / "model" / "mediapipe"
MEDIAPIPE_FACE_LANDMARKER_PATH = MEDIAPIPE_MODEL_DIR / "face_landmarker.task"
MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
IDENTITY_MIN_STRONG_SAMPLES = 3
IDENTITY_MAX_PROTOTYPES = 5
IDENTITY_MAX_EXEMPLARS = 12
IDENTITY_DEFAULT_MIN_QUALITY = 0.35
IDENTITY_PROTOTYPE_DIVERSITY = 0.965
IDENTITY_EXEMPLAR_DIVERSITY = 0.985
IDENTITY_GALLERY_STRATEGY_VERSION = "gallery_v2"
IDENTITY_NUMPY_SCORER_VERSION = "numpy_gallery_v1"
IDENTITY_SCORER_MODEL_PATH = Path(__file__).resolve().parent / "model" / "identity_scorer.onnx"
IDENTITY_MODE_ORDER = ("strict", "balanced", "broad")
IDENTITY_MODE_DEFAULTS = {
    "strict": {"accept": 0.62, "review": 0.52, "margin": 0.055},
    "balanced": {"accept": 0.57, "review": 0.47, "margin": 0.040},
    "broad": {"accept": 0.52, "review": 0.42, "margin": 0.025},
}


@dataclass
class PrimaryFace:
    face: AnalyzedFace
    detected_count: int


@dataclass
class ImportSummary:
    database_path: str
    images_total: int
    faces_saved: int
    images_without_faces: int
    failed_images: int
    low_quality_faces: int
    duplicate_images: int
    average_quality: float
    messages: list[str]


@dataclass
class SearchHit:
    record: FaceRecord
    cosine: float
    similarity: float


@dataclass
class CompareResult:
    face_a: PrimaryFace
    face_b: PrimaryFace
    cosine: float
    similarity: float


@dataclass
class FaceSearchIndex:
    database_path: str
    cache_key: tuple[tuple[str, int, int], ...]
    records: list[FaceRecord]
    metadata: dict[str, str]
    matrix: np.ndarray


@dataclass
class IdentityProfile:
    person_id: int
    person_name: str
    sample_count: int
    prototype_count: int
    embedding_dim: int
    centroid: np.ndarray
    prototypes: np.ndarray
    exemplars: np.ndarray
    exemplar_weights: np.ndarray
    accept_threshold: float
    review_threshold: float
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    quality_mean: float
    evidence_face_ids: list[int]
    exemplar_face_ids: list[int]
    hard_negative_face_ids: list[int]
    hard_negative_embeddings: np.ndarray | None
    thresholds: dict[str, dict[str, float]]
    calibration: dict[str, object]
    status: str
    strategy_version: str
    scoring_model_version: str
    updated_at: str


@dataclass
class IdentityCandidate:
    profile: IdentityProfile
    score: float
    margin: float
    confidence: float
    evidence_face_id: int | None
    scoring_model_version: str = IDENTITY_NUMPY_SCORER_VERSION
    mode: str = "strict"


@dataclass
class IdentityResult:
    decision: str
    candidates: list[IdentityCandidate]
    message: str


@dataclass
class IdentityTrainingSummary:
    profiles_built: int
    weak_profiles: int
    skipped_people: int
    samples_used: int
    messages: list[str]


@dataclass
class IdentityProfileIndex:
    database_path: str
    cache_key: tuple[tuple[str, int, int], ...]
    profiles: list[IdentityProfile]


@dataclass
class FaceCluster:
    cluster_id: int
    records: list[FaceRecord]
    mean_similarity: float
    max_similarity: float
    average_quality: float
    representative_id: int
    suggested_name: str
    existing_people: list[str]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def face_ids(self) -> list[int]:
        return [record.id for record in self.records]


@dataclass
class PersonSummary:
    id: int
    name: str
    notes: str
    face_count: int
    average_quality: float
    ignored_count: int
    review_count: int
    representative_face_id: int | None
    identity_status: str = ""
    identity_sample_count: int = 0
    identity_prototype_count: int = 0
    identity_accept_threshold: float = 0.0
    identity_updated_at: str = ""
    identity_strategy_version: str = ""
    identity_scoring_model_version: str = ""
    identity_health: str = ""


@dataclass
class TagSummary:
    id: int
    name: str
    face_count: int


@dataclass
class MaintenanceResult:
    action: str
    ok: bool
    message: str
    output_path: str = ""


@dataclass
class LegacyConversionSummary:
    source_path: str
    output_path: str
    rows_total: int
    faces_saved: int
    skipped_rows: int
    messages: list[str]


class LegacyDlibVector(list):
    def __setstate__(self, state) -> None:
        self._legacy_state = state


class LegacyDtbUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "_dlib_pybind11" and name == "vector":
            return LegacyDlibVector
        return super().find_class(module, name)


_SEARCH_INDEX_CACHE: dict[str, FaceSearchIndex] = {}
_SEARCH_INDEX_LOCK = threading.RLock()
_IDENTITY_PROFILE_CACHE: dict[str, IdentityProfileIndex] = {}
_IDENTITY_PROFILE_LOCK = threading.RLock()
_IDENTITY_SCORER_LOCK = threading.RLock()
_IDENTITY_SCORER_SESSION = None
_IDENTITY_SCORER_CACHE_KEY: tuple[str, int, int] | None = None
_FACE_LANDMARKER_LOCK = threading.RLock()
_FACE_LANDMARKER = None


def runtime_metadata(engine) -> dict[str, object]:
    import insightface
    import onnxruntime

    return {
        "model_name": MODEL_NAME,
        "insightface_version": insightface.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "providers": ",".join(engine.info.providers),
        "model_root": engine.info.model_root,
        "application": "FSC Studio",
    }


def load_legacy_dtb(path: str | Path):
    with Path(path).open("rb") as handle:
        return LegacyDtbUnpickler(handle).load()


def default_legacy_conversion_output_path(source_path: str | Path) -> str:
    source = Path(source_path)
    return str(source.with_name(f"{source.stem}_insightface.fscdb"))


def legacy_runtime_metadata(engine, source_path: str | Path) -> dict[str, object]:
    metadata = runtime_metadata(engine)
    metadata.update(
        {
            "converted_from": str(source_path),
            "converter": "FSC legacy DTB converter",
        }
    )
    return metadata


def convert_legacy_dtb_to_database(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    prefer_gpu: bool = True,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> LegacyConversionSummary:
    source = Path(source_path)
    if source.suffix.lower() != ".dtb":
        raise FaceEngineError("Legacy conversion expects a .dtb file.")
    output = Path(normalize_database_path(output_path or default_legacy_conversion_output_path(source)))
    legacy_rows = load_legacy_dtb(source)

    engine = get_engine(prefer_gpu=prefer_gpu)
    conn = initialize_database(output, legacy_runtime_metadata(engine, source), replace=True)

    saved = 0
    skipped = 0
    messages: list[str] = []
    total = len(legacy_rows) if limit is None else min(len(legacy_rows), max(0, int(limit)))

    try:
        for index, row in enumerate(legacy_rows[:total], start=1):
            try:
                image_rgb = np.asarray(row[3], dtype=np.uint8)
                file_name = str(row[4])
                if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                    raise ValueError("legacy image is not an RGB array")

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                faces = engine.extract_faces_from_bgr(image_bgr, f"legacy:{file_name}")
                if len(faces) != 1:
                    skipped += 1
                    message = f"{file_name}: skipped, detected {len(faces)} faces"
                    messages.append(message)
                    if progress:
                        progress(message, index, total)
                    continue

                face = faces[0]
                face.file_name = file_name
                face.source_path = f"legacy:{file_name}"
                record = analysis_to_database_record(face)
                record["image_hash"] = hashlib.sha256(image_rgb.tobytes()).hexdigest()
                insert_face(conn, record)
                conn.commit()
                saved += 1
                message = f"{file_name}: saved"
                messages.append(message)
                if progress:
                    progress(message, index, total)
            except Exception as exc:
                skipped += 1
                message = f"row {index}: skipped ({exc})"
                messages.append(message)
                if progress:
                    progress(message, index, total)
    finally:
        conn.close()

    clear_search_index_cache(output)
    return LegacyConversionSummary(
        source_path=str(source),
        output_path=str(output),
        rows_total=total,
        faces_saved=saved,
        skipped_rows=skipped,
        messages=messages,
    )


def create_database(database_path: str | Path, prefer_gpu: bool = True) -> str:
    engine = get_engine(prefer_gpu=prefer_gpu)
    normalized = normalize_database_path(database_path)
    conn = initialize_database(normalized, runtime_metadata(engine), replace=True)
    conn.close()
    clear_search_index_cache(normalized)
    return normalized


def import_images_to_database(
    database_path: str | Path,
    image_paths: Iterable[str | Path],
    *,
    replace: bool = False,
    prefer_gpu: bool = True,
    min_quality: float = 0.0,
    progress: ProgressCallback | None = None,
    commit_interval: int = 50,
) -> ImportSummary:
    paths = [str(path) for path in image_paths]
    normalized = normalize_database_path(database_path)
    engine = get_engine(prefer_gpu=prefer_gpu)
    db_path = Path(normalized)

    if replace or not db_path.exists():
        conn = initialize_database(normalized, runtime_metadata(engine), replace=True)
    else:
        conn = connect_database(normalized)

    faces_saved = 0
    images_without_faces = 0
    failed_images = 0
    low_quality_faces = 0
    duplicate_images = 0
    quality_total = 0.0
    messages: list[str] = []
    seen_hashes: set[str] = set()

    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        pending_writes = 0
        for index, image_path in enumerate(paths, start=1):
            path_obj = Path(image_path)
            if progress:
                progress(f"IMAGE_PREVIEW|{path_obj.resolve()}|Analyzing {path_obj.name}", index, len(paths))
            try:
                image_hash = sha256_file(image_path)
                is_duplicate = image_hash in seen_hashes or bool(existing_image_hashes(conn, [image_hash]))
                if is_duplicate:
                    duplicate_images += 1
                seen_hashes.add(image_hash)
                faces = engine.extract_faces_from_path(image_path)
            except Exception as exc:
                failed_images += 1
                message = f"{path_obj.name}: failed ({exc})"
                messages.append(message)
                if progress:
                    progress(message, index, len(paths))
                continue

            if not faces:
                images_without_faces += 1
                message = f"{path_obj.name}: no faces"
                messages.append(message)
                if progress:
                    progress(message, index, len(paths))
                continue

            for face_index, face in enumerate(faces, start=1):
                if len(faces) > 1:
                    face.file_name = f"{path_obj.name} #{face_index}"
                if face.quality_score < min_quality:
                    low_quality_faces += 1
                    messages.append(f"{face.file_name}: skipped low quality ({face.quality_score:.3f})")
                    continue
                record = analysis_to_database_record(face)
                record["image_hash"] = image_hash
                record["review_state"] = "duplicate" if is_duplicate else "open"
                if is_duplicate:
                    record["notes"] = "Same source image hash already exists in this database or import batch."
                row_id = insert_face(conn, record)
                pending_writes += 1
                faces_saved += 1
                quality_total += face.quality_score
                duplicate_note = " duplicate" if is_duplicate else ""
                messages.append(f"{face.file_name}: saved face {row_id}{duplicate_note}")
            if pending_writes >= max(1, int(commit_interval)):
                conn.commit()
                pending_writes = 0
            if progress:
                progress(f"Processed {len(faces)} face(s) from {path_obj.name}", index, len(paths))
        if pending_writes:
            conn.commit()
    finally:
        conn.close()

    clear_search_index_cache(normalized)
    average_quality = quality_total / faces_saved if faces_saved else 0.0
    return ImportSummary(
        database_path=normalized,
        images_total=len(paths),
        faces_saved=faces_saved,
        images_without_faces=images_without_faces,
        failed_images=failed_images,
        low_quality_faces=low_quality_faces,
        duplicate_images=duplicate_images,
        average_quality=average_quality,
        messages=messages,
    )


def collect_image_paths(
    roots: Iterable[str | Path],
    *,
    recursive: bool = True,
) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for root in roots:
        path = Path(root)
        candidates = path.rglob("*") if path.is_dir() and recursive else path.glob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = str(candidate.resolve())
            key = resolved.lower()
            if key in seen:
                continue
            seen.add(key)
            images.append(resolved)
    images.sort(key=lambda value: value.lower())
    return images


def load_records(
    database_path: str | Path,
    *,
    include_ignored: bool = True,
    person_filter: str = "",
    tag_filter: str = "",
    review_filter: str = "",
    min_quality: float | None = None,
    text_filter: str = "",
    include_preview: bool = True,
    limit: int | None = None,
) -> tuple[list[FaceRecord], dict[str, str]]:
    return load_face_records(
        database_path,
        include_ignored=include_ignored,
        person_filter=person_filter,
        tag_filter=tag_filter,
        review_filter=review_filter,
        min_quality=min_quality,
        text_filter=text_filter,
        include_preview=include_preview,
        limit=limit,
    )


def load_database_statistics(database_path: str | Path) -> dict[str, object]:
    return database_statistics(database_path)


def check_database_integrity(database_path: str | Path) -> MaintenanceResult:
    conn = connect_database(database_path)
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        messages = [str(row[0]) for row in rows]
    finally:
        conn.close()
    ok = messages == ["ok"]
    detail = "; ".join(messages[:20]) if messages else "No integrity result returned."
    if len(messages) > 20:
        detail = f"{detail}; ... {len(messages) - 20} more"
    return MaintenanceResult("integrity_check", ok, detail)


def backup_database(database_path: str | Path, output_path: str | Path | None = None) -> MaintenanceResult:
    source = Path(normalize_database_path(database_path)).resolve()
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = source.with_name(f"{source.stem}_backup_{timestamp}{source.suffix}")
    else:
        output = Path(normalize_database_path(output_path)).resolve()
    if output == source:
        raise FaceEngineError("Backup path must differ from the source database.")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_conn = connect_database(source)
    try:
        destination_conn = sqlite3.connect(str(output))
        try:
            source_conn.backup(destination_conn)
        finally:
            destination_conn.close()
    finally:
        source_conn.close()
    return MaintenanceResult("backup", True, f"Backup created: {output}", str(output))


def checkpoint_database(database_path: str | Path, *, truncate: bool = True) -> MaintenanceResult:
    conn = connect_database(database_path)
    try:
        mode = "TRUNCATE" if truncate else "PASSIVE"
        row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
    finally:
        conn.close()
        clear_search_index_cache(database_path)
    if row is None:
        return MaintenanceResult("wal_checkpoint", False, "No checkpoint result returned.")
    return MaintenanceResult(
        "wal_checkpoint",
        True,
        f"busy={int(row[0])} log={int(row[1])} checkpointed={int(row[2])}",
    )


def vacuum_database(database_path: str | Path) -> MaintenanceResult:
    conn = connect_database(database_path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
        clear_search_index_cache(database_path)
    return MaintenanceResult("vacuum", True, "VACUUM completed.")


def load_people(database_path: str | Path) -> list[PersonRecord]:
    return db_list_people(database_path)


def load_person_summaries(
    database_path: str | Path,
    *,
    text_filter: str = "",
    limit: int | None = None,
) -> list[PersonSummary]:
    conn = connect_database(database_path)
    try:
        create_identity_profile_schema(conn)
        params: list[object] = []
        where_sql = ""
        if text_filter.strip():
            where_sql = "WHERE p.name LIKE ? COLLATE NOCASE OR p.notes LIKE ? COLLATE NOCASE"
            pattern = f"%{text_filter.strip()}%"
            params.extend([pattern, pattern])
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            params.append(max(0, int(limit)))
        rows = conn.execute(
            f"""
            SELECT
                p.id,
                p.name,
                p.notes,
                COUNT(f.id) AS face_count,
                AVG(f.quality_score) AS average_quality,
                SUM(CASE WHEN f.ignored = 1 THEN 1 ELSE 0 END) AS ignored_count,
                SUM(CASE WHEN f.review_state != 'reviewed' OR f.ignored = 1 THEN 1 ELSE 0 END) AS review_count,
                profile.status AS identity_status,
                profile.sample_count AS identity_sample_count,
                profile.prototype_count AS identity_prototype_count,
                profile.accept_threshold AS identity_accept_threshold,
                profile.updated_at AS identity_updated_at,
                profile.strategy_version AS identity_strategy_version,
                profile.scoring_model_version AS identity_scoring_model_version,
                profile.calibration_json AS identity_calibration_json,
                (
                    SELECT f2.id
                    FROM faces f2
                    WHERE f2.person_id = p.id
                    ORDER BY f2.quality_score DESC, f2.id ASC
                    LIMIT 1
                ) AS representative_face_id
            FROM persons p
            LEFT JOIN faces f ON f.person_id = p.id
            LEFT JOIN person_identity_profiles profile ON profile.person_id = p.id
            {where_sql}
            GROUP BY p.id, p.name, p.notes, profile.status, profile.sample_count, profile.prototype_count,
                     profile.accept_threshold, profile.updated_at, profile.strategy_version,
                     profile.scoring_model_version, profile.calibration_json
            ORDER BY face_count DESC, p.name COLLATE NOCASE
            {limit_sql}
            """,
            params,
        ).fetchall()
        summaries: list[PersonSummary] = []
        for row in rows:
            calibration = _json_dict(str(row["identity_calibration_json"] or "{}"))
            summaries.append(
                PersonSummary(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    notes=str(row["notes"] or ""),
                    face_count=int(row["face_count"] or 0),
                    average_quality=float(row["average_quality"] or 0.0),
                    ignored_count=int(row["ignored_count"] or 0),
                    review_count=int(row["review_count"] or 0),
                    representative_face_id=(
                        int(row["representative_face_id"]) if row["representative_face_id"] is not None else None
                    ),
                    identity_status=str(row["identity_status"] or ""),
                    identity_sample_count=int(row["identity_sample_count"] or 0),
                    identity_prototype_count=int(row["identity_prototype_count"] or 0),
                    identity_accept_threshold=float(row["identity_accept_threshold"] or 0.0),
                    identity_updated_at=str(row["identity_updated_at"] or ""),
                    identity_strategy_version=str(row["identity_strategy_version"] or ""),
                    identity_scoring_model_version=str(row["identity_scoring_model_version"] or ""),
                    identity_health=str(calibration.get("health") or ""),
                )
            )
        return summaries
    finally:
        conn.close()


def load_tags(database_path: str | Path) -> list[TagRecord]:
    return db_list_tags(database_path)


def load_tag_summaries(database_path: str | Path, *, limit: int = 12) -> list[TagSummary]:
    conn = connect_database(database_path)
    try:
        rows = conn.execute(
            """
            SELECT
                t.id,
                t.name,
                COUNT(ft.face_id) AS face_count
            FROM tags t
            LEFT JOIN face_tags ft ON ft.tag_id = t.id
            GROUP BY t.id, t.name
            ORDER BY face_count DESC, t.name COLLATE NOCASE
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [
            TagSummary(
                id=int(row["id"]),
                name=str(row["name"]),
                face_count=int(row["face_count"] or 0),
            )
            for row in rows
        ]
    finally:
        conn.close()


def rename_person(database_path: str | Path, person_id: int, new_name: str, notes: str = "") -> None:
    clean_name = new_name.strip()
    if not clean_name:
        raise FaceEngineError("Person name cannot be empty.")
    conn = connect_database(database_path)
    try:
        conn.execute(
            "UPDATE persons SET name = ?, notes = ? WHERE id = ?",
            (clean_name, notes, int(person_id)),
        )
        conn.commit()
    finally:
        conn.close()
    clear_search_index_cache(database_path)


def merge_people(database_path: str | Path, source_person_id: int, target_person_id: int) -> int:
    source_id = int(source_person_id)
    target_id = int(target_person_id)
    if source_id == target_id:
        return 0
    conn = connect_database(database_path)
    try:
        source = conn.execute("SELECT id FROM persons WHERE id = ?", (source_id,)).fetchone()
        target = conn.execute("SELECT id FROM persons WHERE id = ?", (target_id,)).fetchone()
        if source is None or target is None:
            raise FaceEngineError("Source or target person does not exist.")
        cursor = conn.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (target_id, source_id))
        moved = int(cursor.rowcount if cursor.rowcount is not None else 0)
        conn.execute("DELETE FROM persons WHERE id = ?", (source_id,))
        conn.commit()
    finally:
        conn.close()
    clear_search_index_cache(database_path)
    return moved


def clear_person_assignment(database_path: str | Path, person_id: int, *, delete_person: bool = True) -> int:
    conn = connect_database(database_path)
    try:
        cursor = conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (int(person_id),))
        cleared = int(cursor.rowcount if cursor.rowcount is not None else 0)
        if delete_person:
            conn.execute("DELETE FROM persons WHERE id = ?", (int(person_id),))
        conn.commit()
    finally:
        conn.close()
    clear_search_index_cache(database_path)
    return cleared


def assign_person(database_path: str | Path, face_id: int, person_name: str) -> None:
    db_assign_person(database_path, face_id, person_name)
    clear_search_index_cache(database_path)


def set_tags(database_path: str | Path, face_id: int, tag_text: str) -> None:
    db_set_face_tags(database_path, face_id, parse_tag_text(tag_text))
    clear_search_index_cache(database_path)


def update_review(
    database_path: str | Path,
    face_id: int,
    *,
    ignored: bool | None = None,
    review_state: str | None = None,
    notes: str | None = None,
) -> None:
    db_update_face_review(database_path, face_id, ignored=ignored, review_state=review_state, notes=notes)
    clear_search_index_cache(database_path)


def assign_faces_to_person(
    database_path: str | Path,
    face_ids: Iterable[int],
    person_name: str,
    *,
    tag_text: str = "",
    mark_reviewed: bool = True,
) -> int:
    ids = [int(face_id) for face_id in face_ids]
    clean_name = person_name.strip()
    if not ids or not clean_name:
        return 0
    conn = connect_database(database_path)
    try:
        person_id = ensure_person(conn, clean_name)
        tags = parse_tag_text(tag_text)
        for face_id in ids:
            conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
            if tags:
                set_face_tags_on_connection(conn, face_id, tags)
            if mark_reviewed:
                conn.execute("UPDATE faces SET review_state = 'reviewed', ignored = 0 WHERE id = ?", (face_id,))
        conn.commit()
    finally:
        conn.close()
    clear_search_index_cache(database_path)
    return len(ids)


def update_faces_metadata(
    database_path: str | Path,
    face_ids: Iterable[int],
    *,
    person_name: str | None = None,
    tag_text: str | None = None,
    append_tags: bool = False,
    ignored: bool | None = None,
    review_state: str | None = None,
    notes: str | None = None,
) -> int:
    ids = [int(face_id) for face_id in face_ids]
    if not ids:
        return 0

    conn = connect_database(database_path)
    try:
        person_id: int | None = None
        if person_name is not None:
            clean_name = person_name.strip()
            person_id = ensure_person(conn, clean_name) if clean_name else None

        tags = parse_tag_text(tag_text or "") if tag_text is not None else []
        for face_id in ids:
            fields: list[str] = []
            values: list[object] = []
            if person_name is not None:
                fields.append("person_id = ?")
                values.append(person_id)
            if ignored is not None:
                fields.append("ignored = ?")
                values.append(1 if ignored else 0)
            if review_state is not None:
                fields.append("review_state = ?")
                values.append(review_state)
            if notes is not None:
                fields.append("notes = ?")
                values.append(notes)
            if fields:
                values.append(face_id)
                conn.execute(f"UPDATE faces SET {', '.join(fields)} WHERE id = ?", values)
            if tag_text is not None:
                if append_tags:
                    existing = [
                        str(row["name"])
                        for row in conn.execute(
                            """
                            SELECT t.name
                            FROM face_tags ft
                            JOIN tags t ON t.id = ft.tag_id
                            WHERE ft.face_id = ?
                            ORDER BY t.name COLLATE NOCASE
                            """,
                            (face_id,),
                        ).fetchall()
                    ]
                    merged = existing + [tag for tag in tags if tag.lower() not in {item.lower() for item in existing}]
                    set_face_tags_on_connection(conn, face_id, merged)
                else:
                    set_face_tags_on_connection(conn, face_id, tags)
        conn.commit()
    finally:
        conn.close()
    clear_search_index_cache(database_path)
    return len(ids)


def load_review_queue(
    database_path: str | Path,
    *,
    limit: int = 500,
    text_filter: str = "",
) -> tuple[list[FaceRecord], dict[str, str]]:
    return load_review_records(database_path, limit=limit, text_filter=text_filter)


def analyze_primary_face(image_path: str | Path, *, prefer_gpu: bool = True) -> PrimaryFace:
    faces = get_engine(prefer_gpu=prefer_gpu).extract_faces_from_path(image_path)
    if not faces:
        raise FaceEngineError("No face was detected in the selected image.")
    return PrimaryFace(face=faces[0], detected_count=len(faces))


def analyze_faces(image_path: str | Path, *, prefer_gpu: bool = True) -> list[AnalyzedFace]:
    faces = get_engine(prefer_gpu=prefer_gpu).extract_faces_from_path(image_path)
    if not faces:
        raise FaceEngineError("No face was detected in the selected image.")
    return faces


def load_or_build_landmarks3d(database_path: str | Path, record: FaceRecord) -> list | None:
    if record.landmarks3d:
        return record.landmarks3d
    if not record.source_path:
        return None
    source = Path(record.source_path)
    if not source.exists():
        return None

    faces = get_engine(prefer_gpu=True).extract_faces_from_path(source)
    if not faces:
        return None
    selected = _best_matching_analysis_for_record(record, faces)
    if not selected.landmarks3d:
        return None
    db_update_face_landmarks3d(database_path, record.id, selected.landmarks3d)
    return selected.landmarks3d


def load_or_build_face_mesh3d(database_path: str | Path, record: FaceRecord) -> list | str | None:
    if record.face_mesh3d:
        return record.face_mesh3d
    if not record.source_path:
        return None
    source = Path(record.source_path)
    if not source.exists():
        return None

    try:
        image_bgr = read_image_bgr(source)
        meshes = _extract_mediapipe_face_meshes(image_bgr)
    except ImportError:
        return 'Install the optional "dense-mesh" extra to enable MediaPipe Face Mesh.'
    except Exception as exc:
        return f"MediaPipe Face Mesh failed: {exc}"
    if not meshes:
        return None

    selected = _best_matching_mesh_for_record(record, meshes)
    db_update_face_mesh3d(database_path, record.id, selected)
    return selected


def rebuild_identity_profiles(
    database_path: str | Path,
    person_ids: Iterable[int] | None = None,
    *,
    min_quality: float = IDENTITY_DEFAULT_MIN_QUALITY,
    max_prototypes: int = IDENTITY_MAX_PROTOTYPES,
) -> IdentityTrainingSummary:
    return rebuild_identity_gallery_profiles(
        database_path,
        person_ids=person_ids,
        min_quality=min_quality,
        max_exemplars=max(IDENTITY_MAX_EXEMPLARS, int(max_prototypes)),
    )


def rebuild_identity_gallery_profiles(
    database_path: str | Path,
    person_ids: Iterable[int] | None = None,
    *,
    min_quality: float = IDENTITY_DEFAULT_MIN_QUALITY,
    max_exemplars: int = IDENTITY_MAX_EXEMPLARS,
) -> IdentityTrainingSummary:
    normalized = normalize_database_path(database_path)
    target_ids = [int(value) for value in person_ids] if person_ids is not None else None
    messages: list[str] = []
    profiles_built = 0
    weak_profiles = 0
    skipped_people = 0
    samples_used = 0

    conn = connect_database(normalized)
    try:
        create_identity_profile_schema(conn)
        params: list[object] = []
        person_where = ""
        if target_ids is not None:
            if not target_ids:
                return IdentityTrainingSummary(0, 0, 0, 0, ["No people selected."])
            placeholders = ",".join("?" for _ in target_ids)
            person_where = f"WHERE p.id IN ({placeholders})"
            params.extend(target_ids)

        people_rows = conn.execute(
            f"""
            SELECT p.id, p.name
            FROM persons p
            {person_where}
            ORDER BY p.name COLLATE NOCASE
            """,
            params,
        ).fetchall()
        if not people_rows:
            return IdentityTrainingSummary(0, 0, 0, 0, ["No named people are available for training."])

        face_params: list[object] = [float(min_quality)]
        face_where = "f.person_id IS NOT NULL AND f.ignored = 0 AND f.quality_score >= ?"
        if target_ids is not None:
            placeholders = ",".join("?" for _ in target_ids)
            face_where += f" AND f.person_id IN ({placeholders})"
            face_params.extend(target_ids)
        face_rows = conn.execute(
            f"""
            SELECT
                f.id,
                f.person_id,
                p.name AS person_name,
                f.embedding_blob,
                f.embedding_dim,
                f.quality_score
            FROM faces f
            JOIN persons p ON p.id = f.person_id
            WHERE {face_where}
            ORDER BY p.name COLLATE NOCASE, f.quality_score DESC, f.id ASC
            """,
            face_params,
        ).fetchall()

        rows_by_person: dict[int, list[sqlite3.Row]] = {}
        for row in face_rows:
            rows_by_person.setdefault(int(row["person_id"]), []).append(row)

        all_rows = list(face_rows)

        if target_ids is None:
            conn.execute("DELETE FROM person_identity_profiles")
        else:
            placeholders = ",".join("?" for _ in target_ids)
            conn.execute(f"DELETE FROM person_identity_profiles WHERE person_id IN ({placeholders})", target_ids)

        for person in people_rows:
            person_id = int(person["id"])
            person_name = str(person["name"])
            person_face_rows = rows_by_person.get(person_id, [])
            profile = _build_identity_profile_from_rows(
                person_id,
                person_name,
                person_face_rows,
                negative_rows=[row for row in all_rows if int(row["person_id"]) != person_id],
                max_exemplars=max_exemplars,
            )
            if profile is None:
                skipped_people += 1
                messages.append(f"{person_name}: skipped, no usable assigned faces above quality {min_quality:.2f}.")
                continue
            _save_identity_profile_on_connection(conn, profile)
            profiles_built += 1
            samples_used += profile.sample_count
            if profile.status == "weak":
                weak_profiles += 1
            messages.append(
                f"{person_name}: {profile.sample_count} sample(s), {profile.prototype_count} exemplar(s), "
                f"{profile.status}, health {profile.calibration.get('health', 'unknown')}, "
                f"strict accept {profile.accept_threshold:.3f}."
            )
        conn.commit()
    finally:
        conn.close()

    clear_identity_profile_cache(normalized)
    return IdentityTrainingSummary(
        profiles_built=profiles_built,
        weak_profiles=weak_profiles,
        skipped_people=skipped_people,
        samples_used=samples_used,
        messages=messages,
    )


def identify_person(
    database_path: str | Path,
    query_embedding: np.ndarray,
    *,
    top_k: int = 5,
    identity_mode: str = "strict",
) -> IdentityResult:
    mode = _normalize_identity_mode(identity_mode)
    index = get_identity_profile_index(database_path)
    if not index.profiles:
        return IdentityResult("unknown", [], "No trained identity profiles.")
    query = _normalize_vector(np.asarray(query_embedding, dtype=np.float32).reshape(-1))
    if query.size == 0:
        return IdentityResult("unknown", [], "Query embedding is empty.")

    scored: list[tuple[IdentityProfile, float, int | None]] = []
    for profile in index.profiles:
        exemplars = _profile_exemplars(profile)
        if profile.embedding_dim != query.size or exemplars.size == 0:
            continue
        score, evidence_face_id = _score_identity_profile(profile, query)
        scored.append((profile, score, evidence_face_id))

    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored:
        return IdentityResult("unknown", [], "No identity profile matches the query embedding dimensions.")

    candidates: list[IdentityCandidate] = []
    limit = max(1, int(top_k))
    for position, (profile, score, evidence_face_id) in enumerate(scored[:limit]):
        next_score = float(scored[position + 1][1]) if position + 1 < len(scored) else profile.review_threshold
        margin = float(score - next_score)
        candidates.append(
            IdentityCandidate(
                profile=profile,
                score=float(score),
                margin=margin,
                confidence=_identity_confidence(profile, float(score), margin, mode),
                evidence_face_id=evidence_face_id,
                scoring_model_version=profile.scoring_model_version,
                mode=mode,
            )
        )

    best = candidates[0]
    thresholds = _profile_thresholds_for_mode(best.profile, mode)
    if best.score < thresholds["review"]:
        return IdentityResult("unknown", candidates, "No person passes the review threshold.")
    if best.profile.status == "weak":
        return IdentityResult("review", candidates, "Best profile is weak; manual confirmation is required.")
    if best.score >= thresholds["accept"] and best.margin >= thresholds["margin"]:
        return IdentityResult("confirmed", candidates, f"Identity confirmed by {mode} gallery profile.")
    return IdentityResult("review", candidates, "Identity is plausible but needs review.")


def compare_images(
    image_a: str | Path,
    image_b: str | Path,
    *,
    prefer_gpu: bool = True,
) -> CompareResult:
    face_a = analyze_primary_face(image_a, prefer_gpu=prefer_gpu)
    face_b = analyze_primary_face(image_b, prefer_gpu=prefer_gpu)
    cosine = cosine_similarity(face_a.face.embedding, face_b.face.embedding)
    return CompareResult(
        face_a=face_a,
        face_b=face_b,
        cosine=cosine,
        similarity=similarity_percent(cosine),
    )


def search_database(
    database_path: str | Path,
    query_embedding: np.ndarray,
    *,
    top_k: int = 30,
    threshold: float = -1.0,
    person_filter: str = "",
    tag_filter: str = "",
    min_quality: float = 0.0,
    include_ignored: bool = False,
) -> tuple[list[SearchHit], dict[str, str]]:
    index = get_search_index(database_path)
    if not index.records:
        return [], index.metadata

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if index.matrix.shape[1] != query.size:
        raise FaceEngineError(f"Embedding dimensions differ: database {index.matrix.shape[1]} vs query {query.size}.")

    candidate_indexes = np.fromiter(
        (
            position
            for position, record in enumerate(index.records)
            if _record_matches_search_filters(
                record,
                include_ignored=include_ignored,
                person_filter=person_filter,
                tag_filter=tag_filter,
                min_quality=min_quality,
            )
        ),
        dtype=np.int64,
    )
    if candidate_indexes.size == 0:
        return [], index.metadata

    scores = index.matrix[candidate_indexes] @ query
    passing_positions = np.flatnonzero(scores >= threshold)
    if passing_positions.size == 0:
        return [], index.metadata

    sorted_positions = passing_positions[np.argsort(scores[passing_positions])[::-1]]
    limit = max(1, int(top_k))
    hits = [
        SearchHit(
            record=index.records[int(candidate_indexes[int(position)])],
            cosine=float(scores[int(position)]),
            similarity=similarity_percent(float(scores[int(position)])),
        )
        for position in sorted_positions[:limit]
    ]
    return hits, index.metadata


def search_database_progressive(
    database_path: str | Path,
    query_embedding: np.ndarray,
    *,
    top_k: int = 30,
    threshold: float = -1.0,
    person_filter: str = "",
    tag_filter: str = "",
    min_quality: float = 0.0,
    include_ignored: bool = False,
    progress: ProgressCallback | None = None,
    preview_interval: float = 0.08,
) -> tuple[list[SearchHit], dict[str, str]]:
    index = get_search_index(database_path)
    if not index.records:
        return [], index.metadata
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    candidates = [
        (position, record)
        for position, record in enumerate(index.records)
        if _record_matches_search_filters(
            record,
            include_ignored=include_ignored,
            person_filter=person_filter,
            tag_filter=tag_filter,
            min_quality=min_quality,
        )
    ]
    if not candidates:
        return [], index.metadata
    if index.matrix.shape[1] != query.size:
        raise FaceEngineError(f"Embedding dimensions differ: database {index.matrix.shape[1]} vs query {query.size}.")

    hits: list[SearchHit] = []
    total = len(candidates)
    last_preview_at = 0.0
    interval = max(0.0, float(preview_interval))
    for current, (position, record) in enumerate(candidates, start=1):
        if progress:
            now = time.monotonic()
            if current == 1 or current == total or interval <= 0.0 or now - last_preview_at >= interval:
                progress(f"FACE_PREVIEW|{record.id}|{record.file_name}", current, total)
                last_preview_at = now
        cosine = float(np.dot(query, index.matrix[int(position)]))
        if cosine >= threshold:
            hits.append(SearchHit(record=record, cosine=cosine, similarity=similarity_percent(cosine)))
    hits.sort(key=lambda hit: hit.cosine, reverse=True)
    return hits[: max(1, int(top_k))], index.metadata


def build_face_clusters(
    database_path: str | Path,
    *,
    threshold: float = 0.55,
    min_cluster_size: int = 2,
    include_ignored: bool = False,
    unassigned_only: bool = False,
    min_quality: float = 0.0,
    max_faces: int = 5000,
    block_size: int = 512,
) -> tuple[list[FaceCluster], dict[str, str]]:
    index = get_search_index(database_path)
    if not index.records:
        return [], index.metadata

    candidates = [
        (position, record)
        for position, record in enumerate(index.records)
        if _record_matches_search_filters(
            record,
            include_ignored=include_ignored,
            person_filter="",
            tag_filter="",
            min_quality=min_quality,
        )
        and (not unassigned_only or not record.person_id)
    ]
    if len(candidates) < min_cluster_size:
        return [], index.metadata

    candidates = candidates[: max(1, int(max_faces))]
    source_positions = np.asarray([position for position, _ in candidates], dtype=np.int64)
    records = [record for _, record in candidates]
    matrix = index.matrix[source_positions].astype(np.float32, copy=False)
    n_faces = matrix.shape[0]
    parents = list(range(n_faces))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    threshold = float(threshold)
    block_size = max(64, int(block_size))
    for start in range(0, n_faces, block_size):
        end = min(n_faces, start + block_size)
        scores = matrix[start:end] @ matrix.T
        for local_row in range(end - start):
            row_index = start + local_row
            matches = np.flatnonzero(scores[local_row] >= threshold)
            for match_index in matches:
                other_index = int(match_index)
                if other_index > row_index:
                    union(row_index, other_index)

    groups: dict[int, list[int]] = {}
    for index_position in range(n_faces):
        groups.setdefault(find(index_position), []).append(index_position)

    clusters: list[FaceCluster] = []
    for group_indexes in groups.values():
        if len(group_indexes) < min_cluster_size:
            continue
        group_matrix = matrix[group_indexes]
        group_records = [records[index_position] for index_position in group_indexes]
        similarity_matrix = group_matrix @ group_matrix.T
        upper = similarity_matrix[np.triu_indices(len(group_indexes), k=1)]
        mean_similarity = float(upper.mean()) if upper.size else 1.0
        max_similarity = float(upper.max()) if upper.size else 1.0
        centroid = group_matrix.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm:
            centroid = centroid / norm
        representative_offset = int(np.argmax(group_matrix @ centroid))
        representative = group_records[representative_offset]
        existing_people = sorted({record.person_name for record in group_records if record.person_name})
        suggested_name = existing_people[0] if len(existing_people) == 1 else ""
        clusters.append(
            FaceCluster(
                cluster_id=0,
                records=group_records,
                mean_similarity=mean_similarity,
                max_similarity=max_similarity,
                average_quality=float(np.mean([record.quality_score for record in group_records])),
                representative_id=representative.id,
                suggested_name=suggested_name,
                existing_people=existing_people,
            )
        )

    clusters.sort(key=lambda cluster: (cluster.size, cluster.mean_similarity, cluster.average_quality), reverse=True)
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster.cluster_id = cluster_id
    return clusters, index.metadata


def get_search_index(database_path: str | Path) -> FaceSearchIndex:
    normalized = str(Path(normalize_database_path(database_path)).resolve())
    cache_key = database_cache_key(normalized)
    with _SEARCH_INDEX_LOCK:
        cached = _SEARCH_INDEX_CACHE.get(normalized)
        if cached and cached.cache_key == cache_key:
            return cached

    records, metadata = load_records(normalized, include_ignored=True, include_preview=False)
    cache_key = database_cache_key(normalized)
    if records:
        matrix = np.vstack([record.embedding for record in records]).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)
    index = FaceSearchIndex(
        database_path=normalized,
        cache_key=cache_key,
        records=records,
        metadata=metadata,
        matrix=matrix,
    )
    with _SEARCH_INDEX_LOCK:
        _SEARCH_INDEX_CACHE[normalized] = index
    return index


def get_identity_profile_index(database_path: str | Path) -> IdentityProfileIndex:
    normalized = str(Path(normalize_database_path(database_path)).resolve())
    cache_key = database_cache_key(normalized)
    with _IDENTITY_PROFILE_LOCK:
        cached = _IDENTITY_PROFILE_CACHE.get(normalized)
        if cached and cached.cache_key == cache_key:
            return cached

    profiles = load_identity_profiles(normalized)
    index = IdentityProfileIndex(database_path=normalized, cache_key=cache_key, profiles=profiles)
    with _IDENTITY_PROFILE_LOCK:
        _IDENTITY_PROFILE_CACHE[normalized] = index
    return index


def load_identity_profiles(database_path: str | Path) -> list[IdentityProfile]:
    conn = connect_database(database_path)
    try:
        create_identity_profile_schema(conn)
        rows = conn.execute(
            """
            SELECT
                profile.person_id,
                p.name AS person_name,
                profile.sample_count,
                profile.prototype_count,
                profile.embedding_dim,
                profile.centroid_blob,
                profile.prototypes_blob,
                profile.exemplar_blob,
                profile.exemplar_face_ids_json,
                profile.exemplar_weights_blob,
                profile.hard_negative_face_ids_json,
                profile.thresholds_json,
                profile.calibration_json,
                profile.strategy_version,
                profile.scoring_model_version,
                profile.accept_threshold,
                profile.review_threshold,
                profile.mean_similarity,
                profile.min_similarity,
                profile.max_similarity,
                profile.quality_mean,
                profile.evidence_face_ids_json,
                profile.status,
                profile.updated_at
            FROM person_identity_profiles profile
            JOIN persons p ON p.id = profile.person_id
            ORDER BY p.name COLLATE NOCASE
            """
        ).fetchall()
        profiles: list[IdentityProfile] = []
        for row in rows:
            profile = _identity_profile_from_row(row)
            if profile is not None:
                profiles.append(profile)
        _load_hard_negative_embeddings(conn, profiles)
        return profiles
    finally:
        conn.close()


def clear_search_index_cache(database_path: str | Path | None = None) -> None:
    with _SEARCH_INDEX_LOCK:
        if database_path is None:
            _SEARCH_INDEX_CACHE.clear()
        else:
            normalized = str(Path(normalize_database_path(database_path)).resolve())
            _SEARCH_INDEX_CACHE.pop(normalized, None)
    clear_identity_profile_cache(database_path)


def clear_identity_profile_cache(database_path: str | Path | None = None) -> None:
    with _IDENTITY_PROFILE_LOCK:
        if database_path is None:
            _IDENTITY_PROFILE_CACHE.clear()
            return
        normalized = str(Path(normalize_database_path(database_path)).resolve())
        _IDENTITY_PROFILE_CACHE.pop(normalized, None)


def database_cache_key(database_path: str | Path) -> tuple[tuple[str, int, int], ...]:
    base = Path(normalize_database_path(database_path)).resolve()
    paths = [base, Path(str(base) + "-wal"), Path(str(base) + "-shm")]
    parts: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            parts.append((str(path), -1, -1))
            continue
        parts.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(parts)


def load_preview(database_path: str | Path, face_id: int) -> bytes | None:
    return db_load_face_preview(database_path, face_id)


def _build_identity_profile_from_rows(
    person_id: int,
    person_name: str,
    rows: list[sqlite3.Row],
    *,
    negative_rows: list[sqlite3.Row],
    max_exemplars: int,
) -> IdentityProfile | None:
    vectors: list[np.ndarray] = []
    face_ids: list[int] = []
    qualities: list[float] = []
    dim: int | None = None
    for row in rows:
        row_dim = int(row["embedding_dim"])
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        if embedding.size != row_dim:
            continue
        if dim is None:
            dim = row_dim
        if row_dim != dim:
            continue
        vectors.append(_normalize_vector(embedding.astype(np.float32, copy=True)))
        face_ids.append(int(row["id"]))
        qualities.append(float(row["quality_score"] or 0.0))
    if not vectors or dim is None:
        return None

    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    qualities_array = np.asarray(qualities, dtype=np.float32)
    weights = np.clip(qualities_array, 0.10, 1.0)
    centroid = _normalize_vector(np.average(matrix, axis=0, weights=weights))

    selected_positions = _select_identity_exemplar_positions(
        matrix,
        qualities_array,
        max_exemplars=max_exemplars,
    )
    exemplars = matrix[selected_positions].astype(np.float32, copy=True)
    exemplar_face_ids = [face_ids[position] for position in selected_positions]
    exemplar_weights = _exemplar_weights(matrix, qualities_array, selected_positions)

    pairwise_scores = _upper_pairwise_scores(matrix)
    if pairwise_scores.size:
        mean_similarity = float(pairwise_scores.mean())
        min_similarity = float(pairwise_scores.min())
        max_similarity = float(pairwise_scores.max())
        score_std = float(pairwise_scores.std())
    else:
        mean_similarity = min_similarity = max_similarity = 1.0
        score_std = 0.0

    negative_matrix, negative_face_ids = _rows_to_embedding_matrix(negative_rows, expected_dim=dim)
    negative_scores = _score_matrix_against_profile(
        centroid,
        exemplars,
        exemplar_weights,
        negative_matrix,
        hard_negative_embeddings=None,
    )
    hard_negative_face_ids = _select_hard_negative_face_ids(negative_scores, negative_face_ids)
    hard_negative_embeddings = _hard_negative_embeddings(negative_scores, negative_matrix)
    positive_scores = _leave_one_out_profile_scores(matrix, qualities_array)
    thresholds, calibration = _calibrate_identity_thresholds(
        sample_count=len(vectors),
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        mean_similarity=mean_similarity,
        score_std=score_std,
    )
    sample_count = len(vectors)
    status = "strong" if sample_count >= IDENTITY_MIN_STRONG_SAMPLES else "weak"
    calibration["health"] = _identity_profile_health(sample_count, positive_scores, negative_scores, status)
    strict_thresholds = thresholds["strict"]

    return IdentityProfile(
        person_id=int(person_id),
        person_name=person_name,
        sample_count=sample_count,
        prototype_count=int(exemplars.shape[0]),
        embedding_dim=int(dim),
        centroid=centroid.astype(np.float32, copy=False),
        prototypes=exemplars,
        exemplars=exemplars,
        exemplar_weights=exemplar_weights,
        accept_threshold=float(strict_thresholds["accept"]),
        review_threshold=float(strict_thresholds["review"]),
        mean_similarity=mean_similarity,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        quality_mean=float(qualities_array.mean()) if qualities_array.size else 0.0,
        evidence_face_ids=exemplar_face_ids,
        exemplar_face_ids=exemplar_face_ids,
        hard_negative_face_ids=hard_negative_face_ids,
        hard_negative_embeddings=hard_negative_embeddings,
        thresholds=thresholds,
        calibration=calibration,
        status=status,
        strategy_version=IDENTITY_GALLERY_STRATEGY_VERSION,
        scoring_model_version=_active_identity_scorer_version(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _save_identity_profile_on_connection(conn: sqlite3.Connection, profile: IdentityProfile) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO person_identity_profiles (
            person_id,
            sample_count,
            prototype_count,
            embedding_dim,
            centroid_blob,
            prototypes_blob,
            exemplar_blob,
            exemplar_face_ids_json,
            exemplar_weights_blob,
            hard_negative_face_ids_json,
            thresholds_json,
            calibration_json,
            strategy_version,
            scoring_model_version,
            accept_threshold,
            review_threshold,
            mean_similarity,
            min_similarity,
            max_similarity,
            quality_mean,
            evidence_face_ids_json,
            status,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile.person_id,
            profile.sample_count,
            profile.prototype_count,
            profile.embedding_dim,
            profile.centroid.astype(np.float32, copy=False).tobytes(),
            profile.prototypes.astype(np.float32, copy=False).reshape(-1).tobytes(),
            profile.exemplars.astype(np.float32, copy=False).reshape(-1).tobytes(),
            json.dumps(profile.exemplar_face_ids, separators=(",", ":")),
            profile.exemplar_weights.astype(np.float32, copy=False).reshape(-1).tobytes(),
            json.dumps(profile.hard_negative_face_ids, separators=(",", ":")),
            json.dumps(profile.thresholds, separators=(",", ":")),
            json.dumps(profile.calibration, separators=(",", ":")),
            profile.strategy_version,
            profile.scoring_model_version,
            profile.accept_threshold,
            profile.review_threshold,
            profile.mean_similarity,
            profile.min_similarity,
            profile.max_similarity,
            profile.quality_mean,
            json.dumps(profile.evidence_face_ids, separators=(",", ":")),
            profile.status,
            profile.updated_at,
        ),
    )


def _identity_profile_from_row(row: sqlite3.Row) -> IdentityProfile | None:
    dim = int(row["embedding_dim"])
    prototype_count = int(row["prototype_count"])
    centroid = np.frombuffer(row["centroid_blob"], dtype=np.float32)
    prototypes = np.frombuffer(row["prototypes_blob"], dtype=np.float32)
    if dim <= 0 or prototype_count <= 0 or centroid.size != dim or prototypes.size != prototype_count * dim:
        return None
    try:
        evidence_face_ids = [int(value) for value in json.loads(str(row["evidence_face_ids_json"] or "[]"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence_face_ids = []
    exemplar_blob = row["exemplar_blob"] if "exemplar_blob" in row.keys() else None
    exemplar_weights_blob = row["exemplar_weights_blob"] if "exemplar_weights_blob" in row.keys() else None
    exemplar_face_ids = _json_int_list(row["exemplar_face_ids_json"] if "exemplar_face_ids_json" in row.keys() else "[]")
    hard_negative_face_ids = _json_int_list(
        row["hard_negative_face_ids_json"] if "hard_negative_face_ids_json" in row.keys() else "[]"
    )
    exemplars = np.frombuffer(exemplar_blob, dtype=np.float32) if exemplar_blob else prototypes
    if exemplars.size % dim != 0 or exemplars.size == 0:
        exemplars = prototypes
    exemplar_count = int(exemplars.size // dim)
    weights = np.frombuffer(exemplar_weights_blob, dtype=np.float32) if exemplar_weights_blob else np.empty((0,), dtype=np.float32)
    if weights.size != exemplar_count:
        weights = np.ones((exemplar_count,), dtype=np.float32)
    if not exemplar_face_ids:
        exemplar_face_ids = list(evidence_face_ids)
    thresholds = _normalize_thresholds(_json_dict(row["thresholds_json"] if "thresholds_json" in row.keys() else "{}"))
    calibration = _json_dict(row["calibration_json"] if "calibration_json" in row.keys() else "{}")
    return IdentityProfile(
        person_id=int(row["person_id"]),
        person_name=str(row["person_name"]),
        sample_count=int(row["sample_count"]),
        prototype_count=exemplar_count,
        embedding_dim=dim,
        centroid=centroid.astype(np.float32, copy=True),
        prototypes=prototypes.astype(np.float32, copy=True).reshape(prototype_count, dim),
        exemplars=exemplars.astype(np.float32, copy=True).reshape(exemplar_count, dim),
        exemplar_weights=weights.astype(np.float32, copy=True),
        accept_threshold=float(row["accept_threshold"]),
        review_threshold=float(row["review_threshold"]),
        mean_similarity=float(row["mean_similarity"]),
        min_similarity=float(row["min_similarity"]),
        max_similarity=float(row["max_similarity"]),
        quality_mean=float(row["quality_mean"]),
        evidence_face_ids=evidence_face_ids,
        exemplar_face_ids=exemplar_face_ids,
        hard_negative_face_ids=hard_negative_face_ids,
        hard_negative_embeddings=None,
        thresholds=thresholds,
        calibration=calibration,
        status=str(row["status"] or "weak"),
        strategy_version=str(row["strategy_version"] if "strategy_version" in row.keys() else "legacy_v1"),
        scoring_model_version=str(
            row["scoring_model_version"] if "scoring_model_version" in row.keys() else IDENTITY_NUMPY_SCORER_VERSION
        ),
        updated_at=str(row["updated_at"] or ""),
    )


def _load_hard_negative_embeddings(conn: sqlite3.Connection, profiles: list[IdentityProfile]) -> None:
    needed_ids = sorted({face_id for profile in profiles for face_id in profile.hard_negative_face_ids})
    if not needed_ids:
        return
    placeholders = ",".join("?" for _ in needed_ids)
    rows = conn.execute(
        f"SELECT id, embedding_blob, embedding_dim FROM faces WHERE id IN ({placeholders})",
        needed_ids,
    ).fetchall()
    by_id: dict[int, np.ndarray] = {}
    for row in rows:
        dim = int(row["embedding_dim"])
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        if dim > 0 and embedding.size == dim:
            by_id[int(row["id"])] = _normalize_vector(embedding.astype(np.float32, copy=True))
    for profile in profiles:
        vectors = [by_id[face_id] for face_id in profile.hard_negative_face_ids if face_id in by_id]
        if vectors:
            profile.hard_negative_embeddings = np.vstack(vectors).astype(np.float32, copy=False)


def _json_int_list(value: object) -> list[int]:
    try:
        return [int(item) for item in json.loads(str(value or "[]"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _json_dict(value: object) -> dict[str, object]:
    try:
        data = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_identity_mode(identity_mode: str) -> str:
    mode = str(identity_mode or "strict").strip().lower()
    return mode if mode in IDENTITY_MODE_DEFAULTS else "strict"


def _normalize_thresholds(raw: dict[str, object]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for mode, defaults in IDENTITY_MODE_DEFAULTS.items():
        item = raw.get(mode)
        source = item if isinstance(item, dict) else {}
        thresholds[mode] = {
            "accept": float(source.get("accept", defaults["accept"])),
            "review": float(source.get("review", defaults["review"])),
            "margin": float(source.get("margin", defaults["margin"])),
        }
    return thresholds


def _profile_thresholds_for_mode(profile: IdentityProfile, mode: str) -> dict[str, float]:
    return _normalize_thresholds(profile.thresholds).get(_normalize_identity_mode(mode), IDENTITY_MODE_DEFAULTS["strict"])


def _profile_exemplars(profile: IdentityProfile) -> np.ndarray:
    if profile.exemplars.size:
        return profile.exemplars
    return profile.prototypes


def _select_identity_exemplar_positions(
    matrix: np.ndarray,
    qualities: np.ndarray,
    *,
    max_exemplars: int,
) -> list[int]:
    if matrix.shape[0] == 0:
        return []
    consistency = matrix @ _normalize_vector(matrix.mean(axis=0))
    if matrix.shape[0] > 1:
        pairwise = matrix @ matrix.T
        np.fill_diagonal(pairwise, np.nan)
        with np.errstate(invalid="ignore"):
            consistency = np.nanmean(pairwise, axis=1)
        consistency = np.nan_to_num(consistency, nan=1.0)
    quality_norm = np.clip(qualities, 0.0, 1.0)
    order_score = quality_norm * 0.62 + consistency * 0.38
    order = np.lexsort((np.arange(matrix.shape[0]), -order_score))
    selected: list[int] = []
    limit = max(1, int(max_exemplars))
    for position in order:
        index = int(position)
        if not selected:
            selected.append(index)
        else:
            similarities = matrix[selected] @ matrix[index]
            if float(similarities.max()) < IDENTITY_EXEMPLAR_DIVERSITY or len(selected) < min(3, matrix.shape[0]):
                selected.append(index)
        if len(selected) >= limit:
            break
    return selected


def _exemplar_weights(matrix: np.ndarray, qualities: np.ndarray, positions: list[int]) -> np.ndarray:
    if not positions:
        return np.empty((0,), dtype=np.float32)
    if matrix.shape[0] > 1:
        pairwise = matrix @ matrix.T
        np.fill_diagonal(pairwise, np.nan)
        with np.errstate(invalid="ignore"):
            consistency = np.nanmean(pairwise, axis=1)
        consistency = np.nan_to_num(consistency, nan=1.0)
    else:
        consistency = np.ones((matrix.shape[0],), dtype=np.float32)
    raw = []
    for position in positions:
        quality = float(np.clip(qualities[position], 0.10, 1.0))
        centrality = float(np.clip((consistency[position] + 1.0) * 0.5, 0.10, 1.0))
        raw.append(max(0.05, quality * 0.65 + centrality * 0.35))
    weights = np.asarray(raw, dtype=np.float32)
    total = float(weights.sum())
    if total > 1e-8:
        weights = weights / total
    return weights.astype(np.float32, copy=False)


def _rows_to_embedding_matrix(rows: list[sqlite3.Row], *, expected_dim: int) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    face_ids: list[int] = []
    for row in rows:
        dim = int(row["embedding_dim"])
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        if dim != expected_dim or embedding.size != expected_dim:
            continue
        vectors.append(_normalize_vector(embedding.astype(np.float32, copy=True)))
        face_ids.append(int(row["id"]))
    if not vectors:
        return np.empty((0, expected_dim), dtype=np.float32), []
    return np.vstack(vectors).astype(np.float32, copy=False), face_ids


def _score_identity_profile(profile: IdentityProfile, query: np.ndarray) -> tuple[float, int | None]:
    exemplars = _profile_exemplars(profile)
    weights = profile.exemplar_weights
    if weights.size != exemplars.shape[0]:
        weights = np.ones((exemplars.shape[0],), dtype=np.float32) / max(1, exemplars.shape[0])
    score, best_index = _score_identity_components(
        profile.centroid,
        exemplars,
        weights,
        query,
        hard_negative_embeddings=profile.hard_negative_embeddings,
    )
    if best_index < len(profile.exemplar_face_ids):
        evidence_face_id = profile.exemplar_face_ids[best_index]
    elif best_index < len(profile.evidence_face_ids):
        evidence_face_id = profile.evidence_face_ids[best_index]
    else:
        evidence_face_id = profile.evidence_face_ids[0] if profile.evidence_face_ids else None
    return score, evidence_face_id


def _score_matrix_against_profile(
    centroid: np.ndarray,
    exemplars: np.ndarray,
    exemplar_weights: np.ndarray,
    query_matrix: np.ndarray,
    *,
    hard_negative_embeddings: np.ndarray | None,
) -> np.ndarray:
    if query_matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    scores = [
        _score_identity_components(
            centroid,
            exemplars,
            exemplar_weights,
            query_matrix[index],
            hard_negative_embeddings=hard_negative_embeddings,
        )[0]
        for index in range(query_matrix.shape[0])
    ]
    return np.asarray(scores, dtype=np.float32)


def _score_identity_components(
    centroid: np.ndarray,
    exemplars: np.ndarray,
    exemplar_weights: np.ndarray,
    query: np.ndarray,
    *,
    hard_negative_embeddings: np.ndarray | None,
) -> tuple[float, int]:
    exemplar_scores = exemplars @ query
    best_index = int(np.argmax(exemplar_scores)) if exemplar_scores.size else 0
    best = float(exemplar_scores[best_index]) if exemplar_scores.size else float(centroid @ query)
    top_count = min(3, exemplar_scores.size)
    top_mean = float(np.mean(np.sort(exemplar_scores)[-top_count:])) if top_count else best
    centroid_score = float(centroid @ query)
    reconstruction_score = _ridge_reconstruction_score(exemplars, query)
    weight_score = best
    if exemplar_weights.size == exemplar_scores.size and exemplar_scores.size:
        weight_score = float(np.sum(exemplar_scores * exemplar_weights))
    hard_penalty = _hard_negative_penalty(hard_negative_embeddings, query, max(best, centroid_score))
    features = np.asarray(
        [
            best,
            top_mean,
            centroid_score,
            reconstruction_score,
            weight_score,
            float(exemplar_scores.size),
            hard_penalty,
            1.0,
        ],
        dtype=np.float32,
    )
    fallback = (
        best * 0.32
        + top_mean * 0.20
        + centroid_score * 0.20
        + reconstruction_score * 0.20
        + weight_score * 0.08
        - hard_penalty
    )
    return _optional_identity_scorer_score(features, fallback), best_index


def _ridge_reconstruction_score(exemplars: np.ndarray, query: np.ndarray) -> float:
    if exemplars.size == 0:
        return 0.0
    if exemplars.shape[0] == 1:
        return float(exemplars[0] @ query)
    gram = exemplars @ exemplars.T
    rhs = exemplars @ query
    ridge = np.eye(exemplars.shape[0], dtype=np.float32) * 0.035
    try:
        weights = np.linalg.solve(gram + ridge, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + ridge, rhs, rcond=None)[0]
    weights = np.clip(weights.astype(np.float32, copy=False), 0.0, None)
    total = float(weights.sum())
    if total <= 1e-8:
        return float(np.max(exemplars @ query))
    reconstruction = weights @ exemplars / total
    reconstruction = _normalize_vector(reconstruction)
    return float(reconstruction @ query)


def _hard_negative_penalty(
    hard_negative_embeddings: np.ndarray | None,
    query: np.ndarray,
    profile_score: float,
) -> float:
    if hard_negative_embeddings is None or hard_negative_embeddings.size == 0:
        return 0.0
    hard_best = float(np.max(hard_negative_embeddings @ query))
    return _clamp((hard_best - profile_score + 0.035) * 0.60, 0.0, 0.12)


def _select_hard_negative_face_ids(negative_scores: np.ndarray, negative_face_ids: list[int], limit: int = 12) -> list[int]:
    if negative_scores.size == 0 or not negative_face_ids:
        return []
    order = np.argsort(negative_scores)[::-1][: max(0, int(limit))]
    return [int(negative_face_ids[int(index)]) for index in order]


def _hard_negative_embeddings(negative_scores: np.ndarray, negative_matrix: np.ndarray, limit: int = 12) -> np.ndarray | None:
    if negative_scores.size == 0 or negative_matrix.size == 0:
        return None
    order = np.argsort(negative_scores)[::-1][: max(0, int(limit))]
    if order.size == 0:
        return None
    return negative_matrix[order].astype(np.float32, copy=True)


def _leave_one_out_profile_scores(matrix: np.ndarray, qualities: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        keep = np.arange(matrix.shape[0]) != index
        support = matrix[keep]
        support_quality = qualities[keep]
        weights = np.clip(support_quality, 0.10, 1.0)
        centroid = _normalize_vector(np.average(support, axis=0, weights=weights))
        positions = _select_identity_exemplar_positions(
            support,
            support_quality,
            max_exemplars=min(IDENTITY_MAX_EXEMPLARS, max(1, support.shape[0])),
        )
        exemplars = support[positions].astype(np.float32, copy=False)
        exemplar_weights = _exemplar_weights(support, support_quality, positions)
        scores.append(
            _score_identity_components(
                centroid,
                exemplars,
                exemplar_weights,
                matrix[index],
                hard_negative_embeddings=None,
            )[0]
        )
    return np.asarray(scores, dtype=np.float32)


def _calibrate_identity_thresholds(
    *,
    sample_count: int,
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    mean_similarity: float,
    score_std: float,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    if positive_scores.size:
        pos_p05 = float(np.percentile(positive_scores, 5))
        pos_p10 = float(np.percentile(positive_scores, 10))
        pos_p20 = float(np.percentile(positive_scores, 20))
        pos_mean = float(positive_scores.mean())
    else:
        base = mean_similarity - max(0.04, score_std * 1.2)
        pos_p05 = pos_p10 = pos_p20 = pos_mean = base
    if negative_scores.size:
        neg_p90 = float(np.percentile(negative_scores, 90))
        neg_p95 = float(np.percentile(negative_scores, 95))
        neg_p99 = float(np.percentile(negative_scores, 99))
        neg_max = float(negative_scores.max())
    else:
        neg_p90 = neg_p95 = neg_p99 = neg_max = 0.0

    if sample_count < IDENTITY_MIN_STRONG_SAMPLES:
        strict_accept = 0.92
        balanced_accept = 0.90
        broad_accept = 0.88
    else:
        strict_accept = _clamp(max(neg_p99 + 0.045, pos_p20 - 0.025), 0.48, 0.94)
        balanced_accept = _clamp(max(neg_p95 + 0.030, pos_p10 - 0.045), 0.43, strict_accept)
        broad_accept = _clamp(max(neg_p90 + 0.015, pos_p05 - 0.070), 0.38, balanced_accept)
    thresholds = {
        "strict": {
            "accept": strict_accept,
            "review": _clamp(strict_accept - 0.100, 0.35, strict_accept - 0.020),
            "margin": 0.055,
        },
        "balanced": {
            "accept": balanced_accept,
            "review": _clamp(balanced_accept - 0.115, 0.32, balanced_accept - 0.020),
            "margin": 0.040,
        },
        "broad": {
            "accept": broad_accept,
            "review": _clamp(broad_accept - 0.130, 0.28, broad_accept - 0.020),
            "margin": 0.025,
        },
    }
    calibration: dict[str, object] = {
        "positive_count": int(positive_scores.size),
        "negative_count": int(negative_scores.size),
        "positive_mean": pos_mean,
        "positive_p05": pos_p05,
        "positive_p20": pos_p20,
        "negative_p95": neg_p95,
        "negative_p99": neg_p99,
        "negative_max": neg_max,
        "strict_gap": strict_accept - neg_max if negative_scores.size else None,
    }
    return thresholds, calibration


def _identity_profile_health(
    sample_count: int,
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    status: str,
) -> str:
    if status == "weak":
        return "weak: add at least 3 confirmed faces"
    if sample_count < 5:
        return "fair: add more angles"
    if positive_scores.size and negative_scores.size:
        gap = float(np.percentile(positive_scores, 10) - np.percentile(negative_scores, 95))
        if gap < 0.03:
            return "risky: close hard negatives"
        if gap < 0.08:
            return "fair: narrow margin"
    return "healthy"


def _active_identity_scorer_version() -> str:
    if _identity_scorer_session() is not None:
        return f"onnx:{IDENTITY_SCORER_MODEL_PATH.name}"
    return IDENTITY_NUMPY_SCORER_VERSION


def _optional_identity_scorer_score(features: np.ndarray, fallback: float) -> float:
    session = _identity_scorer_session()
    if session is None:
        return float(fallback)
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        batch = features.astype(np.float32, copy=False).reshape(1, -1)
        output = session.run([output_name], {input_name: batch})[0]
        return float(np.asarray(output, dtype=np.float32).reshape(-1)[0])
    except Exception:
        return float(fallback)


def _identity_scorer_session():
    global _IDENTITY_SCORER_SESSION, _IDENTITY_SCORER_CACHE_KEY
    path = IDENTITY_SCORER_MODEL_PATH
    if not path.exists():
        return None
    stat = path.stat()
    cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    with _IDENTITY_SCORER_LOCK:
        if _IDENTITY_SCORER_SESSION is not None and _IDENTITY_SCORER_CACHE_KEY == cache_key:
            return _IDENTITY_SCORER_SESSION
        try:
            import onnxruntime

            _IDENTITY_SCORER_SESSION = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _IDENTITY_SCORER_CACHE_KEY = cache_key
        except Exception:
            _IDENTITY_SCORER_SESSION = None
            _IDENTITY_SCORER_CACHE_KEY = cache_key
        return _IDENTITY_SCORER_SESSION


def _select_identity_prototype_positions(
    matrix: np.ndarray,
    qualities: np.ndarray,
    *,
    max_prototypes: int,
) -> list[int]:
    order = np.lexsort((np.arange(matrix.shape[0]), -qualities))
    selected: list[int] = []
    for position in order:
        index = int(position)
        if not selected:
            selected.append(index)
            continue
        similarities = matrix[selected] @ matrix[index]
        if float(similarities.max()) < IDENTITY_PROTOTYPE_DIVERSITY:
            selected.append(index)
        if len(selected) >= max(1, int(max_prototypes)):
            break
    return selected


def _upper_pairwise_scores(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)
    scores = matrix @ matrix.T
    return scores[np.triu_indices(matrix.shape[0], k=1)].astype(np.float32, copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32, copy=False)


def _identity_confidence(profile: IdentityProfile, score: float, margin: float, identity_mode: str) -> float:
    thresholds = _profile_thresholds_for_mode(profile, identity_mode)
    threshold_span = max(0.04, thresholds["accept"] - thresholds["review"])
    score_confidence = _clamp((score - thresholds["review"]) / threshold_span, 0.0, 1.0)
    margin_confidence = _clamp(margin / max(thresholds["margin"] * 2.0, 1e-6), 0.0, 1.0)
    if profile.status == "weak":
        return min(0.69, score_confidence * 0.75 + margin_confidence * 0.15)
    return _clamp(score_confidence * 0.68 + margin_confidence * 0.32, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _record_matches_search_filters(
    record: FaceRecord,
    *,
    include_ignored: bool,
    person_filter: str,
    tag_filter: str,
    min_quality: float,
) -> bool:
    if record.ignored and not include_ignored:
        return False
    if person_filter and record.person_name != person_filter:
        return False
    if tag_filter and tag_filter not in (record.tags or []):
        return False
    if record.quality_score < min_quality:
        return False
    return True


def _best_matching_analysis_for_record(record: FaceRecord, faces: list[AnalyzedFace]) -> AnalyzedFace:
    if len(faces) == 1:
        return faces[0]
    if record.bbox and len(record.bbox) >= 4:
        scored_by_bbox = [(_bbox_iou(record.bbox, face.bbox), face) for face in faces]
        best_score, best_face = max(scored_by_bbox, key=lambda item: item[0])
        if best_score > 0:
            return best_face
    return max(faces, key=lambda face: cosine_similarity(record.embedding, face.embedding))


def _extract_mediapipe_face_meshes(image_bgr: np.ndarray) -> list[list[list[float]]]:
    import mediapipe as mp

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    with _FACE_LANDMARKER_LOCK:
        landmarker = _get_face_landmarker(mp)
        results = landmarker.detect(mp_image)

    meshes: list[list[list[float]]] = []
    for face_landmarks in getattr(results, "face_landmarks", None) or []:
        points = [
            [
                float(landmark.x) * width,
                float(landmark.y) * height,
                float(landmark.z) * width,
            ]
            for landmark in face_landmarks
        ]
        if points:
            meshes.append(points)
    return meshes


def _get_face_landmarker(mp):
    global _FACE_LANDMARKER

    with _FACE_LANDMARKER_LOCK:
        if _FACE_LANDMARKER is None:
            _ensure_mediapipe_face_landmarker_model()
            base_options = mp.tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_FACE_LANDMARKER_PATH))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=10,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            _FACE_LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return _FACE_LANDMARKER


def _ensure_mediapipe_face_landmarker_model() -> None:
    if MEDIAPIPE_FACE_LANDMARKER_PATH.exists():
        return

    MEDIAPIPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MEDIAPIPE_FACE_LANDMARKER_PATH.with_suffix(".task.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    urllib.request.urlretrieve(MEDIAPIPE_FACE_LANDMARKER_URL, temporary_path)
    temporary_path.replace(MEDIAPIPE_FACE_LANDMARKER_PATH)


def _best_matching_mesh_for_record(record: FaceRecord, meshes: list[list[list[float]]]) -> list[list[float]]:
    if len(meshes) == 1:
        return meshes[0]
    if record.bbox and len(record.bbox) >= 4:
        scored = [(_bbox_iou(record.bbox, _bbox_from_points(mesh)), mesh) for mesh in meshes]
        best_score, best_mesh = max(scored, key=lambda item: item[0])
        if best_score > 0:
            return best_mesh
    return max(meshes, key=lambda mesh: _bbox_area(_bbox_from_points(mesh)))


def _bbox_from_points(points: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in points if len(point) >= 2]
    ys = [float(point[1]) for point in points if len(point) >= 2]
    if not xs or not ys:
        return []
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox_area(bbox: Iterable[float]) -> float:
    values = list(bbox)[:4]
    if len(values) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in values]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    values_a = list(a)[:4]
    values_b = list(b)[:4]
    if len(values_a) < 4 or len(values_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in values_a]
    bx1, by1, bx2, by2 = [float(value) for value in values_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def export_faces_csv(database_path: str | Path, output_path: str | Path) -> str:
    records, metadata = load_records(database_path, include_ignored=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "file_name",
                "person",
                "tags",
                "det_score",
                "quality_score",
                "review_state",
                "ignored",
                "duplicate_count",
                "image_hash",
                "source_path",
                "notes",
                "database_format",
                "model_name",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.id,
                    record.file_name,
                    record.person_name,
                    record.tag_text,
                    f"{record.det_score:.6f}",
                    f"{record.quality_score:.6f}",
                    record.review_state,
                    int(record.ignored),
                    record.duplicate_count,
                    record.image_hash,
                    record.source_path,
                    record.notes,
                    metadata.get("format_version", ""),
                    metadata.get("model_name", ""),
                ]
            )
    return str(output)


def parse_tag_text(tag_text: str) -> list[str]:
    parts = tag_text.replace(";", ",").replace("|", ",").split(",")
    clean: list[str] = []
    seen = set()
    for part in parts:
        tag = part.strip()
        if not tag or tag.lower() in seen:
            continue
        clean.append(tag)
        seen.add(tag.lower())
    return clean


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
