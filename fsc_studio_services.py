from __future__ import annotations

import csv
import hashlib
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
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
    update_face_review as db_update_face_review,
)
from fsc_face_engine import (
    MODEL_NAME,
    AnalyzedFace,
    FaceEngineError,
    analysis_to_database_record,
    cosine_similarity,
    get_engine,
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
                (
                    SELECT f2.id
                    FROM faces f2
                    WHERE f2.person_id = p.id
                    ORDER BY f2.quality_score DESC, f2.id ASC
                    LIMIT 1
                ) AS representative_face_id
            FROM persons p
            LEFT JOIN faces f ON f.person_id = p.id
            {where_sql}
            GROUP BY p.id, p.name, p.notes
            ORDER BY face_count DESC, p.name COLLATE NOCASE
            {limit_sql}
            """,
            params,
        ).fetchall()
        return [
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
            )
            for row in rows
        ]
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


def clear_search_index_cache(database_path: str | Path | None = None) -> None:
    with _SEARCH_INDEX_LOCK:
        if database_path is None:
            _SEARCH_INDEX_CACHE.clear()
            return
        normalized = str(Path(normalize_database_path(database_path)).resolve())
        _SEARCH_INDEX_CACHE.pop(normalized, None)


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
