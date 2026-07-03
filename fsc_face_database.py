from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


DB_FORMAT_VERSION = "8"
DEFAULT_EXTENSION = ".fscdb"
METRIC_NAME = "cosine_normed_embedding"
DEFAULT_REVIEW_STATE = "open"
REVIEW_STATES = {"open", "reviewed", "duplicate", "low_quality", "ignored"}


class FaceDatabaseError(RuntimeError):
    pass


class LegacyDatabaseError(FaceDatabaseError):
    pass


@dataclass
class FaceRecord:
    id: int
    file_name: str
    source_path: str
    embedding: np.ndarray
    bbox: list
    kps: list
    landmarks: list | None
    landmarks3d: list | None
    face_mesh3d: list | None
    det_score: float
    quality_score: float
    quality: dict
    preview_png: bytes | None
    person_id: int | None = None
    person_name: str = ""
    tags: list[str] | None = None
    image_hash: str = ""
    ignored: bool = False
    review_state: str = DEFAULT_REVIEW_STATE
    notes: str = ""
    duplicate_count: int = 0

    @property
    def tag_text(self) -> str:
        return ", ".join(self.tags or [])


@dataclass
class PersonRecord:
    id: int
    name: str
    notes: str = ""


@dataclass
class TagRecord:
    id: int
    name: str


def normalize_database_path(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.suffix:
        path_obj = path_obj.with_suffix(DEFAULT_EXTENSION)
    return str(path_obj)


def initialize_database(
    database_path: str | Path,
    metadata: dict[str, object] | None = None,
    replace: bool = True,
) -> sqlite3.Connection:
    path = Path(normalize_database_path(database_path))
    if path.suffix.lower() == ".dtb":
        raise LegacyDatabaseError("Legacy .dtb files cannot be overwritten with the new database format.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace and path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_schema(conn)
    write_metadata(conn, _default_metadata(metadata))
    return conn


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    if path.suffix.lower() == ".dtb":
        raise LegacyDatabaseError("This is a legacy dlib .dtb database. Rebuild or convert it to .fscdb first.")
    if not path.exists():
        raise FaceDatabaseError(f"Database file does not exist: {path}")
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        metadata = read_metadata(conn)
    except sqlite3.DatabaseError as exc:
        raise FaceDatabaseError(f"Failed to open SQLite face database: {exc}") from exc

    version = metadata.get("format_version")
    if version == "2":
        _migrate_v2_to_v3(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")
    if version == "3":
        _migrate_v3_to_v4(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")
    if version == "4":
        _migrate_v4_to_v5(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")
    if version == "5":
        _migrate_v5_to_v6(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")
    if version == "6":
        _migrate_v6_to_v7(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")
    if version == "7":
        _migrate_v7_to_v8(conn)
        metadata = read_metadata(conn)
        version = metadata.get("format_version")

    if version != DB_FORMAT_VERSION:
        conn.close()
        raise FaceDatabaseError(f"Unsupported database format version: {version!r}.")
    return conn


def insert_face(conn: sqlite3.Connection, record: dict[str, object]) -> int:
    embedding = np.asarray(record["embedding"], dtype=np.float32).reshape(-1)
    review_state = _normalize_review_state(str(record.get("review_state") or DEFAULT_REVIEW_STATE))
    cursor = conn.execute(
        """
        INSERT INTO faces (
            file_name,
            source_path,
            embedding_blob,
            embedding_dim,
            bbox_json,
            kps_json,
            landmarks_json,
            landmarks3d_json,
            face_mesh3d_json,
            det_score,
            quality_score,
            quality_json,
            preview_png,
            person_id,
            image_hash,
            ignored,
            review_state,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(record.get("file_name", "")),
            str(record.get("source_path", "")),
            embedding.tobytes(),
            int(embedding.size),
            _json_dump(record.get("bbox", [])),
            _json_dump(record.get("kps", [])),
            _json_dump(record.get("landmarks")),
            _json_dump(record.get("landmarks3d")),
            _json_dump(record.get("face_mesh3d")),
            float(record.get("det_score", 0.0)),
            float(record.get("quality_score", 0.0)),
            _json_dump(record.get("quality", {})),
            record.get("preview_png"),
            record.get("person_id"),
            str(record.get("image_hash") or ""),
            1 if bool(record.get("ignored", False)) else 0,
            review_state,
            str(record.get("notes") or ""),
        ),
    )
    return int(cursor.lastrowid)


def load_face_records(
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
    conn = connect_database(database_path)
    try:
        metadata = read_metadata(conn)
        records = _select_face_records(
            conn,
            include_ignored=include_ignored,
            person_filter=person_filter,
            tag_filter=tag_filter,
            review_filter=review_filter,
            min_quality=min_quality,
            text_filter=text_filter,
            include_preview=include_preview,
            limit=limit,
        )
        return records, metadata
    finally:
        conn.close()


def load_review_records(
    database_path: str | Path,
    *,
    limit: int = 500,
    text_filter: str = "",
) -> tuple[list[FaceRecord], dict[str, str]]:
    conn = connect_database(database_path)
    try:
        metadata = read_metadata(conn)
        records = _select_face_records(
            conn,
            include_ignored=True,
            review_filter="needs_review",
            text_filter=text_filter,
            limit=limit,
        )
        return records, metadata
    finally:
        conn.close()


def load_face_preview(database_path: str | Path, face_id: int) -> bytes | None:
    conn = connect_database(database_path)
    try:
        row = conn.execute("SELECT preview_png FROM faces WHERE id = ?", (int(face_id),)).fetchone()
        if row is None:
            return None
        return row["preview_png"]
    finally:
        conn.close()


def update_face_landmarks3d(database_path: str | Path, face_id: int, landmarks3d: object) -> None:
    conn = connect_database(database_path)
    try:
        conn.execute(
            "UPDATE faces SET landmarks3d_json = ? WHERE id = ?",
            (_json_dump(landmarks3d), int(face_id)),
        )
        conn.commit()
    finally:
        conn.close()


def update_face_mesh3d(database_path: str | Path, face_id: int, face_mesh3d: object) -> None:
    conn = connect_database(database_path)
    try:
        conn.execute(
            "UPDATE faces SET face_mesh3d_json = ? WHERE id = ?",
            (_json_dump(face_mesh3d), int(face_id)),
        )
        conn.commit()
    finally:
        conn.close()


def read_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.DatabaseError as exc:
        raise FaceDatabaseError("Missing FSC metadata table.") from exc
    return {str(row[0]): str(row[1]) for row in rows}


def write_metadata(conn: sqlite3.Connection, metadata: dict[str, object]) -> None:
    for key, value in metadata.items():
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (str(key), str(value)),
        )
    conn.commit()


def count_faces(database_path: str | Path) -> int:
    conn = connect_database(database_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
    finally:
        conn.close()


def database_statistics(database_path: str | Path) -> dict[str, object]:
    conn = connect_database(database_path)
    try:
        metadata = read_metadata(conn)
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS face_count,
                AVG(det_score) AS avg_detection,
                AVG(quality_score) AS avg_quality,
                MIN(quality_score) AS min_quality,
                MAX(quality_score) AS max_quality,
                SUM(CASE WHEN ignored = 1 THEN 1 ELSE 0 END) AS ignored_count,
                SUM(CASE WHEN review_state != 'reviewed' OR ignored = 1 THEN 1 ELSE 0 END) AS review_count
            FROM faces
            """
        ).fetchone()
        people_count = int(conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0])
        tag_count = int(conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0])
        duplicate_hash_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT image_hash
                    FROM faces
                    WHERE image_hash IS NOT NULL AND image_hash != ''
                    GROUP BY image_hash
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        return {
            "face_count": int(row["face_count"] or 0),
            "avg_detection": float(row["avg_detection"] or 0.0),
            "avg_quality": float(row["avg_quality"] or 0.0),
            "min_quality": float(row["min_quality"] or 0.0),
            "max_quality": float(row["max_quality"] or 0.0),
            "ignored_count": int(row["ignored_count"] or 0),
            "review_count": int(row["review_count"] or 0),
            "people_count": people_count,
            "tag_count": tag_count,
            "duplicate_hash_count": duplicate_hash_count,
            "format_version": metadata.get("format_version", ""),
            "model_name": metadata.get("model_name", ""),
        }
    finally:
        conn.close()


def list_people(database_path: str | Path) -> list[PersonRecord]:
    conn = connect_database(database_path)
    try:
        return [
            PersonRecord(id=int(row["id"]), name=str(row["name"]), notes=str(row["notes"] or ""))
            for row in conn.execute("SELECT id, name, notes FROM persons ORDER BY name COLLATE NOCASE")
        ]
    finally:
        conn.close()


def list_tags(database_path: str | Path) -> list[TagRecord]:
    conn = connect_database(database_path)
    try:
        return [
            TagRecord(id=int(row["id"]), name=str(row["name"]))
            for row in conn.execute("SELECT id, name FROM tags ORDER BY name COLLATE NOCASE")
        ]
    finally:
        conn.close()


def assign_person(database_path: str | Path, face_id: int, person_name: str, notes: str = "") -> None:
    conn = connect_database(database_path)
    try:
        person_id = None
        clean_name = person_name.strip()
        if clean_name:
            person_id = ensure_person(conn, clean_name, notes)
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, int(face_id)))
        conn.commit()
    finally:
        conn.close()


def set_face_tags(database_path: str | Path, face_id: int, tag_names: Iterable[str]) -> None:
    conn = connect_database(database_path)
    try:
        set_face_tags_on_connection(conn, face_id, tag_names)
        conn.commit()
    finally:
        conn.close()


def update_face_review(
    database_path: str | Path,
    face_id: int,
    *,
    ignored: bool | None = None,
    review_state: str | None = None,
    notes: str | None = None,
) -> None:
    fields: list[str] = []
    values: list[object] = []
    if ignored is not None:
        fields.append("ignored = ?")
        values.append(1 if ignored else 0)
    if review_state is not None:
        fields.append("review_state = ?")
        values.append(_normalize_review_state(review_state))
    if notes is not None:
        fields.append("notes = ?")
        values.append(notes)
    if not fields:
        return
    values.append(int(face_id))
    conn = connect_database(database_path)
    try:
        conn.execute(f"UPDATE faces SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


def ensure_person(conn: sqlite3.Connection, name: str, notes: str = "") -> int:
    clean_name = name.strip()
    if not clean_name:
        raise FaceDatabaseError("Person name cannot be empty.")
    row = conn.execute("SELECT id FROM persons WHERE name = ?", (clean_name,)).fetchone()
    if row:
        if notes:
            conn.execute("UPDATE persons SET notes = ? WHERE id = ? AND COALESCE(notes, '') = ''", (notes, int(row["id"])))
        return int(row["id"])
    cursor = conn.execute(
        "INSERT INTO persons(name, notes) VALUES (?, ?)",
        (clean_name, notes),
    )
    return int(cursor.lastrowid)


def set_face_tags_on_connection(conn: sqlite3.Connection, face_id: int, tag_names: Iterable[str]) -> None:
    clean_names = []
    seen = set()
    for raw_name in tag_names:
        name = str(raw_name).strip()
        if not name or name.lower() in seen:
            continue
        clean_names.append(name)
        seen.add(name.lower())
    conn.execute("DELETE FROM face_tags WHERE face_id = ?", (int(face_id),))
    for name in clean_names:
        tag_id = ensure_tag(conn, name)
        conn.execute(
            "INSERT OR IGNORE INTO face_tags(face_id, tag_id) VALUES (?, ?)",
            (int(face_id), tag_id),
        )


def ensure_tag(conn: sqlite3.Connection, name: str) -> int:
    clean_name = name.strip()
    if not clean_name:
        raise FaceDatabaseError("Tag name cannot be empty.")
    row = conn.execute("SELECT id FROM tags WHERE name = ?", (clean_name,)).fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute("INSERT INTO tags(name) VALUES (?)", (clean_name,))
    return int(cursor.lastrowid)


def existing_image_hashes(conn: sqlite3.Connection, image_hashes: Iterable[str]) -> set[str]:
    clean_hashes = [str(value) for value in image_hashes if value]
    if not clean_hashes:
        return set()
    placeholders = ",".join("?" for _ in clean_hashes)
    rows = conn.execute(
        f"SELECT DISTINCT image_hash FROM faces WHERE image_hash IN ({placeholders})",
        clean_hashes,
    ).fetchall()
    return {str(row["image_hash"]) for row in rows}


def _select_face_records(
    conn: sqlite3.Connection,
    *,
    include_ignored: bool = True,
    person_filter: str = "",
    tag_filter: str = "",
    review_filter: str = "",
    min_quality: float | None = None,
    text_filter: str = "",
    include_preview: bool = True,
    limit: int | None = None,
) -> list[FaceRecord]:
    where: list[str] = []
    params: list[object] = []
    if not include_ignored:
        where.append("f.ignored = 0")
    if person_filter:
        where.append("COALESCE(p.name, '') = ?")
        params.append(person_filter)
    if tag_filter:
        where.append(
            """
            EXISTS (
                SELECT 1
                FROM face_tags ft_filter
                JOIN tags t_filter ON t_filter.id = ft_filter.tag_id
                WHERE ft_filter.face_id = f.id AND t_filter.name = ?
            )
            """
        )
        params.append(tag_filter)
    if review_filter == "needs_review":
        where.append("(f.review_state != 'reviewed' OR f.ignored = 1 OR f.person_id IS NULL)")
    elif review_filter:
        where.append("f.review_state = ?")
        params.append(_normalize_review_state(review_filter))
    if min_quality is not None:
        where.append("f.quality_score >= ?")
        params.append(float(min_quality))
    if text_filter:
        pattern = f"%{text_filter.strip()}%"
        where.append(
            """
            (
                f.file_name LIKE ? COLLATE NOCASE OR
                COALESCE(f.source_path, '') LIKE ? COLLATE NOCASE OR
                COALESCE(p.name, '') LIKE ? COLLATE NOCASE OR
                COALESCE(f.notes, '') LIKE ? COLLATE NOCASE OR
                EXISTS (
                    SELECT 1
                    FROM face_tags ft_text
                    JOIN tags t_text ON t_text.id = ft_text.tag_id
                    WHERE ft_text.face_id = f.id AND t_text.name LIKE ? COLLATE NOCASE
                )
            )
            """
        )
        params.extend([pattern, pattern, pattern, pattern, pattern])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    limit_sql = "LIMIT ?" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    preview_expression = "f.preview_png" if include_preview else "NULL"

    rows = conn.execute(
        f"""
        SELECT
            f.id,
            f.file_name,
            f.source_path,
            f.embedding_blob,
            f.embedding_dim,
            f.bbox_json,
            f.kps_json,
            f.landmarks_json,
            f.landmarks3d_json,
            f.face_mesh3d_json,
            f.det_score,
            f.quality_score,
            f.quality_json,
            {preview_expression} AS preview_png,
            f.person_id,
            COALESCE(p.name, '') AS person_name,
            f.image_hash,
            f.ignored,
            f.review_state,
            f.notes,
            COALESCE(tags.tag_names, '') AS tag_names,
            COALESCE(dupes.duplicate_count, 0) AS duplicate_count
        FROM faces f
        LEFT JOIN persons p ON p.id = f.person_id
        LEFT JOIN (
            SELECT ft.face_id, GROUP_CONCAT(t.name, ', ') AS tag_names
            FROM face_tags ft
            JOIN tags t ON t.id = ft.tag_id
            GROUP BY ft.face_id
        ) tags ON tags.face_id = f.id
        LEFT JOIN (
            SELECT image_hash, COUNT(*) AS duplicate_count
            FROM faces
            WHERE image_hash IS NOT NULL AND image_hash != ''
            GROUP BY image_hash
        ) dupes ON dupes.image_hash = f.image_hash
        {where_sql}
        ORDER BY f.id
        {limit_sql}
        """,
        params,
    ).fetchall()
    return [_row_to_record(row) for row in rows]


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            source_path TEXT,
            embedding_blob BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            bbox_json TEXT,
            kps_json TEXT,
            landmarks_json TEXT,
            landmarks3d_json TEXT,
            face_mesh3d_json TEXT,
            det_score REAL,
            quality_score REAL NOT NULL DEFAULT 0,
            quality_json TEXT,
            preview_png BLOB,
            person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
            image_hash TEXT,
            ignored INTEGER NOT NULL DEFAULT 0,
            review_state TEXT NOT NULL DEFAULT 'open',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE face_tags (
            face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY(face_id, tag_id)
        );

        CREATE TABLE person_identity_profiles (
            person_id INTEGER PRIMARY KEY REFERENCES persons(id) ON DELETE CASCADE,
            sample_count INTEGER NOT NULL,
            prototype_count INTEGER NOT NULL,
            embedding_dim INTEGER NOT NULL,
            centroid_blob BLOB NOT NULL,
            prototypes_blob BLOB NOT NULL,
            exemplar_blob BLOB,
            exemplar_face_ids_json TEXT NOT NULL DEFAULT '[]',
            exemplar_weights_blob BLOB,
            hard_negative_face_ids_json TEXT NOT NULL DEFAULT '[]',
            thresholds_json TEXT NOT NULL DEFAULT '{}',
            calibration_json TEXT NOT NULL DEFAULT '{}',
            strategy_version TEXT NOT NULL DEFAULT 'gallery_v2',
            scoring_model_version TEXT NOT NULL DEFAULT 'numpy_gallery_v1',
            accept_threshold REAL NOT NULL,
            review_threshold REAL NOT NULL,
            mean_similarity REAL NOT NULL,
            min_similarity REAL NOT NULL,
            max_similarity REAL NOT NULL,
            quality_mean REAL NOT NULL,
            evidence_face_ids_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'weak',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_faces_file_name ON faces(file_name);
        CREATE INDEX idx_faces_quality_score ON faces(quality_score);
        CREATE INDEX idx_faces_person_id ON faces(person_id);
        CREATE INDEX idx_faces_image_hash ON faces(image_hash);
        CREATE INDEX idx_faces_ignored ON faces(ignored);
        CREATE INDEX idx_faces_review_state ON faces(review_state);
        CREATE INDEX idx_face_tags_tag_id ON face_tags(tag_id);
        """
    )
    conn.commit()


def create_identity_profile_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS person_identity_profiles (
            person_id INTEGER PRIMARY KEY REFERENCES persons(id) ON DELETE CASCADE,
            sample_count INTEGER NOT NULL,
            prototype_count INTEGER NOT NULL,
            embedding_dim INTEGER NOT NULL,
            centroid_blob BLOB NOT NULL,
            prototypes_blob BLOB NOT NULL,
            exemplar_blob BLOB,
            exemplar_face_ids_json TEXT NOT NULL DEFAULT '[]',
            exemplar_weights_blob BLOB,
            hard_negative_face_ids_json TEXT NOT NULL DEFAULT '[]',
            thresholds_json TEXT NOT NULL DEFAULT '{}',
            calibration_json TEXT NOT NULL DEFAULT '{}',
            strategy_version TEXT NOT NULL DEFAULT 'gallery_v2',
            scoring_model_version TEXT NOT NULL DEFAULT 'numpy_gallery_v1',
            accept_threshold REAL NOT NULL,
            review_threshold REAL NOT NULL,
            mean_similarity REAL NOT NULL,
            min_similarity REAL NOT NULL,
            max_similarity REAL NOT NULL,
            quality_mean REAL NOT NULL,
            evidence_face_ids_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'weak',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    columns = _table_columns(conn, "person_identity_profiles")
    _add_table_column_if_missing(conn, columns, "person_identity_profiles", "exemplar_blob", "BLOB")
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "exemplar_face_ids_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    _add_table_column_if_missing(conn, columns, "person_identity_profiles", "exemplar_weights_blob", "BLOB")
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "hard_negative_face_ids_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "thresholds_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "calibration_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "strategy_version",
        "TEXT NOT NULL DEFAULT 'legacy_v1'",
    )
    _add_table_column_if_missing(
        conn,
        columns,
        "person_identity_profiles",
        "scoring_model_version",
        "TEXT NOT NULL DEFAULT 'numpy_gallery_v1'",
    )


def _default_metadata(extra: dict[str, object] | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "format_version": DB_FORMAT_VERSION,
        "metric": METRIC_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _row_to_record(row: sqlite3.Row) -> FaceRecord:
    dim = int(row["embedding_dim"])
    embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32)
    if embedding.size != dim:
        raise FaceDatabaseError(
            f"Face record {row['id']} has corrupt embedding dimensions: {embedding.size} vs {dim}."
        )
    tag_names = str(row["tag_names"] or "")
    tags = [tag.strip() for tag in tag_names.split(",") if tag.strip()]
    return FaceRecord(
        id=int(row["id"]),
        file_name=str(row["file_name"]),
        source_path=str(row["source_path"] or ""),
        embedding=embedding.astype(np.float32, copy=True),
        bbox=_json_load(row["bbox_json"], []),
        kps=_json_load(row["kps_json"], []),
        landmarks=_json_load(row["landmarks_json"], None),
        landmarks3d=_json_load(row["landmarks3d_json"], None),
        face_mesh3d=_json_load(row["face_mesh3d_json"], None),
        det_score=float(row["det_score"] or 0.0),
        quality_score=float(row["quality_score"] or 0.0),
        quality=_json_load(row["quality_json"], {}),
        preview_png=row["preview_png"],
        person_id=int(row["person_id"]) if row["person_id"] is not None else None,
        person_name=str(row["person_name"] or ""),
        tags=tags,
        image_hash=str(row["image_hash"] or ""),
        ignored=bool(row["ignored"]),
        review_state=str(row["review_state"] or DEFAULT_REVIEW_STATE),
        notes=str(row["notes"] or ""),
        duplicate_count=int(row["duplicate_count"] or 0),
    )


def _json_dump(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str | None, default: object) -> object:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "faces")
    if "quality_score" not in columns:
        conn.execute("ALTER TABLE faces ADD COLUMN quality_score REAL NOT NULL DEFAULT 0")
    if "quality_json" not in columns:
        conn.execute("ALTER TABLE faces ADD COLUMN quality_json TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_quality_score ON faces(quality_score)")
    conn.execute(
        "UPDATE faces SET quality_json = ? WHERE quality_json IS NULL",
        (_json_dump({"migrated": True, "reason": "quality unavailable in v2"}) or "{}",),
    )
    write_metadata(
        conn,
        {
            "format_version": "3",
            "migrated_from": "2",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS face_tags (
            face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY(face_id, tag_id)
        );
        """
    )
    columns = _table_columns(conn, "faces")
    _add_column_if_missing(conn, columns, "person_id", "INTEGER")
    _add_column_if_missing(conn, columns, "image_hash", "TEXT")
    _add_column_if_missing(conn, columns, "ignored", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, columns, "review_state", "TEXT NOT NULL DEFAULT 'open'")
    _add_column_if_missing(conn, columns, "notes", "TEXT NOT NULL DEFAULT ''")
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
        CREATE INDEX IF NOT EXISTS idx_faces_image_hash ON faces(image_hash);
        CREATE INDEX IF NOT EXISTS idx_faces_ignored ON faces(ignored);
        CREATE INDEX IF NOT EXISTS idx_faces_review_state ON faces(review_state);
        CREATE INDEX IF NOT EXISTS idx_face_tags_tag_id ON face_tags(tag_id);
        """
    )
    conn.execute("UPDATE faces SET review_state = 'open' WHERE review_state IS NULL OR review_state = ''")
    conn.execute("UPDATE faces SET ignored = 0 WHERE ignored IS NULL")
    conn.execute("UPDATE faces SET notes = '' WHERE notes IS NULL")
    write_metadata(
        conn,
        {
            "format_version": "4",
            "migrated_from": "3",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "faces")
    _add_column_if_missing(conn, columns, "landmarks3d_json", "TEXT")
    write_metadata(
        conn,
        {
            "format_version": "5",
            "migrated_from": "4",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "faces")
    _add_column_if_missing(conn, columns, "face_mesh3d_json", "TEXT")
    write_metadata(
        conn,
        {
            "format_version": "6",
            "migrated_from": "5",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    create_identity_profile_schema(conn)
    write_metadata(
        conn,
        {
            "format_version": "7",
            "migrated_from": "6",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    create_identity_profile_schema(conn)
    write_metadata(
        conn,
        {
            "format_version": DB_FORMAT_VERSION,
            "migrated_from": "7",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _add_column_if_missing(
    conn: sqlite3.Connection,
    columns: set[str],
    name: str,
    declaration: str,
) -> None:
    _add_table_column_if_missing(conn, columns, "faces", name, declaration)


def _add_table_column_if_missing(
    conn: sqlite3.Connection,
    columns: set[str],
    table_name: str,
    name: str,
    declaration: str,
) -> None:
    if name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {declaration}")
        columns.add(name)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _normalize_review_state(value: str) -> str:
    cleaned = value.strip().lower()
    return cleaned if cleaned in REVIEW_STATES else DEFAULT_REVIEW_STATE
