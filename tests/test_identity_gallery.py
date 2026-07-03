from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsc_face_database import ensure_person, initialize_database, insert_face
from fsc_studio_services import identify_person, rebuild_identity_gallery_profiles


def unit(vector: list[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def add_face(conn, person_id: int, name: str, embedding: np.ndarray, quality: float = 0.95) -> int:
    return insert_face(
        conn,
        {
            "file_name": name,
            "embedding": embedding,
            "bbox": [],
            "kps": [],
            "det_score": 0.99,
            "quality_score": quality,
            "quality": {"synthetic": True},
            "person_id": person_id,
            "review_state": "reviewed",
        },
    )


class IdentityGalleryTests(unittest.TestCase):
    def make_database(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "gallery.fscdb"
        conn = initialize_database(path)
        try:
            alice = ensure_person(conn, "Alice")
            bob = ensure_person(conn, "Bob")
            weak = ensure_person(conn, "Weak")
            alice_vectors = [
                unit([1.00, 0.02, 0.00, 0.00]),
                unit([0.99, 0.05, 0.01, 0.00]),
                unit([0.98, -0.04, 0.02, 0.00]),
                unit([0.97, 0.02, -0.03, 0.00]),
                unit([0.96, -0.03, -0.02, 0.00]),
            ]
            bob_vectors = [
                unit([0.03, 1.00, 0.00, 0.00]),
                unit([-0.02, 0.99, 0.02, 0.00]),
                unit([0.04, 0.98, -0.03, 0.00]),
                unit([-0.03, 0.97, 0.04, 0.00]),
                unit([0.02, 0.96, -0.04, 0.00]),
            ]
            for index, vector in enumerate(alice_vectors):
                add_face(conn, alice, f"alice_{index}.jpg", vector)
            for index, vector in enumerate(bob_vectors):
                add_face(conn, bob, f"bob_{index}.jpg", vector)
            add_face(conn, weak, "weak_0.jpg", unit([0.00, 0.00, 1.00, 0.00]))
            conn.commit()
        finally:
            conn.close()
        return path

    def test_gallery_profiles_identify_strict_same_person(self) -> None:
        path = self.make_database()
        summary = rebuild_identity_gallery_profiles(path)
        self.assertEqual(summary.profiles_built, 3)
        result = identify_person(path, unit([0.995, 0.035, 0.01, 0.00]), identity_mode="strict")
        self.assertEqual(result.decision, "confirmed")
        self.assertEqual(result.candidates[0].profile.person_name, "Alice")

    def test_weak_profiles_do_not_auto_confirm(self) -> None:
        path = self.make_database()
        rebuild_identity_gallery_profiles(path)
        result = identify_person(path, unit([0.00, 0.00, 1.00, 0.00]), identity_mode="broad")
        self.assertEqual(result.candidates[0].profile.person_name, "Weak")
        self.assertEqual(result.decision, "review")

    def test_strict_mode_rejects_far_negative(self) -> None:
        path = self.make_database()
        rebuild_identity_gallery_profiles(path)
        result = identify_person(path, unit([0.60, 0.80, 0.00, 0.00]), identity_mode="strict")
        self.assertIn(result.decision, {"review", "unknown"})


if __name__ == "__main__":
    unittest.main()
