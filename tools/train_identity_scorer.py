from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsc_face_engine import get_engine


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr if norm <= 1e-12 else (arr / norm).astype(np.float32, copy=False)


def ridge_reconstruction_score(exemplars: np.ndarray, query: np.ndarray) -> float:
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
    reconstruction = normalize(weights @ exemplars / total)
    return float(reconstruction @ query)


def identity_features(support: np.ndarray, query: np.ndarray) -> np.ndarray:
    centroid = normalize(support.mean(axis=0))
    scores = support @ query
    best = float(np.max(scores))
    top_count = min(3, scores.size)
    top_mean = float(np.mean(np.sort(scores)[-top_count:])) if top_count else best
    centroid_score = float(centroid @ query)
    reconstruction_score = ridge_reconstruction_score(support, query)
    weight_score = float(np.mean(scores))
    return np.asarray(
        [best, top_mean, centroid_score, reconstruction_score, weight_score, float(support.shape[0]), 0.0, 1.0],
        dtype=np.float32,
    )


def iter_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def build_embedding_dataset(
    dataset_root: Path,
    *,
    prefer_gpu: bool,
    max_images_per_person: int,
) -> dict[str, np.ndarray]:
    engine = get_engine(prefer_gpu=prefer_gpu)
    people: dict[str, np.ndarray] = {}
    for person_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        vectors = []
        for image_path in iter_images(person_dir)[:max_images_per_person]:
            try:
                faces = engine.extract_faces_from_path(image_path)
            except Exception as exc:
                print(f"skip {image_path}: {exc}")
                continue
            if not faces:
                continue
            vectors.append(normalize(faces[0].embedding))
        if len(vectors) >= 2:
            people[person_dir.name] = np.vstack(vectors).astype(np.float32, copy=False)
            print(f"{person_dir.name}: {len(vectors)} usable face(s)")
    return people


def make_training_examples(
    people: dict[str, np.ndarray],
    *,
    support_size: int,
    rounds_per_person: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    names = sorted(people)
    features: list[np.ndarray] = []
    labels: list[float] = []
    for name in names:
        vectors = people[name]
        if vectors.shape[0] < 2:
            continue
        other_names = [item for item in names if item != name]
        for _ in range(rounds_per_person):
            query_index = rng.randrange(vectors.shape[0])
            support_pool = [idx for idx in range(vectors.shape[0]) if idx != query_index]
            rng.shuffle(support_pool)
            support_indexes = support_pool[: max(1, min(support_size, len(support_pool)))]
            support = vectors[support_indexes]
            features.append(identity_features(support, vectors[query_index]))
            labels.append(1.0)
            if other_names:
                negative_name = rng.choice(other_names)
                negative_vectors = people[negative_name]
                negative_query = negative_vectors[rng.randrange(negative_vectors.shape[0])]
                features.append(identity_features(support, negative_query))
                labels.append(0.0)
    if not features:
        raise RuntimeError("Need at least one identity folder with two or more usable faces.")
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


def train_model(features: np.ndarray, labels: np.ndarray, *, epochs: int, lr: float, seed: int):
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = torch.nn.BCELoss()
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                accuracy = ((pred >= 0.5) == (y >= 0.5)).float().mean().item()
            print(f"epoch {epoch:04d} loss {loss.item():.4f} accuracy {accuracy:.3f}")
    return model


def export_onnx(model, output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 8), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optional FSC identity scorer ONNX head.")
    parser.add_argument("dataset_root", type=Path, help="Folder with one subfolder per identity.")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "identity_scorer.onnx")
    parser.add_argument("--cpu", action="store_true", help="Force CPU InsightFace embedding extraction.")
    parser.add_argument("--max-images-per-person", type=int, default=80)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--rounds-per-person", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    people = build_embedding_dataset(
        args.dataset_root,
        prefer_gpu=not args.cpu,
        max_images_per_person=max(1, args.max_images_per_person),
    )
    features, labels = make_training_examples(
        people,
        support_size=max(1, args.support_size),
        rounds_per_person=max(1, args.rounds_per_person),
        seed=args.seed,
    )
    print(f"training examples: {features.shape[0]}, feature_dim: {features.shape[1]}")
    model = train_model(features, labels, epochs=max(1, args.epochs), lr=args.lr, seed=args.seed)
    export_onnx(model, args.output)
    metadata = {
        "dataset_root": str(args.dataset_root),
        "identities": len(people),
        "examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "output": str(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
