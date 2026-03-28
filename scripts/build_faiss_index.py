"""Build a FAISS IVF index from exported embeddings.

Run with:
    .venv/bin/python scripts/build_faiss_index.py

Prerequisite: run scripts/export_faiss_data.py first.

Builds an IndexIVFFlat with 4096 clusters (METRIC_INNER_PRODUCT).
Training uses a random 100k subsample for speed; all 2.3M vectors are added.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402

NLIST = 4096        # number of IVF clusters
TRAIN_SAMPLE = 256_000  # ~62 points/centroid (heuristic: ≥40× nlist for stable centroids)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    store = Path(config.FAISS_STORE_DIR)
    vectors_path = store / "pmc_embeddings.npy"
    manifest_path = store / "manifest.json"
    index_path = store / "index.faiss"

    if not vectors_path.exists():
        print(f"ERROR: {vectors_path} not found. Run scripts/export_faiss_data.py first.")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run scripts/export_faiss_data.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 60)
    print("FAISS Index Builder")
    print(f"Store    : {store}")
    print(f"nlist    : {NLIST}")
    print("=" * 60)

    # Load vectors (memory-map to avoid doubling RAM)
    print("\nLoading vectors (memory-mapped) …")
    vectors = np.load(vectors_path, mmap_mode="r")
    n, d = vectors.shape
    print(f"  Shape: {n:,} × {d}")

    # L2-normalise if not already (required for inner-product == cosine)
    if manifest.get("vectors_normalised", False):
        print("  Vectors already L2-normalised — skipping normalisation.")
        # Need a writable copy for FAISS training/adding
        vectors_normed = np.array(vectors, dtype=np.float32)
    else:
        print("  L2-normalising vectors …")
        vectors_normed = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_normed)

    del vectors  # free mmap

    # Build IVF index
    print(f"\nTraining IVF index on {TRAIN_SAMPLE:,} random vectors …")
    rng = np.random.default_rng(42)
    train_idx = rng.choice(n, min(TRAIN_SAMPLE, n), replace=False)
    train_data = vectors_normed[train_idx]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    index.train(train_data)
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")

    print(f"\nAdding {n:,} vectors to index …")
    t0 = time.time()
    # Add in batches to show progress
    batch_size = 100_000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        index.add(vectors_normed[start:end])
        print(f"  {end:,}/{n:,} ({100 * end / n:.0f}%)", end="\r", flush=True)
    add_time = time.time() - t0
    print(f"\n  Added {index.ntotal:,} vectors in {add_time:.1f}s")

    del vectors_normed

    # Set default nprobe
    index.nprobe = config.FAISS_NPROBE
    print(f"\n  Default nprobe: {index.nprobe}")

    # Write index
    print(f"\nWriting {index_path} …")
    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / (1024 ** 2)
    print(f"  Index size: {size_mb:,.0f} MB")

    # Update manifest with index checksum
    print("  Computing SHA-256 …")
    manifest["checksums"]["index.faiss"] = _sha256(index_path)
    manifest["nlist"] = NLIST
    manifest["default_nprobe"] = config.FAISS_NPROBE
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest updated: {manifest_path}")

    print("\n[DONE] FAISS index built successfully.")
    print("Next steps:")
    print("  1. Run tests: .venv/bin/python -m pytest tests/ -v")
    print("  2. Validate: .venv/bin/python scripts/validate_faiss_vs_bq.py")


if __name__ == "__main__":
    main()
