import hashlib
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path.cwd()
FILES = {
    "train": ROOT / "data/processed/train.parquet",
    "test": ROOT / "data/processed/test.parquet",
    "validation": ROOT / "data/processed/validation.parquet",
}

def story_hash(text):
    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16,
    ).digest()

def collect_hashes(path, split):
    hashes = set()
    rows = 0
    empty = 0
    duplicates = 0

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=8192):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                raise TypeError(f"Non-string story in {path}")

            rows += 1
            if not text.strip():
                empty += 1

            digest = story_hash(text)
            if digest in hashes:
                duplicates += 1
            hashes.add(digest)

    print(
        f"{split:10} rows={rows:,} "
        f"unique_texts={len(hashes):,} "
        f"duplicate_rows={duplicates:,} empty={empty:,}"
    )
    return hashes, rows, duplicates, empty

all_hashes = {}
for split, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"\nChecking {split}: {path.name}", flush=True)
    all_hashes[split] = collect_hashes(path, split)[0]

train = all_hashes["train"]
test = all_hashes["test"]
validation = all_hashes["validation"]

overlaps = {
    "train_test": len(train & test),
    "train_validation": len(train & validation),
    "test_validation": len(test & validation),
}

print("\n=== CROSS-SPLIT OVERLAP ===")
for name, count in overlaps.items():
    print(f"{name}: {count:,}")

if any(overlaps.values()):
    raise RuntimeError("Exact-text overlap detected across final splits.")

print("\nSUCCESS: final splits are exact-text-disjoint.")