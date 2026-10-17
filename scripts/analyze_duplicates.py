import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

root = Path.cwd()
manifest_path = root / "data/raw/tinystories_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def digest(text):
    # Hash the exact UTF-8 story text; do not normalize it.
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()

counts = {
    "train": Counter(),
    "validation": Counter(),
}
rows = {"train": 0, "validation": 0}
empty = {"train": 0, "validation": 0}

for entry in manifest["files"]:
    split = entry["split"]
    path = Path(entry["local_path"])

    print(f"Reading {split}: {path.name}", flush=True)

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=8192):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                raise TypeError(f"Expected string in {path}")

            rows[split] += 1
            if not text.strip():
                empty[split] += 1

            counts[split][digest(text)] += 1

train_hashes = set(counts["train"])
validation_hashes = set(counts["validation"])
shared_hashes = train_hashes & validation_hashes

train_duplicate_rows = sum(n - 1 for n in counts["train"].values())
validation_duplicate_rows = sum(
    n - 1 for n in counts["validation"].values()
)
validation_rows_matching_train = sum(
    n for h, n in counts["validation"].items() if h in train_hashes
)
validation_unique_texts_matching_train = len(shared_hashes)

print("\n=== DUPLICATE ANALYSIS ===")
print("Training rows:", f"{rows['train']:,}")
print("Training unique exact texts:", f"{len(train_hashes):,}")
print("Training duplicate rows within split:", f"{train_duplicate_rows:,}")
print("Empty/whitespace-only training rows:", f"{empty['train']:,}")

print("\nValidation rows:", f"{rows['validation']:,}")
print("Validation unique exact texts:", f"{len(validation_hashes):,}")
print("Validation duplicate rows within split:", f"{validation_duplicate_rows:,}")
print("Empty/whitespace-only validation rows:", f"{empty['validation']:,}")

print("\nCross-split overlap:")
print(
    "Validation rows whose text occurs in training:",
    f"{validation_rows_matching_train:,}",
)
print(
    "Distinct validation texts that occur in training:",
    f"{validation_unique_texts_matching_train:,}",
)
print(
    "Distinct validation texts not found in training:",
    f"{len(validation_hashes - train_hashes):,}",
)

print("\nAnalysis complete. No dataset files were modified.")