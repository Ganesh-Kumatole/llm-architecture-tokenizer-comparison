import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path.cwd()
SOURCE_MANIFEST = ROOT / "data/raw/tinystories_manifest.json"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Deterministic test split: approximately 5% of eligible unique stories.
TEST_PERCENT = 5
HASH_MODULUS = 100
SPLIT_SEED_LABEL = "tinystories-project-test-v1"

def story_hash(text: str) -> bytes:
    """Hash exact UTF-8 text; do not normalize story contents."""
    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16,
    ).digest()

def goes_to_test(digest: bytes) -> bool:
    """Assign a story deterministically using its hash and a fixed label."""
    value = hashlib.blake2b(
        SPLIT_SEED_LABEL.encode("utf-8") + digest,
        digest_size=8,
    ).digest()
    return int.from_bytes(value, "big") % HASH_MODULUS < TEST_PERCENT

manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
revision = manifest["revision"]

# Use repository-relative paths, resolved under the local raw dataset folder.
raw_root = ROOT / "data/raw/tinystories"
train_paths = [
    raw_root / entry["repository_path"]
    for entry in manifest["files"]
    if entry["split"] == "train"
]
validation_paths = [
    raw_root / entry["repository_path"]
    for entry in manifest["files"]
    if entry["split"] == "validation"
]

if not train_paths or not validation_paths:
    raise RuntimeError("Manifest must include train and validation files.")

for path in train_paths + validation_paths:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")

def read_text_batches(path):
    """Yield batches of story strings from a local Parquet file."""
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"Missing 'text' column in {path}")
    for batch in parquet.iter_batches(
        columns=["text"],
        batch_size=8192,
    ):
        yield batch.column(0).to_pylist()

def all_stories(paths):
    for path in paths:
        print(f"Reading {path.name}", flush=True)
        for batch in read_text_batches(path):
            for text in batch:
                if not isinstance(text, str):
                    raise TypeError(f"Expected text string in {path}")
                yield text

# Pass 1: preserve the full official validation split and hash its texts.
validation_hashes = set()
validation_rows = 0
validation_empty = 0

print("\nPass 1/3: hashing official validation stories", flush=True)
for text in all_stories(validation_paths):
    validation_rows += 1
    if not text.strip():
        validation_empty += 1
    validation_hashes.add(story_hash(text))

# Pass 2: identify unique, nonempty training stories with no exact
# text overlap against validation.
print("\nPass 2/3: identifying eligible training stories", flush=True)
seen_train_hashes = set()
eligible_hashes = set()
train_rows = 0
train_empty = 0
train_duplicate_rows = 0
train_validation_overlap_rows = 0

for text in all_stories(train_paths):
    train_rows += 1
    if not text.strip():
        train_empty += 1
        continue

    digest = story_hash(text)

    if digest in seen_train_hashes:
        train_duplicate_rows += 1
        continue
    seen_train_hashes.add(digest)

    if digest in validation_hashes:
        train_validation_overlap_rows += 1
        continue

    eligible_hashes.add(digest)

eligible_count = len(eligible_hashes)
if eligible_count < 100:
    raise RuntimeError(
        f"Only {eligible_count} eligible stories; refusing to split."
    )

# Pass 3: write unique eligible training stories to train/test files.
# Remove each digest from eligible_hashes after writing it once, which
# also prevents duplicate copies from entering the output.
print("\nPass 3/3: writing train/test Parquet files", flush=True)

train_output = OUTPUT_DIR / "train.parquet"
test_output = OUTPUT_DIR / "test.parquet"
validation_output = OUTPUT_DIR / "validation.parquet"

schema = pa.schema([("text", pa.string())])
buffer = []
train_written = 0
test_written = 0
written_hashes = set()

def flush(writer, rows):
    if rows:
        table = pa.Table.from_pydict({"text": rows}, schema=schema)
        writer.write_table(table)

try:
    with pq.ParquetWriter(train_output, schema, compression="zstd") as train_writer, \
         pq.ParquetWriter(test_output, schema, compression="zstd") as test_writer:
        for text in all_stories(train_paths):
            if not text.strip():
                continue

            digest = story_hash(text)
            if digest not in eligible_hashes:
                continue

            eligible_hashes.remove(digest)
            buffer.append((digest, text))

            if len(buffer) >= 8192:
                train_batch = []
                test_batch = []
                for item_hash, item_text in buffer:
                    if goes_to_test(item_hash):
                        test_batch.append(item_text)
                        test_written += 1
                    else:
                        train_batch.append(item_text)
                        train_written += 1
                flush(train_writer, train_batch)
                flush(test_writer, test_batch)
                buffer.clear()

        if buffer:
            train_batch = []
            test_batch = []
            for item_hash, item_text in buffer:
                if goes_to_test(item_hash):
                    test_batch.append(item_text)
                    test_written += 1
                else:
                    train_batch.append(item_text)
                    train_written += 1
            flush(train_writer, train_batch)
            flush(test_writer, test_batch)
            buffer.clear()

    # Write the official validation texts unchanged, including any empty
    # rows if they exist. The current audit found none in validation.
    with pq.ParquetWriter(
        validation_output, schema, compression="zstd"
    ) as validation_writer:
        validation_buffer = []
        for text in all_stories(validation_paths):
            validation_buffer.append(text)
            if len(validation_buffer) >= 8192:
                flush(validation_writer, validation_buffer)
                validation_buffer.clear()
        flush(validation_writer, validation_buffer)

except Exception:
    for path in (train_output, test_output, validation_output):
        if path.exists():
            path.unlink()
    raise

if eligible_hashes:
    raise RuntimeError(
        f"{len(eligible_hashes)} eligible story hashes were not written."
    )

# Verify output row counts by reading Parquet metadata.
output_rows = {
    "train": pq.ParquetFile(train_output).metadata.num_rows,
    "test": pq.ParquetFile(test_output).metadata.num_rows,
    "validation": pq.ParquetFile(validation_output).metadata.num_rows,
}

if output_rows["train"] != train_written:
    raise RuntimeError("Train output count does not match written count.")
if output_rows["test"] != test_written:
    raise RuntimeError("Test output count does not match written count.")
if output_rows["validation"] != validation_rows:
    raise RuntimeError("Validation output count does not match source count.")
if train_written + test_written != eligible_count:
    raise RuntimeError("Train + test count differs from eligible count.")

result = {
    "source_repository": manifest["repository"],
    "source_revision": revision,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": {
        "empty_training_records": "excluded",
        "training_exact_duplicates": "keep first occurrence",
        "training_validation_overlap": "excluded from training",
        "validation": "official validation preserved unchanged",
        "test_fraction": TEST_PERCENT / HASH_MODULUS,
        "test_assignment": (
            "deterministic hash assignment; label="
            + SPLIT_SEED_LABEL
        ),
        "text_normalization": "none",
        "hash_algorithm": "BLAKE2b, 128-bit story digest",
    },
    "source_counts": {
        "training_rows": train_rows,
        "training_empty_rows": train_empty,
        "training_duplicate_rows_skipped": train_duplicate_rows,
        "training_unique_texts_overlapping_validation_skipped":
            train_validation_overlap_rows,
        "validation_rows": validation_rows,
        "validation_empty_rows": validation_empty,
    },
    "eligible_unique_training_stories": eligible_count,
    "output_files": {
        "train": "data/processed/train.parquet",
        "test": "data/processed/test.parquet",
        "validation": "data/processed/validation.parquet",
    },
    "output_counts": output_rows,
}

split_manifest_path = OUTPUT_DIR / "split_manifest.json"
split_manifest_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("\n=== SPLIT PREPARATION SUMMARY ===")
print("Source revision:", revision)
print("Training source rows:", f"{train_rows:,}")
print("Empty training rows skipped:", f"{train_empty:,}")
print("Duplicate training rows skipped:", f"{train_duplicate_rows:,}")
print(
    "Training stories matching validation skipped:",
    f"{train_validation_overlap_rows:,}",
)
print("Eligible unique training stories:", f"{eligible_count:,}")
print("Final train stories:", f"{train_written:,}")
print("Final test stories:", f"{test_written:,}")
print("Final validation stories:", f"{validation_rows:,}")
print("Split manifest:", split_manifest_path.relative_to(ROOT))
print("Split preparation completed.")