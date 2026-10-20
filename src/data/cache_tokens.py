"""Stream story-preserving token caches for validation and test.

Each story is encoded as:
    BOS, tokenized story, EOS

The cache stores:
    tokens.uint16.bin       flat little-endian uint16 token IDs
    story_offsets.npy       [start, end) token positions for each story
    story_token_lengths.npy content-token length per story (excludes BOS/EOS)
    story_ids.jsonl         source story ID per row
    meta.json               provenance, hashes, statistics, and data layout

Windows are NOT packed across stories. A downstream loader must keep every
input/target window within the offsets for a single story.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.tokenization import load_tokenizer, load_yaml
from src.tokenization.normalization import NORMALIZATION_VERSION, normalize
from src.tokenization.specials import BOS_ID, EOS_ID, UNK_ID

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_NAMES = ("character", "word", "bpe")
SPLIT_NAMES = ("validation", "test")
UINT16_MAX_VOCAB = 65_536
WRITE_CHUNK_TOKENS = 250_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flush_token_buffer(file_obj, buffer: list[int]) -> None:
    if buffer:
        np.asarray(buffer, dtype="<u2").tofile(file_obj)
        buffer.clear()


def verify_tokenizer_artifacts():
    """Ensure all tokenizers share the fitted training source and policy."""
    train_path = ROOT / "data/processed/train.parquet"
    train_rows = pq.ParquetFile(train_path).metadata.num_rows
    train_hashes = set()

    for name in TOKENIZER_NAMES:
        folder = ROOT / "artifacts/tokenizers" / name
        cfg_path = folder / "config.json"
        prov_path = folder / "provenance.json"

        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        prov = cfg.get("provenance", {})
        if prov_path.is_file():
            prov.update(
                json.loads(prov_path.read_text(encoding="utf-8"))
            )

        if cfg.get("normalization", {}).get("version") != NORMALIZATION_VERSION:
            raise ValueError(f"{name}: normalization version mismatch")

        if prov.get("n_fit_stories") != train_rows:
            raise ValueError(
                f"{name}: expected {train_rows:,} fit stories, "
                f"found {prov.get('n_fit_stories')!r}"
            )

        training_hash = prov.get("train_file_sha256")
        if training_hash:
            train_hashes.add(training_hash)

        tokenizer = load_tokenizer(folder)
        if tokenizer.vocab_size > UINT16_MAX_VOCAB:
            raise ValueError(
                f"{name}: vocab {tokenizer.vocab_size} exceeds uint16 capacity"
            )

        print(
            f"PASS: {name} artifact; vocab={tokenizer.vocab_size:,}; "
            f"fit_stories={prov['n_fit_stories']:,}"
        )

    if len(train_hashes) > 1:
        raise ValueError(
            "Tokenizer provenance has differing training-file hashes."
        )

    return train_rows, next(iter(train_hashes), None)


def cache_split(
    tokenizer_name: str,
    split_name: str,
    output_root: Path,
    max_stories: int | None = None,
) -> Path:
    if tokenizer_name not in TOKENIZER_NAMES:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {split_name}")
    if max_stories is not None and max_stories < 1:
        raise ValueError("max_stories must be positive")

    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    source = ROOT / data_cfg["data"]["splits"][split_name]
    artifact_dir = ROOT / "artifacts/tokenizers" / tokenizer_name

    parquet = pq.ParquetFile(source)
    source_rows = parquet.metadata.num_rows
    target_rows = (
        source_rows if max_stories is None
        else min(source_rows, max_stories)
    )

    if target_rows == 0:
        raise ValueError(f"No stories in {source}")

    tokenizer = load_tokenizer(artifact_dir)
    vocab_size = tokenizer.vocab_size

    if vocab_size > UINT16_MAX_VOCAB:
        raise ValueError(
            f"Vocabulary {vocab_size} cannot be stored as uint16"
        )

    out_dir = output_root / tokenizer_name / split_name
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise FileExistsError(
                f"{out_dir} already contains files. Inspect it first; "
                "this script will not silently overwrite a cache."
            )
        out_dir.rmdir()

    stage = Path(tempfile.mkdtemp(
        prefix=f".{split_name}-build-", dir=out_dir.parent
    ))

    text_column = data_cfg["data"].get("text_column", "text")
    id_column = "id" if "id" in parquet.schema.names else None
    columns = [text_column] + ([id_column] if id_column else [])

    artifact_hashes = {}
    for filename in ("config.json", "vocab.json", "tokenizer.json"):
        path = artifact_dir / filename
        if path.is_file():
            artifact_hashes[filename] = sha256_file(path)

    token_path = stage / "tokens.uint16.bin"
    offsets_path = stage / "story_offsets.npy"
    ids_path = stage / "story_ids.jsonl"

    offsets = np.lib.format.open_memmap(
        offsets_path,
        mode="w+",
        dtype=np.uint64,
        shape=(target_rows, 2),
    )

    seen = 0
    total_tokens = 0
    total_content_tokens = 0
    total_chars = 0
    unknown_content_tokens = 0
    token_buffer: list[int] = []
    progress_interval = 10_000

    try:
        with token_path.open("wb") as token_file, ids_path.open(
            "w", encoding="utf-8"
        ) as id_file:

            for batch in parquet.iter_batches(
                batch_size=1024, columns=columns
            ):
                texts = batch.column(text_column).to_pylist()

                if id_column:
                    story_ids = batch.column(id_column).to_pylist()
                else:
                    story_ids = list(range(seen, seen + len(texts)))

                for story_id, raw_text in zip(story_ids, texts):
                    if seen >= target_rows:
                        break

                    if not isinstance(raw_text, str):
                        raise TypeError(
                            f"Non-string text at story row {seen}"
                        )

                    normalized = normalize(
                        raw_text, tokenizer.repair_mojibake
                    )
                    ids = tokenizer.encode(
                        raw_text, add_bos=True, add_eos=True
                    )

                    if len(ids) < 2 or ids[0] != BOS_ID or ids[-1] != EOS_ID:
                        raise ValueError(
                            f"{tokenizer_name}: BOS/EOS contract failed "
                            f"at row {seen}"
                        )

                    if min(ids) < 0 or max(ids) >= vocab_size:
                        raise ValueError(
                            f"{tokenizer_name}: out-of-range ID at row {seen}"
                        )

                    start = total_tokens
                    total_tokens += len(ids)
                    offsets[seen] = (start, total_tokens)

                    token_buffer.extend(ids)
                    if len(token_buffer) >= WRITE_CHUNK_TOKENS:
                        flush_token_buffer(token_file, token_buffer)

                    total_content_tokens += len(ids) - 2
                    total_chars += len(normalized)
                    unknown_content_tokens += sum(
                        token_id == UNK_ID for token_id in ids[1:-1]
                    )

                    id_file.write(
                        json.dumps(story_id, ensure_ascii=False) + "\n"
                    )
                    seen += 1

                    if seen % progress_interval == 0:
                        print(
                            f"[{tokenizer_name}/{split_name}] "
                            f"{seen:,}/{target_rows:,} stories; "
                            f"{total_tokens:,} tokens",
                            flush=True,
                        )

                if seen >= target_rows:
                    break

            flush_token_buffer(token_file, token_buffer)
            token_file.flush()

        if seen != target_rows:
            raise RuntimeError(
                f"Expected {target_rows} stories, processed {seen}"
            )

        offsets.flush()
        del offsets

        if token_path.stat().st_size != total_tokens * 2:
            raise RuntimeError("Token file size does not match token count")

        offsets_read = np.load(offsets_path, mmap_mode="r")
        lengths = (
            offsets_read[:, 1] - offsets_read[:, 0] - 2
        ).astype(np.uint32)

        if np.any(offsets_read[:, 1] <= offsets_read[:, 0]):
            raise RuntimeError("Invalid story offsets")

        np.save(stage / "story_token_lengths.npy", lengths)

        meta = {
            "format_version": 1,
            "tokenizer": tokenizer_name,
            "split": split_name,
            "source_file": str(source.relative_to(ROOT)),
            "source_sha256": sha256_file(source),
            "tokenizer_artifact_hashes": artifact_hashes,
            "normalization_version": NORMALIZATION_VERSION,
            "vocab_size": vocab_size,
            "num_source_stories": source_rows,
            "num_stories": seen,
            "is_complete_split": seen == source_rows,
            "num_tokens_including_bos_eos": total_tokens,
            "num_content_tokens": total_content_tokens,
            "num_characters_after_normalization": total_chars,
            "unknown_content_token_rate": (
                unknown_content_tokens / total_content_tokens
                if total_content_tokens else None
            ),
            "chars_per_content_token": (
                total_chars / total_content_tokens
                if total_content_tokens else None
            ),
            "sequence_length_p50": float(np.percentile(lengths, 50)),
            "sequence_length_p90": float(np.percentile(lengths, 90)),
            "sequence_length_p99": float(np.percentile(lengths, 99)),
            "sequence_length_max": int(lengths.max()),
            "dtype": "little-endian uint16",
            "offset_convention": "[start, end)",
            "token_layout": "BOS + story tokens + EOS; stories remain separate",
            "max_context_tokens": data_cfg["data"]["max_context_tokens"],
            "training_window_rule": (
                "Construct windows within each story offset only. "
                "Use a target mask for padding/tail windows; never join "
                "the end of one story to the beginning of another."
            ),
        }

        (stage / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Atomic publish: only publish a complete cache.
        stage.rename(out_dir)

        print(f"\nPASS: saved {tokenizer_name}/{split_name}")
        print(f"Stories: {seen:,}; tokens incl. BOS/EOS: {total_tokens:,}")
        print(f"Cache: {out_dir}")
        return out_dir

    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", choices=TOKENIZER_NAMES, required=True
    )
    parser.add_argument(
        "--split", choices=SPLIT_NAMES, required=True
    )
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--output-root", default="data/tokenized")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    cache_split(
        args.tokenizer,
        args.split,
        output_root=output_root,
        max_stories=args.max_stories,
    )


if __name__ == "__main__":
    main()
