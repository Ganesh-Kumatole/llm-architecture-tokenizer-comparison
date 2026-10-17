import hashlib
import json
import unicodedata
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path.cwd()
MANIFEST_PATH = PROJECT_ROOT / "data/raw/tinystories_manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports/data_audit.json"

# These whitespace characters are normal in text and are not treated
# as suspicious control characters.
ALLOWED_CONTROLS = {"\n", "\r", "\t"}

def length_summary(values):
    """Summarize story lengths without storing Python integers individually."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    lengths = np.frombuffer(values, dtype=np.uint32)
    percentiles = np.percentile(lengths, [50, 90, 95, 99])

    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 3),
        "p50": round(float(percentiles[0]), 3),
        "p90": round(float(percentiles[1]), 3),
        "p95": round(float(percentiles[2]), 3),
        "p99": round(float(percentiles[3]), 3),
    }


def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = manifest["files"]

    # Audit training first, then validation. This makes cross-split
    # duplicate detection deterministic.
    split_order = {"train": 0, "validation": 1}
    files = sorted(
        files,
        key=lambda entry: split_order.get(entry["split"], 99),
    )

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest["repository"],
        "revision": manifest["revision"],
        "manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "files": [],
        "splits": {},
        "checks": {},
        "notes": [
            "Audit only; source data was not modified.",
            "Duplicate detection uses exact raw text, without normalization.",
            "Story lengths are Unicode code-point counts, not token counts.",
            "Control-character checks exclude newline, carriage return, and tab.",
            "A replacement-character check is a warning signal, not proof of bad encoding.",
        ],
    }

    split_stats = {}
    first_seen_split = {}
    total_duplicate_rows = 0
    train_validation_overlap_rows = 0
    total_control_characters = 0
    stories_with_controls = 0
    total_format_characters = 0
    stories_with_format_chars = 0
    total_replacement_chars = 0
    stories_with_replacement_chars = 0

    for entry in files:
        split = entry["split"]
        path = Path(entry["local_path"])

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Parquet file listed in manifest is missing: {path}"
            )

        stats = split_stats.setdefault(
            split,
            {
                "stories_seen": 0,
                "empty_stories": 0,
                "whitespace_only_stories": 0,
                "duplicate_rows": 0,
                "control_characters": 0,
                "stories_with_control_characters": 0,
                "format_characters": 0,
                "stories_with_format_characters": 0,
                "replacement_characters": 0,
                "stories_with_replacement_characters": 0,
                "lengths": array("I"),
            },
        )

        parquet = pq.ParquetFile(path)
        file_rows = 0
        file_empty = 0
        file_whitespace_only = 0
        file_duplicate_rows = 0
        file_controls = 0
        file_format_chars = 0
        file_replacement_chars = 0

        print(f"Auditing {split}: {path.name}", flush=True)

        # Read a small batch at a time to limit RAM use.
        for batch in parquet.iter_batches(
            columns=["text"],
            batch_size=8192,
        ):
            for text in batch.column(0).to_pylist():
                file_rows += 1
                stats["stories_seen"] += 1

                if not isinstance(text, str):
                    raise TypeError(
                        f"Non-string text in {path}, row {file_rows}"
                    )

                char_length = len(text)

                if char_length > 0xFFFFFFFF:
                    raise ValueError("Story is too long for the audit length array")

                stats["lengths"].append(char_length)

                if text == "":
                    file_empty += 1
                    stats["empty_stories"] += 1
                elif not text.strip():
                    file_whitespace_only += 1
                    stats["whitespace_only_stories"] += 1

                # Exact duplicate check on the original text.
                digest = hashlib.blake2b(
                    text.encode("utf-8"),
                    digest_size=16,
                ).digest()

                if digest in first_seen_split:
                    total_duplicate_rows += 1
                    file_duplicate_rows += 1
                    stats["duplicate_rows"] += 1

                    # All training shards are processed before validation.
                    if (
                        split == "validation"
                        and first_seen_split[digest] == "train"
                    ):
                        train_validation_overlap_rows += 1
                else:
                    first_seen_split[digest] = split

                controls = sum(
                    1 for char in text
                    if unicodedata.category(char) == "Cc"
                    and char not in ALLOWED_CONTROLS
                )
                formats = sum(
                    1 for char in text
                    if unicodedata.category(char) == "Cf"
                )
                replacements = text.count("\uFFFD")

                file_controls += controls
                file_format_chars += formats
                file_replacement_chars += replacements

                stats["control_characters"] += controls
                stats["format_characters"] += formats
                stats["replacement_characters"] += replacements

                if controls:
                    stats["stories_with_control_characters"] += 1
                if formats:
                    stats["stories_with_format_characters"] += 1
                if replacements:
                    stats["stories_with_replacement_characters"] += 1

                total_control_characters += controls
                total_format_characters += formats
                total_replacement_chars += replacements

                if controls:
                    stories_with_controls += 1
                if formats:
                    stories_with_format_chars += 1
                if replacements:
                    stories_with_replacement_chars += 1

        metadata_rows = parquet.metadata.num_rows
        expected_rows = entry["rows"]

        file_report = {
            "split": split,
            "repository_path": entry["repository_path"],
            "local_path": str(path),
            "expected_rows_from_manifest": expected_rows,
            "rows_read": file_rows,
            "rows_in_parquet_metadata": metadata_rows,
            "row_count_matches_manifest": (
                file_rows == expected_rows == metadata_rows
            ),
            "size_bytes": path.stat().st_size,
            "empty_stories": file_empty,
            "whitespace_only_stories": file_whitespace_only,
            "duplicate_rows": file_duplicate_rows,
            "nonstandard_control_characters": file_controls,
            "unicode_format_characters": file_format_chars,
            "replacement_characters": file_replacement_chars,
            "lengths": length_summary(stats["lengths"]),
        }

        report["files"].append(file_report)

        print(
            f"  rows={file_rows:,}; "
            f"empty={file_empty}; "
            f"whitespace-only={file_whitespace_only}; "
            f"duplicate rows={file_duplicate_rows}",
            flush=True,
        )

    # Finalize split-level summaries.
    for split, stats in split_stats.items():
        report["splits"][split] = {
            key: value
            for key, value in stats.items()
            if key != "lengths"
        }
        report["splits"][split]["lengths"] = length_summary(
            stats["lengths"]
        )

    report["checks"] = {
        "all_manifest_files_found": True,
        "all_file_row_counts_match": all(
            item["row_count_matches_manifest"]
            for item in report["files"]
        ),
        "total_stories_read": sum(
            stats["stories_seen"] for stats in split_stats.values()
        ),
        "empty_stories": sum(
            stats["empty_stories"] for stats in split_stats.values()
        ),
        "whitespace_only_stories": sum(
            stats["whitespace_only_stories"]
            for stats in split_stats.values()
        ),
        "duplicate_rows_after_first_occurrence": total_duplicate_rows,
        "validation_rows_matching_a_training_story": (
            train_validation_overlap_rows
        ),
        "nonstandard_control_characters": total_control_characters,
        "stories_with_nonstandard_control_characters": stories_with_controls,
        "unicode_format_characters": total_format_characters,
        "stories_with_unicode_format_characters": stories_with_format_chars,
        "replacement_characters": total_replacement_chars,
        "stories_with_replacement_characters": (
            stories_with_replacement_chars
        ),
        "unique_exact_texts": len(first_seen_split),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== DATASET AUDIT SUMMARY ===")
    print("Dataset revision:", manifest["revision"])
    print("Stories read:", f"{report['checks']['total_stories_read']:,}")
    print("Empty stories:", report["checks"]["empty_stories"])
    print("Whitespace-only stories:", report["checks"]["whitespace_only_stories"])
    print(
        "Duplicate rows after first occurrence:",
        f"{total_duplicate_rows:,}",
    )
    print(
        "Validation rows matching training stories:",
        f"{train_validation_overlap_rows:,}",
    )
    print(
        "Stories with nonstandard control characters:",
        stories_with_controls,
    )
    print(
        "Stories with Unicode format characters:",
        stories_with_format_chars,
    )
    print(
        "Stories with replacement characters:",
        stories_with_replacement_chars,
    )

    for split, stats in report["splits"].items():
        print(f"\n{split.upper()} story lengths (characters):")
        print(json.dumps(stats["lengths"], indent=2))

    print("\nAll manifest row counts match:", report["checks"]["all_file_row_counts_match"])
    print("Report saved to:", REPORT_PATH)
    print("Audit completed; no source files were changed.")


if __name__ == "__main__":
    main()
