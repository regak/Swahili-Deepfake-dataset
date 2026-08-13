#!/usr/bin/env python3
"""Standardize a directory of raw audio to 16 kHz mono WAV and filter by
duration and exact duplicates.

Usage:
    python scripts/preprocess_audio.py data/raw/clips data/processed \
        --min-duration 2.0 --max-duration 20.0

    # Only process the files listed in a selected_subset.tsv, leaving the
    # rest of a large corpus directory untouched (no copying needed):
    python scripts/preprocess_audio.py \
        /path/to/full/corpus/clips data/processed \
        --subset data/manifests/selected_subset.tsv \
        --min-duration 2.0 --max-duration 20.0
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swahili_deepfake_dataset.audio_preprocess import (
    duration_seconds,
    find_duplicates,
    standardize_audio,
)

# Python's csv module defaults to a 128KB field-size cap. A field that big
# is virtually never genuine data -- it almost always means a stray
# unescaped quote or embedded newline upstream desynced the parser's
# quote-matching, causing it to swallow a large chunk of the file into one
# "field". See select_subset.py for the same fix and fuller rationale.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize and quality-filter raw audio.")
    parser.add_argument("input_dir", help="Directory of raw audio files")
    parser.add_argument("output_dir", help="Directory to write standardized WAV files")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional TSV with an 'id' column (e.g. selected_subset.tsv). If given, only "
        "files directly under input_dir matching an id are read and standardized -- the rest "
        "of input_dir (e.g. a full corpus of hundreds of thousands of clips) is never touched, "
        "so nothing needs to be copied out of it first.",
    )
    parser.add_argument("--manifest", default="preprocess_manifest.tsv")
    parser.add_argument(
        "--filtered-out-list",
        default="filtered_out.txt",
        help="Plain-text file listing the source filename, duration, and reason for every filtered-out clip.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subset:
        with open(args.subset, newline="", encoding="utf-8") as f:
            wanted = {row["id"] for row in csv.DictReader(f, delimiter="\t")}
        raw_files = []
        missing = []
        for wanted_id in sorted(wanted):
            candidate = input_dir / wanted_id
            if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.is_file():
                raw_files.append(candidate)
            else:
                missing.append(wanted_id)
        if missing:
            print(
                f"Warning: {len(missing)}/{len(wanted)} requested id(s) not found directly "
                f"under {input_dir} (expects a flat clips directory), e.g.: {missing[:5]}"
            )
    else:
        raw_files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not raw_files:
        raise SystemExit(f"No audio files found under {input_dir}")

    # Filtered-out files are deleted, not just flagged, so output_dir only
    # ever contains audio that actually passed quality filtering -- anything
    # downstream that lists output_dir (or checks a file's existence there,
    # like build_manifest.py) sees the correct, already-filtered set without
    # needing to separately consult this script's manifest.
    rows = []
    standardized_paths = []
    for raw_path in raw_files:
        out_path = output_dir / f"{raw_path.stem}.wav"
        try:
            standardize_audio(str(raw_path), str(out_path), target_sr=args.target_sr)
            dur = duration_seconds(str(out_path))
            kept = args.min_duration <= dur <= args.max_duration
            reason = "" if kept else "duration_out_of_range"
        except Exception as exc:  # noqa: BLE001 - report and continue over a large batch
            dur = 0.0
            kept = False
            reason = f"error:{exc}"
        if not kept and out_path.exists():
            out_path.unlink()
        rows.append({
            "id": raw_path.stem,
            "source_filename": raw_path.name,
            "path": str(out_path),
            "duration": f"{dur:.3f}",
            "kept": kept,
            "reason": reason,
        })
        if kept:
            standardized_paths.append(str(out_path))

    duplicates = find_duplicates(standardized_paths)
    duplicate_paths = {dup for dup, _original in duplicates}
    for row in rows:
        if row["path"] in duplicate_paths and row["kept"]:
            row["kept"] = False
            row["reason"] = "duplicate"
            Path(row["path"]).unlink(missing_ok=True)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "source_filename", "path", "duration", "kept", "reason"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    filtered_rows = [row for row in rows if not row["kept"]]
    with open(args.filtered_out_list, "w", encoding="utf-8") as f:
        f.write("filename\tduration_seconds\treason\n")
        for row in filtered_rows:
            f.write(f"{row['source_filename']}\t{row['duration']}\t{row['reason']}\n")

    kept_count = sum(1 for row in rows if row["kept"])
    print(f"Processed {len(rows)} files: {kept_count} kept, {len(rows) - kept_count} filtered out")

    reason_counts = Counter(row["reason"].split(":", 1)[0] for row in rows if not row["kept"])
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")
    if filtered_rows:
        print(f"Filtered-out filenames and durations written to {args.filtered_out_list}")


if __name__ == "__main__":
    main()
