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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swahili_deepfake_dataset.audio_preprocess import (
    duration_seconds,
    find_duplicates,
    standardize_audio,
)

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
        rows.append({"id": raw_path.stem, "path": str(out_path), "duration": f"{dur:.3f}", "kept": kept, "reason": reason})
        if kept:
            standardized_paths.append(str(out_path))

    duplicates = find_duplicates(standardized_paths)
    duplicate_paths = {dup for dup, _original in duplicates}
    for row in rows:
        if row["path"] in duplicate_paths and row["kept"]:
            row["kept"] = False
            row["reason"] = "duplicate"

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "duration", "kept", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    kept_count = sum(1 for row in rows if row["kept"])
    print(f"Processed {len(rows)} files: {kept_count} kept, {len(rows) - kept_count} filtered out")
    print(f"Duplicates removed: {len(duplicates)}")


if __name__ == "__main__":
    main()
