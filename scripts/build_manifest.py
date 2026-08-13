#!/usr/bin/env python3
"""Combine the selected real subset and generated fakes into a final labeled
manifest with speaker-independent train/val/test splits.

Usage:
    python scripts/build_manifest.py \
        data/manifests/selected_subset.tsv data/manifests/fake_manifest.tsv \
        data/processed --output data/manifests/dataset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swahili_deepfake_dataset.manifest import build_manifest, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final labeled dataset manifest.")
    parser.add_argument("subset_tsv", help="TSV produced by select_subset.py")
    parser.add_argument("fake_manifest_tsv", help="TSV produced by generate_deepfakes.py")
    parser.add_argument("real_audio_dir", help="Directory containing standardized real audio")
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset_manifest.csv")
    parser.add_argument("--summary", default="split_summary.json")
    args = parser.parse_args()

    with open(args.subset_tsv, newline="", encoding="utf-8") as f:
        real_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(args.fake_manifest_tsv, newline="", encoding="utf-8") as f:
        fake_rows = list(csv.DictReader(f, delimiter="\t"))

    entries = build_manifest(
        real_rows,
        fake_rows,
        real_audio_dir=args.real_audio_dir,
        audio_ext=args.audio_ext,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "audio_path", "label", "source", "speaker_id", "split"])
        for e in entries:
            writer.writerow([e.id, e.audio_path, e.label, e.source, e.speaker_id, e.split])

    summary = split_summary(entries)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(entries)} entries to {args.output}")
    for split, counts in summary.items():
        print(f"  {split}: {counts['real']} real, {counts['fake']} fake, {counts['speakers']} speakers")


if __name__ == "__main__":
    main()
