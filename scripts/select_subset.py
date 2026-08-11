#!/usr/bin/env python3
"""Select a phoneme-balanced, speaker-diverse subset of Swahili utterances
from a corpus metadata file (e.g. Mozilla Common Voice's validated.tsv).

Usage:
    python scripts/select_subset.py data/raw/validated.tsv \
        --target-size 10000 --max-per-speaker 100 \
        --output data/manifests/selected_subset.tsv \
        --report data/manifests/phoneme_coverage_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swahili_deepfake_dataset.phonemes import CORE_SWAHILI_GRAPHEMES
from swahili_deepfake_dataset.selection import Utterance, coverage_report, select_balanced_subset


def load_utterances(tsv_path: str, id_col: str, speaker_col: str, text_col: str) -> list[Utterance]:
    utterances = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            utterances.append(
                Utterance(id=row[id_col], speaker_id=row.get(speaker_col, "unknown"), text=text)
            )
    return utterances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a phoneme-balanced, speaker-diverse subset for deepfake dataset construction."
    )
    parser.add_argument("corpus_tsv", help="Path to corpus metadata TSV (e.g. Common Voice validated.tsv)")
    parser.add_argument("--id-col", default="path", help="Column holding the utterance/audio identifier")
    parser.add_argument("--speaker-col", default="client_id", help="Column holding the speaker identifier")
    parser.add_argument("--text-col", default="sentence", help="Column holding the transcript text")
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--max-per-speaker", type=int, default=100)
    parser.add_argument("--speaker-bonus", type=float, default=0.5)
    parser.add_argument("--output", default="selected_subset.tsv")
    parser.add_argument("--report", default="phoneme_coverage_report.json")
    args = parser.parse_args()

    utterances = load_utterances(args.corpus_tsv, args.id_col, args.speaker_col, args.text_col)
    if not utterances:
        raise SystemExit(f"No utterances with non-empty transcripts found in {args.corpus_tsv}")

    result = select_balanced_subset(
        utterances,
        target_size=min(args.target_size, len(utterances)),
        max_per_speaker=args.max_per_speaker,
        speaker_bonus=args.speaker_bonus,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "speaker_id", "text"])
        for u in result.selected:
            writer.writerow([u.id, u.speaker_id, u.text])

    report = coverage_report(result, CORE_SWAHILI_GRAPHEMES)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Candidate pool: {len(utterances)} utterances, {len({u.speaker_id for u in utterances})} speakers")
    print(f"Selected: {len(result.selected)} utterances, {report['num_speakers']} speakers")
    print(f"Core phoneme coverage: {report['core_coverage_pct']:.1f}% ({report['core_covered']}/{report['core_total']})")
    if report["missing_core_phonemes"]:
        print(f"Missing core phonemes: {', '.join(report['missing_core_phonemes'])}")


if __name__ == "__main__":
    main()
