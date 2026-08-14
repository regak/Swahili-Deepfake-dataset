#!/usr/bin/env python3
"""Select a phoneme-balanced, speaker-diverse subset of Swahili utterances
from a corpus metadata file (e.g. Mozilla Common Voice's validated.tsv, or
a Mozilla Data Collective load_dataset() export).

Usage:
    python scripts/select_subset.py data/raw/validated.tsv \
        --target-size 10000 --max-per-speaker 100 \
        --output data/manifests/selected_subset.tsv \
        --report data/manifests/phoneme_coverage_report.json

    # MDC's load_dataset() column layout (audio_path/transcription/speaker_id, comma-separated):
    python scripts/select_subset.py data/raw/mdc_export.csv --schema mdc \
        --target-size 10000 --max-per-speaker 100
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

# Python's csv module defaults to a 128KB field-size cap. A field that big
# is virtually never a genuine transcript -- it almost always means a stray
# unescaped quote or embedded newline desynced the parser's quote-matching,
# causing it to swallow a large chunk of the file into one "field". Raising
# the cap avoids a hard crash on that; load_utterances' returned row counts
# are what actually reveal whether parsing went wrong.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# Column-name/delimiter/quoting presets per corpus export format. Individual
# --id-col/--speaker-col/--text-col/--delimiter flags override these.
#
# quoting=csv.QUOTE_NONE for common_voice is deliberate, not a default:
# Common Voice's validated.tsv is a raw TSV, not an RFC4180-quoted CSV --
# a "sentence" field can contain a literal '"' character (a quoted phrase,
# a curly-quote-like glyph, etc.) that was never meant to function as a CSV
# quote character. Python's csv module defaults to quotechar='"' with
# QUOTE_MINIMAL, so it misreads that literal '"' as the start of a quoted
# field and keeps consuming subsequent tabs/newlines as literal field
# content until it finds another '"' to "close" it -- silently merging
# several rows' data into one row's text field instead of raising an error.
# QUOTE_NONE disables quote-character handling entirely, matching how these
# files are actually formatted. MDC's export is a real pandas-written CSV
# with proper quoting, so it keeps the csv module's default (QUOTE_MINIMAL).
SCHEMA_PRESETS = {
    "common_voice": {
        "id_col": "path", "speaker_col": "client_id", "text_col": "sentence",
        "delimiter": "\t", "quoting": csv.QUOTE_NONE,
    },
    "mdc": {
        "id_col": "audio_path", "speaker_col": "speaker_id", "text_col": "transcription",
        "delimiter": ",", "quoting": csv.QUOTE_MINIMAL,
    },
}

# A genuine Common Voice/MDC sentence is one utterance, typically well under
# 300 characters. Anything far beyond that is essentially always a
# quoting/delimiter desync that glommed multiple rows into one field (see
# SCHEMA_PRESETS above) rather than a real transcript -- reject it instead
# of letting it corrupt phoneme-balance stats and downstream synthesis.
MAX_PLAUSIBLE_TEXT_LENGTH = 500


def load_utterances(
    corpus_path: str, id_col: str, speaker_col: str, text_col: str, delimiter: str, quoting: int = csv.QUOTE_MINIMAL
) -> tuple[list[Utterance], int, int]:
    utterances = []
    total_rows = 0
    oversized = 0
    with open(corpus_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quoting=quoting)
        for row in reader:
            total_rows += 1
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            if len(text) > MAX_PLAUSIBLE_TEXT_LENGTH:
                oversized += 1
                continue
            raw_id = row[id_col]
            # Normalize to a bare filename: MDC's audio_path is a full
            # (often environment-specific) path, while Common Voice's path
            # column is already just a filename -- Path(...).name is a
            # no-op for the latter.
            utterance_id = Path(raw_id).name if raw_id else raw_id
            utterances.append(
                Utterance(id=utterance_id, speaker_id=row.get(speaker_col, "unknown"), text=text)
            )
    return utterances, total_rows, oversized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a phoneme-balanced, speaker-diverse subset for deepfake dataset construction."
    )
    parser.add_argument("corpus_tsv", help="Path to corpus metadata file (e.g. Common Voice validated.tsv, or an MDC export)")
    parser.add_argument(
        "--schema", choices=sorted(SCHEMA_PRESETS), default="common_voice",
        help="Column-name/delimiter preset matching the corpus source. Overridden per-field by --id-col/--speaker-col/--text-col/--delimiter.",
    )
    parser.add_argument("--id-col", default=None, help="Column holding the utterance/audio identifier (overrides --schema)")
    parser.add_argument("--speaker-col", default=None, help="Column holding the speaker identifier (overrides --schema)")
    parser.add_argument("--text-col", default=None, help="Column holding the transcript text (overrides --schema)")
    parser.add_argument("--delimiter", default=None, help="Field delimiter, e.g. ',' or '\\t' (overrides --schema)")
    parser.add_argument(
        "--quote-minimal", action="store_true",
        help="Use standard RFC4180 CSV quote handling (quotechar='\"', QUOTE_MINIMAL) instead of the "
        "schema's preset. Common Voice's raw TSV export needs QUOTE_NONE (the common_voice preset's "
        "default) because literal '\"' characters in sentence text are not CSV quote characters -- "
        "only pass this if your corpus file is genuinely RFC4180-quoted and --schema common_voice "
        "would otherwise misparse it.",
    )
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--max-per-speaker", type=int, default=100)
    parser.add_argument("--speaker-bonus", type=float, default=0.5)
    parser.add_argument("--output", default="selected_subset.tsv")
    parser.add_argument("--report", default="phoneme_coverage_report.json")
    args = parser.parse_args()

    preset = SCHEMA_PRESETS[args.schema]
    id_col = args.id_col or preset["id_col"]
    speaker_col = args.speaker_col or preset["speaker_col"]
    text_col = args.text_col or preset["text_col"]
    delimiter = args.delimiter or preset["delimiter"]
    quoting = csv.QUOTE_MINIMAL if args.quote_minimal else preset["quoting"]

    utterances, total_rows, oversized = load_utterances(args.corpus_tsv, id_col, speaker_col, text_col, delimiter, quoting)
    empty = total_rows - len(utterances) - oversized
    print(
        f"Parsed {total_rows} row(s) from {args.corpus_tsv}; {len(utterances)} usable, "
        f"{empty} skipped (empty transcript), {oversized} skipped (implausibly long -- likely a "
        f"quoting/delimiter desync that merged multiple rows into one field)"
    )
    if total_rows and (empty + oversized) / total_rows > 0.5:
        print(
            "Warning: over half the rows were skipped. For an MDC/Common Voice export this usually "
            "means a delimiter/quoting mismatch (rows got merged or misread), not genuinely empty "
            "transcripts -- double check --delimiter and --schema/--text-col against the actual file "
            "before trusting the selection below."
        )
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
