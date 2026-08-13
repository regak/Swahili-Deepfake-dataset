#!/usr/bin/env python3
"""Generate deepfake speech for a selected subset of real Swahili utterances,
using one or more TTS/voice-cloning backends.

Usage:
    python scripts/generate_deepfakes.py \
        data/manifests/selected_subset.tsv data/processed data/generated \
        --generators xtts_v2 your_tts \
        --manifest data/manifests/fake_manifest.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swahili_deepfake_dataset.deepfake_gen import GENERATORS, SynthesisRequest

# Python's csv module defaults to a 128KB field-size cap. A field that big
# is virtually never genuine data -- it almost always means a stray
# unescaped quote or embedded newline upstream desynced the parser's
# quote-matching, causing it to swallow a large chunk of the file into one
# "field". See select_subset.py for the same fix and fuller rationale.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deepfake speech for a selected subset of real Swahili utterances."
    )
    parser.add_argument("subset_tsv", help="TSV produced by select_subset.py (id, speaker_id, text)")
    parser.add_argument("real_audio_dir", help="Directory containing real reference audio, named <id><ext>")
    parser.add_argument("output_dir", help="Directory to write generated deepfake audio into")
    parser.add_argument(
        "--generators", nargs="+", default=["xtts_v2"], choices=sorted(GENERATORS),
        help="Multiple generators are recommended so detectors can be benchmarked for cross-generator generalization.",
    )
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--manifest", default="fake_manifest.tsv")
    args = parser.parse_args()

    with open(args.subset_tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No rows found in {args.subset_tsv}")

    output_dir = Path(args.output_dir)
    fake_rows = []
    for gen_name in args.generators:
        generator = GENERATORS[gen_name]()
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            uid = row["id"]
            ref_path = Path(args.real_audio_dir) / f"{uid}{args.audio_ext}"
            out_path = gen_dir / f"{uid}{args.audio_ext}"
            request = SynthesisRequest(
                utterance_id=uid,
                text=row["text"],
                reference_audio_path=str(ref_path),
                output_path=str(out_path),
            )
            generator.synthesize(request)
            fake_rows.append(
                {
                    "id": f"{gen_name}_{uid}",
                    "source_utterance_id": uid,
                    "speaker_id": row["speaker_id"],
                    "generator": gen_name,
                    "audio_path": str(out_path),
                }
            )

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "source_utterance_id", "speaker_id", "generator", "audio_path"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(fake_rows)

    print(f"Generated {len(fake_rows)} deepfake samples across {len(args.generators)} generator(s)")


if __name__ == "__main__":
    main()
