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
import time
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
    parser.add_argument("real_audio_dir", help="Directory containing real reference audio, named <stem of id><ext>")
    parser.add_argument("output_dir", help="Directory to write generated deepfake audio into")
    parser.add_argument(
        "--generators", nargs="+", default=["xtts_v2"], choices=sorted(GENERATORS),
        help="Multiple generators are recommended so detectors can be benchmarked for cross-generator generalization.",
    )
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--manifest", default="fake_manifest.tsv")
    parser.add_argument(
        "--device", default="cpu",
        help="Device for synthesis, e.g. 'cpu' or 'cuda' (GPU). Passed to generators that accept a "
        "device argument (xtts_v2, your_tts). Default 'cpu' is safe everywhere but slow -- pass "
        "'cuda' on a GPU runtime.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate even if the output file already exists. Default: skip already-generated "
        "files, so re-running the same command after an interruption resumes instead of restarting.",
    )
    parser.add_argument(
        "--language", default=None,
        help="Override the language code passed to voice-cloning backends that require one "
        "(xtts_v2, your_tts). Neither backend has native Swahili support, so a closest-supported "
        "language is used as a documented approximation (see docs/METHODOLOGY.md, 'Generator "
        "limitations'). Defaults: xtts_v2='es', your_tts='en'. Applies to all such generators in "
        "--generators for this run.",
    )
    parser.add_argument("--progress-every", type=int, default=50, help="Print a progress line every N files processed.")
    args = parser.parse_args()

    with open(args.subset_tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No rows found in {args.subset_tsv}")

    real_audio_dir = Path(args.real_audio_dir)
    # preprocess_audio.py writes output by stem (e.g. clip1.mp3 -> clip1.wav),
    # since standardization changes the format regardless of the original
    # extension -- id may still carry that original extension, so strip it
    # before appending audio_ext.
    existing_rows = [r for r in rows if (real_audio_dir / f"{Path(r['id']).stem}{args.audio_ext}").is_file()]
    missing_count = len(rows) - len(existing_rows)
    if missing_count:
        print(
            f"Warning: {missing_count}/{len(rows)} selected utterance(s) have no reference audio under "
            f"{real_audio_dir} (e.g. filtered out by preprocess_audio.py's duration/duplicate checks) "
            "-- skipping them rather than failing synthesis partway through."
        )
    rows = existing_rows
    if not rows:
        raise SystemExit(f"No reference audio found under {real_audio_dir} for any row in {args.subset_tsv}")

    output_dir = Path(args.output_dir)
    fake_rows = []
    for gen_name in args.generators:
        generator_kwargs = {"device": args.device}
        if args.language is not None and gen_name in ("xtts_v2", "your_tts"):
            generator_kwargs["language"] = args.language
        generator = GENERATORS[gen_name](**generator_kwargs)
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        generated = skipped = failed = 0
        start_time = time.monotonic()
        for i, row in enumerate(rows, start=1):
            uid = row["id"]
            uid_stem = Path(uid).stem
            ref_path = real_audio_dir / f"{uid_stem}{args.audio_ext}"
            out_path = gen_dir / f"{uid_stem}{args.audio_ext}"

            if out_path.is_file() and not args.overwrite:
                skipped += 1
            else:
                request = SynthesisRequest(
                    utterance_id=uid,
                    text=row["text"],
                    reference_audio_path=str(ref_path),
                    output_path=str(out_path),
                )
                try:
                    generator.synthesize(request)
                    generated += 1
                except Exception as exc:  # noqa: BLE001 - report and continue over a long batch
                    failed += 1
                    print(f"  [{gen_name}] FAILED on {uid}: {exc}")
                    continue

            fake_rows.append(
                {
                    "id": f"{gen_name}_{uid}",
                    "source_utterance_id": uid,
                    "speaker_id": row["speaker_id"],
                    "generator": gen_name,
                    "audio_path": str(out_path),
                }
            )
            if i % args.progress_every == 0 or i == len(rows):
                elapsed = time.monotonic() - start_time
                print(
                    f"  [{gen_name}] {i}/{len(rows)} processed "
                    f"({generated} generated, {skipped} skipped, {failed} failed, {elapsed:.0f}s elapsed)"
                )
        print(f"[{gen_name}] done: {generated} generated, {skipped} skipped (already existed), {failed} failed")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "source_utterance_id", "speaker_id", "generator", "audio_path"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(fake_rows)

    print(f"Generated {len(fake_rows)} deepfake samples (including any resumed/skipped) across {len(args.generators)} generator(s)")


if __name__ == "__main__":
    main()
