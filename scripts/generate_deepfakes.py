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

# A genuine utterance's text is one sentence, typically well under 300
# characters. A selected_subset.tsv row far beyond that is almost always a
# leftover from a csv quoting/delimiter desync in select_subset.py's
# upstream corpus parsing (fixed there, but pre-existing subset files may
# still carry corrupted rows) -- reject it here too rather than feeding a
# garbage multi-row blob into a generator's text-to-speech/alignment step.
MAX_PLAUSIBLE_TEXT_LENGTH = 500

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
    parser.add_argument(
        "--nfe-step", type=int, default=None,
        help="Override the number of flow-matching sampling steps for f5_tts_sw (default in the "
        "underlying model is 32). Lower values (e.g. 16) roughly halve synthesis time at some cost "
        "to audio quality -- useful for shortening long batches. Ignored by other generators.",
    )
    args = parser.parse_args()

    with open(args.subset_tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No rows found in {args.subset_tsv}")

    oversized_rows = [r for r in rows if len(r["text"]) > MAX_PLAUSIBLE_TEXT_LENGTH]
    if oversized_rows:
        print(
            f"Warning: {len(oversized_rows)}/{len(rows)} row(s) in {args.subset_tsv} have an implausibly "
            f"long text field (>{MAX_PLAUSIBLE_TEXT_LENGTH} chars) -- almost certainly a leftover csv "
            "quoting/delimiter desync from select_subset.py's corpus parsing, not a genuine transcript "
            "(see select_subset.py's QUOTE_NONE fix). Skipping them; re-running select_subset.py on the "
            "raw corpus is recommended so phoneme-balance stats and speaker selection aren't skewed by "
            f"these rows too, e.g.: {[r['id'] for r in oversized_rows[:5]]}"
        )
        rows = [r for r in rows if len(r["text"]) <= MAX_PLAUSIBLE_TEXT_LENGTH]
        if not rows:
            raise SystemExit(f"No usable rows left in {args.subset_tsv} after filtering out oversized text fields")

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
        if args.nfe_step is not None and gen_name == "f5_tts_sw":
            generator_kwargs["nfe_step"] = args.nfe_step
        generator = GENERATORS[gen_name](**generator_kwargs)
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Clean up any partial output left by a previous run that was
        # interrupted mid-write (e.g. a Colab disconnect) -- see the atomic
        # rename below for why these can only be incomplete, never a file
        # that was ever mistaken for a finished output.
        for stale_tmp in gen_dir.glob("*.tmp"):
            stale_tmp.unlink()
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
                # Write to a temp path and rename into place only once
                # synthesize() has fully succeeded, so an interruption mid-
                # write (Colab disconnect, OOM kill, ...) can never leave
                # behind a truncated file that a later resume mistakes for
                # a completed one -- out_path.is_file() only becomes true
                # after the file is actually whole.
                tmp_path = out_path.with_name(out_path.name + ".tmp")
                request = SynthesisRequest(
                    utterance_id=uid,
                    text=row["text"],
                    reference_audio_path=str(ref_path),
                    output_path=str(tmp_path),
                )
                try:
                    generator.synthesize(request)
                    tmp_path.replace(out_path)
                    generated += 1
                except Exception as exc:  # noqa: BLE001 - report and continue over a long batch
                    failed += 1
                    tmp_path.unlink(missing_ok=True)
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
