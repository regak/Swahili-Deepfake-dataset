"""Quality-control pass over generated deepfake audio.

For every `<generated_dir>/<name>.wav`, reads the matching real reference
`<real_audio_dir>/<name>.wav` header and reports per-clip metrics, then prints
aggregate flag rates. Intended to catch generation failures (silence, runaway
"babbling", truncation) that a plain file-count check cannot, and to produce
reproducible numbers for the paper.

Per-clip metrics: duration, sample rate, RMS, peak, fraction of clipped
samples, leading/trailing silence (|x| < 0.01), and the duration ratio
generated / real.

Flags (thresholds are CLI-tunable; they are heuristics, not ground truth --
listen to a sample of flagged clips before filtering on them):
  near_silent   RMS below --min-rms
  too_short     duration ratio below --min-ratio
  too_long      duration ratio above --max-ratio (runaway generation)
  clipping      more than --max-clip of samples at full scale
  long_trail    trailing silence above --max-trail seconds
  nonfinite     NaN/inf samples

Example:
    python scripts/qc_generated.py data/generated/chatterbox data/processed \
        --sample 1500 --output data/manifests/qc_chatterbox.csv

Reading many files from a Google Drive FUSE mount is slow (~65 clips/min
observed); use --sample there, or run against a local copy.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def analyse(gen_path: Path, real_dir: Path, args) -> dict:
    import numpy as np
    import soundfile as sf

    row = {"name": gen_path.name}
    try:
        info = sf.info(str(gen_path))
        x, sr = sf.read(str(gen_path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        dur = len(x) / sr
        row.update(dur=dur, sr=sr)
        real = real_dir / gen_path.name
        if real.is_file():
            ri = sf.info(str(real))
            row["real_dur"] = ri.frames / ri.samplerate
            row["ratio"] = dur / row["real_dur"] if row["real_dur"] else float("nan")
        finite = bool(np.isfinite(x).all())
        row["nonfinite"] = not finite
        if not finite:
            x = np.nan_to_num(x)
        ax = np.abs(x)
        row["rms"] = float(np.sqrt((x**2).mean() + 1e-12))
        row["peak"] = float(ax.max()) if len(ax) else 0.0
        row["clip_frac"] = float((ax >= 0.999).mean()) if len(ax) else 0.0
        nz = np.where(ax > 0.01)[0]
        row["lead_s"] = (nz[0] / sr) if len(nz) else dur
        row["trail_s"] = ((len(x) - nz[-1]) / sr) if len(nz) else dur
        row["channels"] = info.channels
        flags = []
        if row["rms"] < args.min_rms:
            flags.append("near_silent")
        if "ratio" in row and row["ratio"] < args.min_ratio:
            flags.append("too_short")
        if "ratio" in row and row["ratio"] > args.max_ratio:
            flags.append("too_long")
        if row["clip_frac"] > args.max_clip:
            flags.append("clipping")
        if row["trail_s"] > args.max_trail:
            flags.append("long_trail")
        if row["nonfinite"]:
            flags.append("nonfinite")
        row["flags"] = "|".join(flags)
    except Exception as exc:  # noqa: BLE001 - report unreadable files, keep going
        row["error"] = repr(exc)[:200]
        row["flags"] = "unreadable"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("generated_dir")
    p.add_argument("real_audio_dir")
    p.add_argument("--output", default=None, help="Write per-clip metrics to this CSV.")
    p.add_argument("--sample", type=int, default=None, help="Analyse a random sample of this many clips.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-rms", type=float, default=0.003)
    p.add_argument("--min-ratio", type=float, default=0.5)
    p.add_argument("--max-ratio", type=float, default=2.0)
    p.add_argument("--max-clip", type=float, default=0.001)
    p.add_argument("--max-trail", type=float, default=1.0)
    args = p.parse_args()

    gen_dir, real_dir = Path(args.generated_dir), Path(args.real_audio_dir)
    files = sorted(f for f in gen_dir.glob("*.wav") if ".tmp." not in f.name)
    if not files:
        raise SystemExit(f"No .wav files under {gen_dir}")
    if args.sample and args.sample < len(files):
        random.Random(args.seed).shuffle(files)
        files = sorted(files[: args.sample])
    print(f"Analysing {len(files)} clip(s) from {gen_dir} ...", file=sys.stderr)

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda f: analyse(f, real_dir, args), files))

    n = len(rows)
    flag_names = ["near_silent", "too_short", "too_long", "clipping", "long_trail", "nonfinite", "unreadable"]
    print(f"\nclips analysed: {n}")
    for fn in flag_names:
        c = sum(fn in r["flags"].split("|") for r in rows)
        print(f"  {fn:12s} {c:6d}  ({100 * c / n:.2f}%)")
    any_flag = sum(1 for r in rows if r["flags"] and set(r["flags"].split("|")) - {"long_trail"})
    print(f"  any flag other than long_trail: {any_flag} ({100 * any_flag / n:.2f}%)")
    srs = sorted({r.get("sr") for r in rows if r.get("sr")})
    print(f"  sample rates seen: {srs}")

    if args.output:
        cols = ["name", "dur", "real_dur", "ratio", "sr", "channels", "rms", "peak", "clip_frac",
                "lead_s", "trail_s", "nonfinite", "flags", "error"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"per-clip metrics -> {args.output}")


if __name__ == "__main__":
    main()
