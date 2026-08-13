#!/usr/bin/env python3
"""Extract only the audio clips referenced in a selected_subset.tsv from a
corpus archive, instead of extracting the entire archive.

Full Common Voice releases are large enough (tens of GB of audio) that
extracting everything before selection is often infeasible on disk-limited
environments (e.g. Kaggle notebooks). Since select_subset.py only needs the
corpus's transcript TSV (not audio) to pick a subset, full audio extraction
can be deferred until after selection, and then limited to just the
selected utterances -- typically a few GB instead of the whole corpus.

Usage:
    python scripts/extract_selected_clips.py \
        data/raw/cv-corpus-26.0-2026-06-12-sw.tar.gz \
        data/manifests/selected_subset.tsv \
        data/raw/clips
"""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract only the audio clips referenced in a selected_subset.tsv from a tar(.gz) archive."
    )
    parser.add_argument("archive_path", help="Path to the corpus archive (.tar or .tar.gz)")
    parser.add_argument(
        "subset_tsv",
        help="TSV produced by select_subset.py (must have an 'id' column matching clip filenames in the archive)",
    )
    parser.add_argument("output_dir", help="Directory to extract the selected clips into")
    args = parser.parse_args()

    with open(args.subset_tsv, newline="", encoding="utf-8") as f:
        wanted = {row["id"] for row in csv.DictReader(f, delimiter="\t")}
    if not wanted:
        raise SystemExit(f"No ids found in {args.subset_tsv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    remaining = set(wanted)
    found = 0
    # Iterating the TarFile object streams through the archive sequentially
    # rather than building an in-memory index of every member up front
    # (getmembers()), which matters for corpora with hundreds of thousands
    # of entries.
    with tarfile.open(args.archive_path) as tar:
        for member in tar:
            if not remaining:
                break
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in remaining:
                continue
            member.name = name  # flatten any archive subdirectory structure
            tar.extract(member, path=output_dir)
            remaining.discard(name)
            found += 1

    print(f"Extracted {found}/{len(wanted)} requested clips to {output_dir}")
    if remaining:
        print(f"Warning: {len(remaining)} requested clip(s) were not found in the archive, e.g.: "
              f"{sorted(remaining)[:5]}")


if __name__ == "__main__":
    main()
