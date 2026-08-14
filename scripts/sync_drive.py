#!/usr/bin/env python3
"""Sync data/ (real audio, generated deepfakes, manifests) between the local
disk and a Google Drive folder, so work survives a Colab session ending.

Colab's local disk (/content/...) is wiped when a session closes or times
out; generate_deepfakes.py's outputs are otherwise lost unless copied
somewhere durable first. Writing directly to a Drive-mounted path instead
is simpler but noticeably slower for a batch of many small files (Drive's
FUSE mount adds real per-file latency), so this script instead keeps the
pipeline's usual local paths as the fast working copy and syncs to/from a
Drive folder on demand.

Requires Google Drive already mounted (this cannot be done from a plain
script -- it needs the Colab-specific API, run once per session in a
notebook cell):

    from google.colab import drive
    drive.mount('/content/drive')

Usage:
    # Start of a session: pull back whatever was already generated, so
    # generate_deepfakes.py's skip-existing-files resume logic sees it.
    python scripts/sync_drive.py restore

    # Periodically during a long run (e.g. in a separate cell), or once
    # right before intentionally closing the session:
    python scripts/sync_drive.py backup

    # Keep syncing local -> Drive every 5 minutes until interrupted (run in
    # its own cell alongside generate_deepfakes.py running in another):
    python scripts/sync_drive.py watch --interval 300
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/swahili-deepfake-dataset/data")


def run_rsync(src: Path, dst: Path) -> None:
    if shutil.which("rsync") is None:
        raise SystemExit(
            "rsync not found. Colab images normally include it; if missing, install with "
            "`apt-get install -y rsync`."
        )
    dst.mkdir(parents=True, exist_ok=True)
    # Trailing slash on src copies its *contents* into dst, not the
    # directory itself -- matters for repeated syncs landing in the same place.
    subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=True)


def count_files(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file()) if directory.is_dir() else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync data/ between local disk and a Google Drive folder to survive Colab session resets."
    )
    parser.add_argument("mode", choices=["restore", "backup", "watch"], help="restore: Drive -> local. backup: local -> Drive. watch: repeat backup on an interval.")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Local data directory (default: the repo's data/)")
    parser.add_argument("--drive-dir", default=str(DEFAULT_DRIVE_DIR), help="Drive-mounted destination directory")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between syncs in watch mode (default 300 = 5 min)")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    drive_dir = Path(args.drive_dir)

    if args.mode in ("backup", "watch") and not Path("/content/drive").is_dir():
        raise SystemExit(
            "Google Drive doesn't appear to be mounted at /content/drive. Run this in a notebook cell "
            "first:\n  from google.colab import drive\n  drive.mount('/content/drive')"
        )

    if args.mode == "restore":
        before = count_files(local_dir)
        run_rsync(drive_dir, local_dir)
        after = count_files(local_dir)
        print(f"Restored from {drive_dir}: {before} -> {after} local file(s)")

    elif args.mode == "backup":
        run_rsync(local_dir, drive_dir)
        print(f"Backed up {count_files(local_dir)} local file(s) to {drive_dir}")

    elif args.mode == "watch":
        print(f"Syncing {local_dir} -> {drive_dir} every {args.interval}s. Interrupt (stop the cell) to end.")
        try:
            while True:
                run_rsync(local_dir, drive_dir)
                print(f"  synced {count_files(local_dir)} file(s) at {time.strftime('%H:%M:%S')}", flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
