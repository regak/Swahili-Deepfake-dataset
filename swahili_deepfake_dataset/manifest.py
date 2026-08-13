"""Build the labeled real/fake dataset manifest with speaker-independent splits.

A speaker's utterances (real and every synthetic/cloned version derived from
them) are always assigned to a single split, so no speaker appears in both
train and test — required for a meaningful evaluation of detector
generalization rather than speaker memorization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    audio_path: str
    label: str  # "real" or "fake"
    source: str  # e.g. "common_voice", "xtts_v2", "your_tts"
    speaker_id: str
    split: str  # "train", "val", or "test"


def assign_speaker_splits(
    speaker_ids: Iterable[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Deterministically assign each speaker to train/val/test."""
    unique_speakers = sorted(set(speaker_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_speakers)

    n = len(unique_speakers)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    splits: Dict[str, str] = {}
    for i, speaker in enumerate(unique_speakers):
        if i < n_train:
            splits[speaker] = "train"
        elif i < n_train + n_val:
            splits[speaker] = "val"
        else:
            splits[speaker] = "test"
    return splits


def build_manifest(
    real_rows: List[dict],
    fake_rows: List[dict],
    real_audio_dir: str,
    audio_ext: str = ".wav",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> List[ManifestEntry]:
    """Combine real and fake rows into a single labeled, split manifest.

    Args:
        real_rows: dicts with keys `id`, `speaker_id` (from select_subset.py output).
        fake_rows: dicts with keys `id`, `speaker_id`, `generator`, `audio_path`
            (from generate_deepfakes.py output).
        real_audio_dir: directory containing real audio named `<id><audio_ext>`.
    """
    all_speakers = [row["speaker_id"] for row in real_rows]
    splits = assign_speaker_splits(all_speakers, train_ratio, val_ratio, seed)

    entries: List[ManifestEntry] = []
    for row in real_rows:
        speaker = row["speaker_id"]
        entries.append(
            ManifestEntry(
                id=row["id"],
                audio_path=f"{real_audio_dir.rstrip('/')}/{row['id']}{audio_ext}",
                label="real",
                source="corpus",
                speaker_id=speaker,
                split=splits.get(speaker, "train"),
            )
        )
    for row in fake_rows:
        speaker = row["speaker_id"]
        entries.append(
            ManifestEntry(
                id=row["id"],
                audio_path=row["audio_path"],
                label="fake",
                source=row["generator"],
                speaker_id=speaker,
                split=splits.get(speaker, "train"),
            )
        )
    return entries


def split_summary(entries: List[ManifestEntry]) -> Dict[str, Dict[str, int]]:
    """Count real/fake/speaker totals per split, for reporting in the paper."""
    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_entries = [e for e in entries if e.split == split]
        summary[split] = {
            "real": sum(1 for e in split_entries if e.label == "real"),
            "fake": sum(1 for e in split_entries if e.label == "fake"),
            "speakers": len({e.speaker_id for e in split_entries}),
        }
    return summary
