"""Audio standardization and quality filtering.

Standardizes arbitrary input audio to 16 kHz mono PCM16 WAV, and provides
quality filters (duration bounds, exact-duplicate detection) recommended
before phoneme-coverage selection, so the selection pipeline sees only
usable candidate utterances.

Requires the optional `soundfile` and `scipy` dependencies
(`pip install -r requirements.txt`).
"""

from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path
from typing import Iterable, List, Tuple


def standardize_audio(input_path: str, output_path: str, target_sr: int = 16000) -> str:
    """Resample/downmix `input_path` to mono `target_sr` Hz PCM16 WAV."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(input_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        common = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // common, sr // common)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, data, target_sr, subtype="PCM_16")
    return output_path


def duration_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def passes_duration_filter(path: str, min_duration: float = 2.0, max_duration: float = 20.0) -> bool:
    dur = duration_seconds(path)
    return min_duration <= dur <= max_duration


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(paths: Iterable[str]) -> List[Tuple[str, str]]:
    """Return (duplicate_path, original_path) pairs for byte-identical files."""
    seen: dict = {}
    duplicates: List[Tuple[str, str]] = []
    for path in paths:
        digest = file_sha256(path)
        if digest in seen:
            duplicates.append((path, seen[digest]))
        else:
            seen[digest] = path
    return duplicates
