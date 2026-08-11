"""Phoneme-balanced, speaker-diverse utterance selection.

Implements a lazy-greedy approximation to a weighted set-cover objective:
iteratively pick the utterance with the highest marginal gain, where gain
rewards covering rare phonemes (inverse-frequency weighting), gives
diminishing returns on phonemes already well represented, and adds a bonus
for introducing a speaker not yet in the selected set. A per-speaker cap
prevents a handful of prolific speakers from dominating the dataset.

See docs/METHODOLOGY.md for the full rationale.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from .phonemes import tokenize_text


@dataclass(frozen=True)
class Utterance:
    id: str
    speaker_id: str
    text: str


@dataclass
class SelectionResult:
    selected: List[Utterance] = field(default_factory=list)
    covered_phonemes: Counter = field(default_factory=Counter)
    speaker_counts: Counter = field(default_factory=Counter)


def select_balanced_subset(
    utterances: List[Utterance],
    target_size: int,
    max_per_speaker: Optional[int] = None,
    speaker_bonus: float = 0.5,
    rare_weight_power: float = 0.5,
) -> SelectionResult:
    """Greedily select up to `target_size` utterances maximizing phoneme
    coverage/balance under a per-speaker cap.

    Args:
        utterances: Candidate pool.
        target_size: Maximum number of utterances to select.
        max_per_speaker: Optional cap on utterances contributed by any one
            speaker. None disables the cap.
        speaker_bonus: Flat bonus added the first time a speaker is selected,
            to encourage speaker diversity alongside phoneme coverage.
        rare_weight_power: Exponent applied to inverse phoneme frequency.
            Higher values prioritize rare phonemes more aggressively.
    """
    if target_size <= 0 or not utterances:
        return SelectionResult()

    by_id: Dict[str, Utterance] = {u.id: u for u in utterances}
    phoneme_multisets: Dict[str, Counter] = {
        u.id: Counter(tokenize_text(u.text)) for u in utterances
    }

    global_freq: Counter = Counter()
    for counts in phoneme_multisets.values():
        global_freq.update(counts)
    if not global_freq:
        return SelectionResult()

    weight = {p: 1.0 / (freq ** rare_weight_power) for p, freq in global_freq.items()}

    covered: Counter = Counter()
    speaker_counts: Counter = Counter()
    selected: List[Utterance] = []
    remaining = {u.id for u in utterances}

    def gain(uid: str) -> float:
        u = by_id[uid]
        if max_per_speaker is not None and speaker_counts[u.speaker_id] >= max_per_speaker:
            return float("-inf")
        g = sum(
            weight[p] * count / (1 + covered[p])
            for p, count in phoneme_multisets[uid].items()
        )
        if speaker_counts[u.speaker_id] == 0:
            g += speaker_bonus
        return g

    heap = [(-gain(uid), uid) for uid in remaining]
    heapq.heapify(heap)

    while heap and len(selected) < target_size:
        _, uid = heapq.heappop(heap)
        if uid not in remaining:
            continue
        current_gain = gain(uid)
        if current_gain == float("-inf"):
            remaining.discard(uid)
            continue
        if not heap or current_gain >= -heap[0][0]:
            selected.append(by_id[uid])
            remaining.discard(uid)
            covered.update(phoneme_multisets[uid])
            speaker_counts[by_id[uid].speaker_id] += 1
        else:
            heapq.heappush(heap, (-current_gain, uid))

    return SelectionResult(selected=selected, covered_phonemes=covered, speaker_counts=speaker_counts)


def coverage_report(result: SelectionResult, core_inventory: FrozenSet[str]) -> dict:
    """Summarize how well `result` covers a target phoneme inventory."""
    covered_core = {p for p in core_inventory if result.covered_phonemes.get(p, 0) > 0}
    core_total = len(core_inventory)
    return {
        "num_selected": len(result.selected),
        "num_speakers": len(result.speaker_counts),
        "core_total": core_total,
        "core_covered": len(covered_core),
        "core_coverage_pct": (100.0 * len(covered_core) / core_total) if core_total else 0.0,
        "missing_core_phonemes": sorted(set(core_inventory) - covered_core),
        "phoneme_counts": dict(result.covered_phonemes),
        "speaker_utterance_counts": dict(result.speaker_counts),
    }
