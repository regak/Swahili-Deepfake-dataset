"""Grapheme-based phoneme tokenization for Swahili text.

Swahili orthography is highly phonetic, so a grapheme-level tokenizer is a
practical proxy for a full phoneme transcription (see docs/METHODOLOGY.md,
"Option A: Use Graphemes as a Proxy"). Multi-character graphemes (digraphs,
the prenasalized-consonant trigraph "ng'", etc.) are matched greedily before
falling back to single letters, so e.g. "ng'ombe" tokenizes as
["ng'", "o", "m", "b", "e"] rather than as individual letters.

This is an approximation, not a full IPA transcription: labialized clusters
such as "kw"/"mw"/"gw" are tokenized as two separate consonant phonemes
rather than a single labialized phoneme. A grapheme-to-phoneme (G2P) model
could replace this module without changing the selection algorithm's
interface (see swahili_deepfake_dataset.selection).
"""

from __future__ import annotations

from collections import Counter
from typing import List

# Multi-character graphemes, longest first so greedy matching prefers them
# over their single-character prefixes (e.g. "ng'" before "ng" before "n").
_MULTI_CHAR_GRAPHEMES = sorted(
    {
        "ng'",
        "ch",
        "dh",
        "gh",
        "kh",
        "ng",
        "ny",
        "sh",
        "th",
        "mb",
        "mp",
        "nd",
        "nt",
        "nz",
        "nj",
        "ts",
    },
    key=len,
    reverse=True,
)

_SINGLE_CHAR_GRAPHEMES = frozenset("abcdefghijklmnopqrstuvwxyz")

# The minimal set of Swahili sounds recommended for coverage tracking
# (see docs/METHODOLOGY.md, "Option A"): the five vowels plus the digraph/
# trigraph consonants that are easy to under-sample by accident.
CORE_SWAHILI_GRAPHEMES = frozenset(
    {"a", "e", "i", "o", "u", "ch", "dh", "gh", "ng", "ng'", "ny", "sh", "th"}
)


def tokenize_word(word: str) -> List[str]:
    """Tokenize a single Swahili word into a list of grapheme-phoneme tokens."""
    cleaned = "".join(ch for ch in word.lower() if ch.isalpha() or ch == "'")
    tokens: List[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        for grapheme in _MULTI_CHAR_GRAPHEMES:
            if cleaned.startswith(grapheme, i):
                tokens.append(grapheme)
                i += len(grapheme)
                break
        else:
            ch = cleaned[i]
            if ch in _SINGLE_CHAR_GRAPHEMES:
                tokens.append(ch)
            i += 1
    return tokens


def tokenize_text(text: str) -> List[str]:
    """Tokenize a full utterance into a flat list of grapheme-phoneme tokens."""
    tokens: List[str] = []
    for word in text.split():
        tokens.extend(tokenize_word(word))
    return tokens


def phoneme_frequencies(texts: List[str]) -> Counter:
    """Compute grapheme-phoneme frequencies across a collection of utterances."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))
    return counter
