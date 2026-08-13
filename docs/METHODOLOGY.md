# Dataset Construction Methodology

This document describes the methodology used to construct the Swahili audio
deepfake dataset in this repository, for use in the paper's Dataset /
Methodology section.

## 1. Rationale: corpus-based construction over recording from scratch

Unlike prior low-resource-language deepfake studies that recorded a new
phoneme-balanced corpus from scratch (e.g. the Urdu deepfake dataset), Swahili
already has large, publicly available, permissively licensed speech corpora.
Recording thousands of new utterances is costly and unnecessary when existing
corpora already exceed the scale required for a robust benchmark. Instead,
this project treats **dataset construction as a selection and optimization
problem** over existing corpora: aggregate candidate speech, then select a
subset that is phonetically representative and speaker-diverse, rather than
using a corpus as-is or sampling randomly.

## 2. Candidate corpora

| Source | Scale | Notes |
|---|---|---|
| Mozilla Common Voice (Swahili) | ~1,064 hours, ~1,518 speakers, CC0 | Primary source: largest, most speaker-diverse, open license |
| FLEURS (Swahili) | ~10–12 hours, speaker-independent splits provided | Clean benchmark-style supplementary source |
| KenSpeech | ~27.5 hours, 26 speakers | Supplementary only — too few speakers alone |

Common Voice Swahili alone exceeds the scale needed for a publication-grade
benchmark; FLEURS/KenSpeech are used to diversify recording conditions and
accents rather than as primary volume sources.

## 3. Pipeline

```
1. Aggregate candidate corpora (metadata: transcript, speaker ID, audio path)
                    |
2. Clean and standardize audio (16 kHz mono WAV; filter noise/short/duplicate clips)
                    |
3. Phoneme-coverage analysis of transcripts
                    |
4. Greedy phoneme-balanced, speaker-diverse subset selection
                    |
5. Generate deepfakes with multiple synthesis systems (XTTS-v2, YourTTS, Fish Speech, ...)
                    |
6. Label real/fake and assign speaker-independent train/val/test splits
                    |
7. Benchmark detection models (AASIST, RawNet2, CNN/Transformer baselines)
```

Each stage corresponds to a module in `swahili_deepfake_dataset/` and a CLI
script in `scripts/` (see the top-level README for usage).

## 4. Phoneme coverage analysis

### Why it matters

Randomly sampling utterances from a large corpus skews toward common words
and sounds. A detector trained on such a skewed sample may learn frequent
phonetic patterns rather than general synthesis artifacts, and may
generalize poorly to speech containing under-represented sounds — exactly
where a synthesis system is more likely to fail unnaturally. Prioritizing
phonetic coverage during selection is intended to expose the detector to a
representative — not just abundant — sample of Swahili phonetics.

### Grapheme-based proxy tokenization

Swahili orthography is highly phonetic, so `swahili_deepfake_dataset/phonemes.py`
uses a **grapheme-level proxy** instead of a full IPA transcription: text is
greedily tokenized into single letters and known multi-character graphemes
(`ch`, `dh`, `gh`, `kh`, `ng`, `ng'`, `ny`, `sh`, `th`, and prenasalized
clusters `mb`, `mp`, `nd`, `nt`, `nz`, `nj`, `ts`), longest match first. This
is an approximation — labialized clusters (`kw`, `mw`, `gw`, ...) are
tokenized as separate consonant phonemes — and could be replaced with a
grapheme-to-phoneme (G2P) model without changing the selection algorithm's
interface.

The **core coverage target** tracked for reporting is the minimal set of
sounds most likely to be accidentally under-sampled: the five vowels plus
the digraph/trigraph consonants (`CORE_SWAHILI_GRAPHEMES` in `phonemes.py`).

### Selection algorithm

`swahili_deepfake_dataset/selection.py` implements a **lazy-greedy
approximation to a weighted set-cover objective** (`select_balanced_subset`):

- Each candidate utterance is scored by the marginal gain it contributes:
  `sum over phonemes p in the utterance of weight(p) * count(p) / (1 + already_covered(p))`,
  where `weight(p) = 1 / frequency(p)^0.5` down-weights common phonemes so
  rare ones are prioritized, and the `1 + already_covered(p)` term gives
  diminishing returns once a phoneme is already well represented (favoring
  balance, not just one-time coverage).
- A flat bonus is added the first time a candidate's speaker is selected, to
  encourage speaker diversity alongside phonetic diversity.
- An optional per-speaker cap prevents a handful of prolific speakers from
  dominating the selected set.
- Selection proceeds greedily (highest marginal gain first, using a lazy
  priority queue for efficiency) until a target dataset size is reached.

This produces a defensible, reproducible answer to "why these recordings?":
utterances were selected to maximize phonetic coverage and balance under a
speaker-diversity constraint, not sampled arbitrarily.

## 5. Deepfake generation

Multiple synthesis/voice-cloning systems are used deliberately (see
`swahili_deepfake_dataset/deepfake_gen.py`), because a detector trained
against only one generator's artifacts risks learning generator-specific
quirks rather than general synthetic-speech artifacts. Supported backends:

- **XTTS-v2** (Coqui TTS) — has native Swahili (`sw`) language support.
- **YourTTS** (Coqui TTS) — zero-shot voice cloning; no native Swahili
  language embedding, so the closest supported language code is used as a
  documented approximation. This limitation should be reported explicitly
  in any published results using this backend.
- **Fish Speech** — voice cloning via reference audio.

Each real utterance selected in Stage 4 is used as the source text (and, for
voice-cloning backends, the reference audio) for one synthetic counterpart
per generator, producing multiple fake variants per real utterance across
generators.

## 6. Labeling and splits

`swahili_deepfake_dataset/manifest.py` assigns each **speaker** — not each
utterance — to exactly one of train/val/test (default 70/15/15), so no
speaker's real or synthetic speech appears in more than one split. This
enforces **speaker-independent evaluation**: a detector cannot succeed by
memorizing speaker identity rather than learning synthesis artifacts, which
is the evaluation standard used in published deepfake-detection benchmarks.

## 7. Benchmarking (downstream, not implemented in this repo)

The constructed dataset is intended to support the paper's central
research question:

> How well do deepfake detectors generalize across multiple Swahili speech
> sources and multiple synthesis techniques?

Candidate baseline detectors for comparison: AASIST, RawNet2, and
CNN/Transformer-based classifiers, evaluated per-generator and pooled
across generators to measure cross-generator generalization.

## 8. Target dataset scale

| Item | Target |
|---|---|
| Speakers | 200+ |
| Real utterances | 10,000+ |
| Deepfake generators | 2–4 |
| Fake utterances | 10,000–30,000 |
| Total audio duration | 20–50 hours |

This scale is comparable to published low-resource-language deepfake
detection datasets and is intended to support a paper-level contribution
(dataset + generalization benchmark), not just a proof-of-concept.

## 9. Licensing

Common Voice is released under CC0; FLEURS and other candidate corpora
should have their license terms checked and recorded per-source before
inclusion, and cited accordingly in any publication or dataset release.
