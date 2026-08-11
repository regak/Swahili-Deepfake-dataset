# Paper Outline

Working outline for a paper built on this dataset and pipeline. Fill in each
section as results become available; delete this line when the paper draft
moves to its own document.

## Working title

"[Dataset name]: A Multi-Corpus, Multi-Generator Benchmark for Swahili Audio
Deepfake Detection"

## Abstract

- Problem: low-resource languages (Swahili) lack public deepfake-detection
  benchmarks.
- Approach: aggregate existing Swahili speech corpora, apply phoneme-coverage-
  aware selection, generate deepfakes with multiple synthesis systems, and
  benchmark detectors for cross-generator/cross-corpus generalization.
- Headline result: [fill in once benchmarking is complete].

## 1. Introduction

- Motivation: rise of voice deepfakes/fraud in Swahili-speaking regions;
  absence of Swahili-specific detection benchmarks.
- Gap: existing deepfake-detection datasets are English/Mandarin/high-resource
  focused; low-resource-language studies (e.g. Urdu) required recording new
  corpora from scratch.
- Contribution: (1) a phoneme-balanced, speaker-diverse Swahili deepfake
  dataset built from existing corpora rather than new recordings, (2) a
  multi-generator benchmark for cross-synthesis generalization, (3) baseline
  detector results.

## 2. Related Work

- Deepfake/synthetic speech detection benchmarks (ASVspoof, In-the-Wild, etc.)
- Low-resource-language speech corpora (Common Voice, FLEURS, KenSpeech)
- Prior low-resource deepfake dataset construction (e.g. Urdu phoneme-balanced
  recording methodology) — contrast with this paper's corpus-reuse approach.

## 3. Dataset Construction (see docs/METHODOLOGY.md for full detail)

- 3.1 Source corpora and licensing
- 3.2 Audio standardization and quality filtering
- 3.3 Phoneme-coverage analysis (grapheme-proxy tokenization)
- 3.4 Greedy phoneme-balanced, speaker-diverse selection algorithm
- 3.5 Deepfake generation (generators used, per-generator sample counts)
- 3.6 Labeling and speaker-independent train/val/test splits
- 3.7 Dataset statistics table (speakers, hours, samples per generator/split)

## 4. Experimental Setup

- Baseline detectors: AASIST, RawNet2, CNN/Transformer baseline(s)
- Training/eval protocol, feature extraction
- Metrics: EER, accuracy, F1, per-generator breakdown

## 5. Results

- 5.1 Overall detection performance per baseline
- 5.2 Cross-generator generalization (train on generator A, test on B)
- 5.3 Phoneme-coverage ablation (balanced selection vs. random sampling)
- 5.4 Error analysis / failure cases

## 6. Discussion

- What phoneme coverage buys you empirically (tie back to 5.3)
- Generator-specific artifacts and detector blind spots
- Limitations (e.g. YourTTS's lack of native Swahili support, corpus
  recording-condition variability)

## 7. Conclusion and Future Work

- Summary of contribution
- Planned dataset release (hosting platform, license)
- Future work: additional generators, human perception study, larger scale

## Reproducibility checklist

- [ ] Dataset manifest (`data/manifests/dataset_manifest.csv`) versioned/released
- [ ] Selection script parameters (target size, per-speaker cap, seed) recorded
- [ ] Per-source corpus licenses documented
- [ ] Speaker-independent split seed fixed and reported
- [ ] Generator versions/checkpoints recorded
