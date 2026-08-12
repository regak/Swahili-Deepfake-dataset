# Swahili-Deepfake-dataset

A multi-corpus, multi-generator Swahili audio deepfake detection dataset and
construction pipeline.

Rather than recording a new corpus from scratch, this project aggregates
existing public Swahili speech corpora (e.g. Mozilla Common Voice, FLEURS),
selects a phoneme-balanced and speaker-diverse subset, generates synthetic
("deepfake") counterparts with multiple TTS/voice-cloning systems, and
produces a labeled real/fake dataset with speaker-independent splits for
training and benchmarking detection models.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full methodology and
[`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md) for the paper this dataset
supports.

## Repository structure

```
swahili_deepfake_dataset/   Core pipeline logic (no heavy ML deps required)
  phonemes.py                Swahili grapheme-phoneme tokenizer + frequency counting
  selection.py                Greedy phoneme-coverage + speaker-diversity selection
  manifest.py                  Real/fake labeling + speaker-independent train/val/test splits
  audio_preprocess.py          Audio standardization (16 kHz mono WAV) and quality filtering
  deepfake_gen.py               Pluggable interface for TTS/voice-cloning backends

scripts/                    CLI entry points
  select_subset.py            Corpus metadata -> phoneme-balanced subset
  preprocess_audio.py         Raw audio -> standardized, filtered WAV
  generate_deepfakes.py       Selected subset -> synthetic speech (per generator)
  build_manifest.py            Real + fake -> final labeled dataset manifest

tests/                      Unit tests for the core pipeline logic
docs/                       Methodology and paper outline
notebooks/                  Kaggle/Jupyter notebooks (e.g. corpus download)
data/                       Not committed to git (see .gitignore); local working directories
  raw/         Downloaded corpus audio + metadata
  processed/   Standardized/filtered real audio
  generated/   Synthesized deepfake audio, per generator
  manifests/   Selection outputs, coverage reports, final dataset manifest
```

## Setup

```bash
pip install -r requirements.txt
```

Audio preprocessing requires `soundfile`/`scipy` (included above). Deepfake
generation requires an additional heavy dependency depending on the backend
(e.g. `pip install TTS` for XTTS-v2/YourTTS) — install only the backend(s)
you plan to use; see `requirements.txt` for details.

## Pipeline usage

1. **Obtain a source corpus.** Download Common Voice Swahili (or another
   corpus with a `path`/`sentence`/speaker-ID metadata table) into
   `data/raw/`. If you're sourcing it from Mozilla Data Collective (MDC)
   rather than the classic Common Voice downloads page, run
   [`notebooks/download_mdc_common_voice_sw.ipynb`](notebooks/download_mdc_common_voice_sw.ipynb)
   on Kaggle (with Internet enabled and an `MDC_API_KEY` Kaggle Secret set)
   to fetch and extract it there, then copy the resulting `clips/` and
   `validated.tsv` into `data/raw/`.

2. **Select a phoneme-balanced subset:**

   ```bash
   python scripts/select_subset.py data/raw/validated.tsv \
       --target-size 10000 --max-per-speaker 100 \
       --output data/manifests/selected_subset.tsv \
       --report data/manifests/phoneme_coverage_report.json
   ```

3. **Standardize and filter the corresponding real audio:**

   ```bash
   python scripts/preprocess_audio.py data/raw/clips data/processed \
       --min-duration 2.0 --max-duration 20.0
   ```

4. **Generate deepfakes** with one or more backends:

   ```bash
   python scripts/generate_deepfakes.py \
       data/manifests/selected_subset.tsv data/processed data/generated \
       --generators xtts_v2 your_tts \
       --manifest data/manifests/fake_manifest.tsv
   ```

5. **Build the final labeled manifest** with speaker-independent splits:

   ```bash
   python scripts/build_manifest.py \
       data/manifests/selected_subset.tsv data/manifests/fake_manifest.tsv \
       data/processed --output data/manifests/dataset_manifest.csv
   ```

`data/manifests/dataset_manifest.csv` is then ready for training/evaluating
detection models (e.g. AASIST, RawNet2, CNN/Transformer baselines).

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The core selection/labeling logic (`phonemes.py`, `selection.py`,
`manifest.py`) has no external dependencies and is fully unit tested.

## Dataset targets

| Item | Target |
|---|---|
| Speakers | 200+ |
| Real utterances | 10,000+ |
| Deepfake generators | 2–4 |
| Fake utterances | 10,000–30,000 |
| Total audio duration | 20–50 hours |

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the rationale behind
these targets and the selection methodology.
