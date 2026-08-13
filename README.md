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
  extract_selected_clips.py    Corpus archive + selected subset -> just those clips (skips full extraction)
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
   `data/raw/`. Full corpora are large (tens of GB of audio) — extracting
   everything before selection is usually unnecessary and, on disk-limited
   environments, often won't fit at all.

   If you're sourcing it from Mozilla Data Collective (MDC) rather than the
   classic Common Voice downloads page,
   [`notebooks/download_mdc_common_voice_sw.ipynb`](notebooks/download_mdc_common_voice_sw.ipynb)
   runs the download-through-selection flow on Kaggle *without ever writing
   the full archive to disk* — the Common Voice Swahili archive (~21 GB) is
   larger than a typical Kaggle session's writable quota, so the notebook
   calls the MDC API directly for a signed URL and stream-extracts from it:
   one pass keeps only the transcript TSVs, then `select_subset.py` (step 2
   below) runs there, then a second streaming pass keeps only the selected
   clips — never the full corpus. Copy its resulting `selected_subset.tsv`
   and `clips/` output into `data/manifests/` and `data/raw/` respectively
   and skip to step 3.

2. **Select a phoneme-balanced subset** (skip if you already did this via
   the notebook above):

   ```bash
   python scripts/select_subset.py data/raw/validated.tsv \
       --target-size 10000 --max-per-speaker 100 \
       --output data/manifests/selected_subset.tsv \
       --report data/manifests/phoneme_coverage_report.json
   ```

   If the metadata came from `datacollective`'s `load_dataset()` (columns
   `audio_path`/`transcription`/`speaker_id`, comma-separated) instead of a
   classic Common Voice `validated.tsv`, save that DataFrame to a file and
   pass `--schema mdc`:

   ```bash
   python scripts/select_subset.py data/raw/mdc_export.csv --schema mdc \
       --target-size 10000 --max-per-speaker 100
   ```

   If you have the full archive locally and only want the selected clips'
   audio (not the whole corpus), extract just those:

   ```bash
   python scripts/extract_selected_clips.py \
       data/raw/cv-corpus-26.0-2026-06-12-sw.tar.gz \
       data/manifests/selected_subset.tsv \
       data/raw/clips
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
