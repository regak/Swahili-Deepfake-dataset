# Generators

One synthetic ("fake") clip is produced per real utterance per generator, from
the real utterance's own text and (for cloning backends) its own recording as
the reference. Backends live in `swahili_deepfake_dataset/deepfake_gen.py`
(registry: `GENERATORS`) and are driven by `scripts/generate_deepfakes.py`.

## Comparison

| Key | Model | Swahili | License | Output rate | Status in the released set |
|---|---|---|---|---|---|
| `f5_tts_sw` | F5-TTS, Swahili finetune (`stem-content-ai-project/f5-tts-sw`) | native (finetuned) | verify on the HF repo before redistributing | 24 kHz | complete, 9,702 / 9,702 |
| `chatterbox` | Chatterbox Multilingual (Resemble AI), `language_id="sw"` | listed (1 of 23) | MIT | 24 kHz | complete, 9,702 / 9,702 |
| `voxcpm` | VoxCPM2 (OpenBMB), 2B params | listed (1 of 30) | Apache-2.0 | 48 kHz | in progress |
| `xtts_v2` | Coqui XTTS-v2 | **no** (approximated with `es`) | see model card | not checked | not used |
| `your_tts` | Coqui YourTTS | **no** (approximated with `en`) | see model card | not checked | not used |
| `fish_speech` | Fish Speech | not verified | see model card | not checked | not used |

Real reference clips are 16 kHz mono WAV (`data/processed/`).

Candidates evaluated and **not** integrated:

- **MOSS-TTS-v1.5** (OpenMOSS, Apache-2.0): lists Swahili among 31 languages
  (added in v1.5; v1.0's 20 languages exclude it). About 8B parameters, so it
  needs a large GPU or quantization. Next candidate; no notebook yet.
- **OmniVoice** (k2-fsa): the model card claims 600+ languages but lists none,
  so Swahili support is unconfirmed; code is Apache-2.0 but the weights are
  CC-BY-NC.
- **Fish Audio S2-Pro**: `sw` is in its language list, but the license is a
  research / non-commercial one, not open source.
- **ElevenLabs**: proprietary hosted API.

(Licenses above were read from the model cards on 2026-11-09; re-check before
redistributing any generated audio.)

## Running a generator

Each generator is run with its **own manifest file** — the manifest is
rewritten on every run, so reusing one name would overwrite earlier
generators' rows:

```bash
python scripts/generate_deepfakes.py \
    data/manifests/selected_subset.present.tsv data/processed data/generated \
    --generators chatterbox --device cuda \
    --manifest data/manifests/fake_manifest_chatterbox.tsv
```

Generation is resumable: finished clips are skipped, and files are written via
`<name>.tmp.wav` then renamed, so an interrupted run never leaves a truncated
clip that a resume mistakes for a finished one.

Merge the per-generator manifests before `build_manifest.py` (concatenate the
rows under one header into `fake_manifest_combined.tsv`).

Kaggle versions with checkpointed runs and rclone pushes to Google Drive:
`notebooks/kaggle_chatterbox.ipynb`, `notebooks/kaggle_voxcpm.ipynb`. They
need a Kaggle Secret `RCLONE_CONF` (an rclone remote named `gdrive` with write
scope); no credentials are stored in the notebooks.

## Quality control

File counts do not catch silent or runaway generations. Run
`scripts/qc_generated.py` per generator and report the flag rates.

Chatterbox, random sample of ~1,700 of 9,702 clips (about 17%):

| Check | Share of clips |
|---|---|
| near-silent (RMS < 0.003) | 0.7% |
| clipping (> 0.1% of samples at full scale) | 1.1% |
| much shorter than the real clip (< 0.5x) | 0.8% |
| much longer than the real clip (> 2x) | 4.6% |
| much longer than the real clip (> 3x) | 1.6% |
| trailing silence > 1 s | 10.1% |

Roughly 5-6% of Chatterbox clips look like failed generations (runaway length
or near-silence); Chatterbox logs `forcing EOS token, long_tail=True` when its
alignment analyzer truncates a generation. These flags are heuristics: listen
to a sample of flagged clips before filtering on them, and either exclude
them or keep them and report the rate. Full-set numbers should replace the
sample numbers once computed (Drive reads are ~65 clips/min).

## Known dataset confounds

- **Sample rate:** real 16 kHz, F5-TTS / Chatterbox 24 kHz, VoxCPM2 48 kHz.
  A detector can separate classes by sample rate alone. Resample all audio to
  one rate before training or evaluation and say so in the data card.
- **Duration:** mean duration differs by class (sample: real 6.3 s, F5-TTS
  5.0 s, Chatterbox 6.0 s). Consider duration-matched evaluation subsets.
- **Source-text reuse:** every fake speaks the same sentence as its real
  counterpart, so text content carries no label signal (a feature), but a
  fake and its real source must never be split across train/test; the splits
  are speaker-independent, which already guarantees this.
