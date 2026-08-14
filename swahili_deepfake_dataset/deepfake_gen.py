"""Pluggable interface for deepfake speech generation backends.

The paper's dataset uses multiple synthesis systems so that detectors can be
benchmarked on cross-generator generalization, not just artifacts from a
single TTS model (see docs/METHODOLOGY.md). Each backend lazily imports its
heavy ML dependency so the rest of this package stays lightweight and
testable without installing multi-gigabyte model checkpoints.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisRequest:
    utterance_id: str
    text: str
    reference_audio_path: str
    output_path: str


class DeepfakeGenerator(ABC):
    name: str

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> str:
        """Generate synthetic speech for `request` and return the output audio path."""


class CoquiXTTSGenerator(DeepfakeGenerator):
    """Voice-cloning generator backed by Coqui TTS's XTTS-v2 model.

    XTTS-v2 has no native Swahili ("sw") language support -- its supported
    language list is fixed (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar,
    zh-cn, hu, ko, ja, hi) and passing "sw" raises an error. The closest
    supported language is used as a documented approximation (see
    docs/METHODOLOGY.md, "Generator limitations"), matching the pattern
    already used for YourTTSGenerator below.
    """

    name = "xtts_v2"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cpu",
        language: str = "es",
    ):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "CoquiXTTSGenerator requires the 'TTS' package (or its actively-maintained fork "
                "'coqui-tts' on Python >=3.10, since the original 'TTS' package caps at Python <3.12). "
                "Install with `pip install coqui-tts` (recommended on current Python) or `pip install TTS`."
            ) from exc
        self._tts = TTS(model_name).to(device)
        self._language = language

    def synthesize(self, request: SynthesisRequest) -> str:
        self._tts.tts_to_file(
            text=request.text,
            speaker_wav=request.reference_audio_path,
            language=self._language,
            file_path=request.output_path,
        )
        return request.output_path


class YourTTSGenerator(DeepfakeGenerator):
    """Zero-shot voice-cloning generator backed by Coqui TTS's YourTTS model.

    YourTTS has no native Swahili language embedding; the closest supported
    language code is used as a documented approximation (see
    docs/METHODOLOGY.md, "Generator limitations").
    """

    name = "your_tts"

    def __init__(self, device: str = "cpu", language: str = "en"):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "YourTTSGenerator requires the 'TTS' package (or its actively-maintained fork "
                "'coqui-tts' on Python >=3.10, since the original 'TTS' package caps at Python <3.12). "
                "Install with `pip install coqui-tts` (recommended on current Python) or `pip install TTS`."
            ) from exc
        self._tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
        self._language = language

    def synthesize(self, request: SynthesisRequest) -> str:
        self._tts.tts_to_file(
            text=request.text,
            speaker_wav=request.reference_audio_path,
            language=self._language,
            file_path=request.output_path,
        )
        return request.output_path


_F5_ONES = ["sifuri", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa"]
_F5_TENS = {
    10: "kumi", 20: "ishirini", 30: "thelathini", 40: "arobaini", 50: "hamsini",
    60: "sitini", 70: "sabini", 80: "themanini", 90: "tisini",
}
_F5_SYMBOLS = {
    "*": " nyota ", "#": " alama ya reli ", "/": " kwa ", "+": " jumlisha ",
    "=": " sawa na ", "&": " na ", "@": " at ", "_": " ",
}
_F5_KEEP_RE = re.compile(r"[^A-Za-zÀ-ǿ .,?!'\-]")


def _f5_two(n: int) -> str:
    if n < 10:
        return _F5_ONES[n]
    return _F5_TENS.get(n) or f"{_F5_TENS[n // 10 * 10]} na {_F5_ONES[n % 10]}"


def _f5_three(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts += ["mia", _F5_ONES[hundreds]]
    if rest:
        parts.append(("na " + _f5_two(rest)) if hundreds else _f5_two(rest))
    return " ".join(parts)


def _f5_cardinal(n: int) -> str:
    if n == 0:
        return "sifuri"
    parts = []
    for value, name in [(10**9, "bilioni"), (10**6, "milioni"), (1000, "elfu")]:
        if n >= value:
            quotient, n = divmod(n, value)
            parts.append(f"{name} {_f5_three(quotient)}")
    if n:
        parts.append(("na " + _f5_three(n)) if parts and n < 100 else _f5_three(n))
    return " ".join(parts)


def _f5_digits(s: str) -> str:
    return " ".join(_F5_ONES[int(c)] for c in s if c.isdigit())


def _f5_num(token: str) -> str:
    return _f5_digits(token) if (len(token) >= 5 or token.startswith("0")) else _f5_cardinal(int(token))


def normalize_swahili_text_for_f5tts(text: str) -> str:
    """Spell out digits/symbols as Swahili words for the f5-tts-sw checkpoint.

    The model (stem-content-ai-project/f5-tts-sw) was trained on text where
    numbers are spelled out and only basic punctuation appears -- it never
    saw digit/symbol glyphs, so raw text with digits or symbols degrades
    synthesis quality noticeably. This heuristic normalizer is adapted
    verbatim from the model card's recommended pre-processor; swap it out if
    your input text has different conventions (e.g. currency, dates).
    """
    text = text.strip()
    text = re.sub(
        r"\*[\d*#]*#",
        lambda m: " " + re.sub(r"\d+", lambda d: _f5_digits(d.group()), m.group()) + " ",
        text,
    )
    text = re.sub(r"(\d+)\s*%", lambda m: " asilimia " + _f5_cardinal(int(m.group(1))) + " ", text)
    for sym, word in _F5_SYMBOLS.items():
        text = text.replace(sym, word)
    text = re.sub(r"\d+", lambda m: " " + _f5_num(m.group()) + " ", text)
    text = _F5_KEEP_RE.sub(" ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\s+", " ", text)).strip()
    return text


class F5TTSSwahiliGenerator(DeepfakeGenerator):
    """Zero-shot voice-cloning generator backed by a Swahili-finetuned F5-TTS
    checkpoint (default: stem-content-ai-project/f5-tts-sw on Hugging Face).

    Unlike XTTS-v2/YourTTS above, this checkpoint is finetuned specifically
    on Swahili speech (FLEURS-R sw + Common Voice sw v17, per its model
    card), so it has genuine Swahili pronunciation rather than a
    closest-supported-language approximation. It reports CER 0.029 / WER
    0.202 on held-out Swahili sentences (Whisper-large-v3 sw scoring) at
    time of writing -- see the model card for the full benchmark and
    training-data provenance.

    F5-TTS requires `ref_text`, the transcript of the reference audio,
    unlike XTTS/YourTTS which only need the reference waveform. In this
    pipeline the reference audio for a request is the real recording of
    that same row's text, so `ref_text` and `gen_text` are both
    `request.text` -- we are asking the model to resynthesize the same
    utterance in a cloned voice, matching the real/fake pairing used
    elsewhere in this module.

    Caveats worth checking before citing results from this generator in a
    publication: (1) the model card does not state a license for the
    weights on this page's text alone -- confirm the license field on the
    Hugging Face repo before redistributing outputs; (2) it was partly
    finetuned on Common Voice sw v17, the same corpus family used as this
    project's "real" source data, so if select_subset.py's chosen speakers
    overlap with this checkpoint's finetuning data, this generator may have
    an unfair quality advantage for those specific speakers relative to
    generators with no Swahili training data at all -- worth reporting as a
    limitation, similar in spirit to the XTTS-v2/YourTTS approximation
    caveat above.
    """

    name = "f5_tts_sw"

    def __init__(
        self,
        device: str = "cpu",
        repo_id: str = "stem-content-ai-project/f5-tts-sw",
        nfe_step: int = 32,
    ):
        try:
            from f5_tts.api import F5TTS
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "F5TTSSwahiliGenerator requires the 'f5-tts' and 'huggingface_hub' packages. "
                "Install with `pip install f5-tts huggingface_hub`."
            ) from exc
        ckpt_file = hf_hub_download(repo_id, "model.safetensors")
        vocab_file = hf_hub_download(repo_id, "vocab.txt")
        self._tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_file, vocab_file=vocab_file, device=device)
        # Number of flow-matching sampling steps. Lower = faster, at some
        # cost to audio quality; the model card's benchmark numbers were
        # produced at the library default (32), so treat anything lower as
        # an unvalidated quality/speed tradeoff for your own run.
        self._nfe_step = nfe_step

    def synthesize(self, request: SynthesisRequest) -> str:
        text = normalize_swahili_text_for_f5tts(request.text)
        self._tts.infer(
            ref_file=request.reference_audio_path,
            ref_text=text,
            gen_text=text,
            file_wave=request.output_path,
            nfe_step=self._nfe_step,
        )
        return request.output_path


class FishSpeechGenerator(DeepfakeGenerator):
    """Voice-cloning generator backed by Fish Speech.

    Requires the `fish-speech` package and a downloaded checkpoint; see
    https://github.com/fishaudio/fish-speech for setup instructions.
    """

    name = "fish_speech"

    def __init__(self, checkpoint_dir: str, device: str = "cpu"):
        try:
            from fish_speech.inference import FishSpeechModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FishSpeechGenerator requires the 'fish-speech' package. "
                "See https://github.com/fishaudio/fish-speech for installation."
            ) from exc
        self._model = FishSpeechModel(checkpoint_dir, device=device)

    def synthesize(self, request: SynthesisRequest) -> str:
        self._model.clone_and_synthesize(
            text=request.text,
            reference_audio=request.reference_audio_path,
            output_path=request.output_path,
        )
        return request.output_path


GENERATORS = {
    "xtts_v2": CoquiXTTSGenerator,
    "your_tts": YourTTSGenerator,
    "f5_tts_sw": F5TTSSwahiliGenerator,
    "fish_speech": FishSpeechGenerator,
}
