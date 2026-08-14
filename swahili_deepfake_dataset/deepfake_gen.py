"""Pluggable interface for deepfake speech generation backends.

The paper's dataset uses multiple synthesis systems so that detectors can be
benchmarked on cross-generator generalization, not just artifacts from a
single TTS model (see docs/METHODOLOGY.md). Each backend lazily imports its
heavy ML dependency so the rest of this package stays lightweight and
testable without installing multi-gigabyte model checkpoints.
"""

from __future__ import annotations

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
    """Voice-cloning generator backed by Coqui TTS's XTTS-v2 model."""

    name = "xtts_v2"

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cpu"):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "CoquiXTTSGenerator requires the 'TTS' package (or its actively-maintained fork "
                "'coqui-tts' on Python >=3.10, since the original 'TTS' package caps at Python <3.12). "
                "Install with `pip install coqui-tts` (recommended on current Python) or `pip install TTS`."
            ) from exc
        self._tts = TTS(model_name).to(device)

    def synthesize(self, request: SynthesisRequest) -> str:
        self._tts.tts_to_file(
            text=request.text,
            speaker_wav=request.reference_audio_path,
            language="sw",
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

    def __init__(self, device: str = "cpu"):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "YourTTSGenerator requires the 'TTS' package (or its actively-maintained fork "
                "'coqui-tts' on Python >=3.10, since the original 'TTS' package caps at Python <3.12). "
                "Install with `pip install coqui-tts` (recommended on current Python) or `pip install TTS`."
            ) from exc
        self._tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)

    def synthesize(self, request: SynthesisRequest) -> str:
        self._tts.tts_to_file(
            text=request.text,
            speaker_wav=request.reference_audio_path,
            language="en",
            file_path=request.output_path,
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
    "fish_speech": FishSpeechGenerator,
}
