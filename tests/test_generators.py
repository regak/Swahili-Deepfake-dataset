"""Registry / interface tests for the generator backends.

These must run without any heavy TTS dependency installed: each backend lazily
imports its model package inside __init__, so importing the module and
inspecting GENERATORS is always safe.
"""

import pytest

from swahili_deepfake_dataset.deepfake_gen import GENERATORS, DeepfakeGenerator, SynthesisRequest


def test_registry_contains_expected_backends():
    assert {"xtts_v2", "your_tts", "f5_tts_sw", "fish_speech", "chatterbox", "voxcpm"} <= set(GENERATORS)


def test_registry_keys_match_class_names():
    for key, cls in GENERATORS.items():
        assert issubclass(cls, DeepfakeGenerator)
        assert cls.name == key, f"{cls.__name__}.name={cls.name!r} but registered as {key!r}"


@pytest.mark.parametrize("name,package", [("chatterbox", "chatterbox-tts"), ("voxcpm", "voxcpm")])
def test_missing_dependency_gives_actionable_error(name, package, monkeypatch):
    """Without the model package installed the constructor must raise ImportError naming the pip package."""
    import builtins

    real_import = builtins.__import__

    def blocked(mod, *a, **k):
        if mod.split(".")[0] in ("chatterbox", "voxcpm"):
            raise ImportError(mod)
        return real_import(mod, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=package):
        GENERATORS[name]()


def test_synthesis_request_is_immutable():
    req = SynthesisRequest("u1", "habari", "ref.wav", "out.wav")
    with pytest.raises(Exception):
        req.text = "changed"  # type: ignore[misc]
