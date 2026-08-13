from swahili_deepfake_dataset.manifest import assign_speaker_splits, build_manifest, split_summary


def test_assign_speaker_splits_proportions_and_determinism():
    speakers = [f"spk{i}" for i in range(100)]
    splits_a = assign_speaker_splits(speakers, train_ratio=0.7, val_ratio=0.15, seed=42)
    splits_b = assign_speaker_splits(speakers, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert splits_a == splits_b  # deterministic given the same seed

    counts = {"train": 0, "val": 0, "test": 0}
    for split in splits_a.values():
        counts[split] += 1
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15


def test_assign_speaker_splits_different_seed_can_differ():
    speakers = [f"spk{i}" for i in range(50)]
    splits_a = assign_speaker_splits(speakers, seed=1)
    splits_b = assign_speaker_splits(speakers, seed=2)
    assert splits_a != splits_b


def test_build_manifest_keeps_speaker_in_single_split():
    real_rows = [
        {"id": "u1", "speaker_id": "s1"},
        {"id": "u2", "speaker_id": "s1"},
        {"id": "u3", "speaker_id": "s2"},
    ]
    fake_rows = [
        {"id": "xtts_v2_u1", "speaker_id": "s1", "generator": "xtts_v2", "audio_path": "/fake/u1.wav"},
        {"id": "xtts_v2_u3", "speaker_id": "s2", "generator": "xtts_v2", "audio_path": "/fake/u3.wav"},
    ]
    entries = build_manifest(real_rows, fake_rows, real_audio_dir="/real")

    by_speaker = {}
    for e in entries:
        by_speaker.setdefault(e.speaker_id, set()).add(e.split)
    for speaker, splits in by_speaker.items():
        assert len(splits) == 1, f"speaker {speaker} appears in multiple splits: {splits}"


def test_build_manifest_labels_real_and_fake_correctly():
    real_rows = [{"id": "u1", "speaker_id": "s1"}]
    fake_rows = [{"id": "xtts_v2_u1", "speaker_id": "s1", "generator": "xtts_v2", "audio_path": "/fake/u1.wav"}]
    entries = build_manifest(real_rows, fake_rows, real_audio_dir="/real", audio_ext=".wav")

    real_entry = next(e for e in entries if e.label == "real")
    fake_entry = next(e for e in entries if e.label == "fake")
    assert real_entry.audio_path == "/real/u1.wav"
    assert real_entry.source == "corpus"
    assert fake_entry.audio_path == "/fake/u1.wav"
    assert fake_entry.source == "xtts_v2"


def test_split_summary_counts():
    real_rows = [{"id": f"u{i}", "speaker_id": f"s{i}"} for i in range(10)]
    fake_rows = [
        {"id": f"xtts_v2_u{i}", "speaker_id": f"s{i}", "generator": "xtts_v2", "audio_path": f"/fake/u{i}.wav"}
        for i in range(10)
    ]
    entries = build_manifest(real_rows, fake_rows, real_audio_dir="/real", train_ratio=0.7, val_ratio=0.15)
    summary = split_summary(entries)
    total_real = sum(s["real"] for s in summary.values())
    total_fake = sum(s["fake"] for s in summary.values())
    assert total_real == 10
    assert total_fake == 10
