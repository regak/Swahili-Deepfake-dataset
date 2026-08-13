from swahili_deepfake_dataset.selection import Utterance, coverage_report, select_balanced_subset


def make_corpus():
    # "a" is a very common vowel; "ng'" and "th" are deliberately rare,
    # each appearing in only one utterance.
    return [
        Utterance(id="u1", speaker_id="s1", text="baba baba baba"),
        Utterance(id="u2", speaker_id="s1", text="baba baba"),
        Utterance(id="u3", speaker_id="s2", text="baba"),
        Utterance(id="u4", speaker_id="s3", text="ng'ombe"),
        Utterance(id="u5", speaker_id="s4", text="thamani"),
        Utterance(id="u6", speaker_id="s5", text="baba baba baba baba"),
    ]


def test_selection_prioritizes_rare_phonemes():
    corpus = make_corpus()
    result = select_balanced_subset(corpus, target_size=2, max_per_speaker=None, speaker_bonus=0.0)
    selected_ids = {u.id for u in result.selected}
    # The two rare-phoneme utterances should be picked well before the
    # fifth repetitive "baba"-only utterance.
    assert "u4" in selected_ids
    assert "u5" in selected_ids


def test_selection_respects_target_size():
    corpus = make_corpus()
    result = select_balanced_subset(corpus, target_size=3)
    assert len(result.selected) == 3


def test_selection_respects_max_per_speaker():
    corpus = [
        Utterance(id=f"u{i}", speaker_id="only_speaker", text="baba mtoto chakula")
        for i in range(10)
    ]
    result = select_balanced_subset(corpus, target_size=10, max_per_speaker=3)
    assert len(result.selected) == 3


def test_selection_handles_empty_corpus():
    result = select_balanced_subset([], target_size=5)
    assert result.selected == []


def test_coverage_report_reports_missing_core_phonemes():
    corpus = [Utterance(id="u1", speaker_id="s1", text="baba mtoto")]
    result = select_balanced_subset(corpus, target_size=1)
    core = frozenset({"a", "o", "ng'"})
    report = coverage_report(result, core)
    assert report["core_covered"] == 2  # "a" and "o" present, "ng'" is not
    assert report["missing_core_phonemes"] == ["ng'"]
