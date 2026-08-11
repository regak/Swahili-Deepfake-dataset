from swahili_deepfake_dataset.phonemes import (
    CORE_SWAHILI_GRAPHEMES,
    phoneme_frequencies,
    tokenize_text,
    tokenize_word,
)


def test_tokenize_word_simple():
    assert tokenize_word("mtoto") == ["m", "t", "o", "t", "o"]


def test_tokenize_word_digraph():
    assert tokenize_word("chakula") == ["ch", "a", "k", "u", "l", "a"]


def test_tokenize_word_prenasalized():
    assert tokenize_word("ngoma") == ["ng", "o", "m", "a"]


def test_tokenize_word_apostrophe_trigraph():
    # "mb" is itself a tracked prenasalized-consonant grapheme (like "nd"/"ng"),
    # so it is matched as one token rather than "m" + "b".
    assert tokenize_word("ng'ombe") == ["ng'", "o", "mb", "e"]


def test_tokenize_word_nyumba():
    assert tokenize_word("nyumba") == ["ny", "u", "mb", "a"]


def test_tokenize_text_multiple_words():
    tokens = tokenize_text("Habari za leo")
    assert tokens == ["h", "a", "b", "a", "r", "i", "z", "a", "l", "e", "o"]


def test_tokenize_text_strips_punctuation():
    assert tokenize_text("Habari, za leo!") == tokenize_text("Habari za leo")


def test_phoneme_frequencies_counts_across_corpus():
    freqs = phoneme_frequencies(["baba", "baba", "mtoto"])
    assert freqs["b"] == 4
    assert freqs["a"] == 4  # "baba" x2 -> 4, "mtoto" contributes no "a"
    assert freqs["m"] == 1


def test_core_graphemes_are_all_reachable_by_tokenizer():
    sample_words = "a e i o u chakula dhambi ghafla ngoma ng'ombe nyumba shule thamani"
    tokens = set(tokenize_text(sample_words))
    assert CORE_SWAHILI_GRAPHEMES.issubset(tokens)
