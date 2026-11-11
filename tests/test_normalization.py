from src.data.normalization import doc_hash, strongest_match_level, tokens, word_ngrams


def test_tokens_ignore_case_unicode_form_and_punctuation():
    assert tokens("Età: ETÀ, età!") == ["età", "età", "età"]
    assert tokens("é") == tokens("é")


def test_hash_ignores_only_format():
    assert doc_hash("Una  donna,\r\ndi anni") == doc_hash("una donna di anni")
    assert doc_hash("una donna") != doc_hash("un uomo")


def test_match_levels_report_the_strictest_equal_level():
    assert strongest_match_level("a b", "a b") == "byte_identical"
    assert strongest_match_level("a b ", "a b") == "whitespace_trimmed"
    assert strongest_match_level("a  b", "a b") == "whitespace_normalised"
    assert strongest_match_level("A b", "a b") == "unicode_case_whitespace"
    assert strongest_match_level("a, b", "a b") == "token_level_punct_insensitive"
    assert strongest_match_level("a b", "a c") is None


def test_ngrams():
    assert word_ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}
    assert word_ngrams(["a"], 2) == set()
