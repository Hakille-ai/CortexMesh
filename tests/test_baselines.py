from __future__ import annotations

import pytest

from cortexmesh.baselines import CharNGramBaseline


def test_predict_next_uses_longest_known_context() -> None:
    baseline = CharNGramBaseline(max_order=2).fit("ababa")

    assert baseline.predict_next("ab") == "a"
    assert baseline.predict_next("ba") == "b"


def test_predict_next_backs_off_to_global_counts() -> None:
    baseline = CharNGramBaseline(max_order=3).fit("zzabc")

    assert baseline.predict_next("unknown") == "z"


def test_generate_appends_deterministic_predictions() -> None:
    baseline = CharNGramBaseline(max_order=2).fit("abababab")

    assert baseline.generate("ab", steps=4) == "ababab"
    assert baseline.generate("ab", steps=0) == "ab"


def test_score_next_token_accuracy_uses_fixed_context_length() -> None:
    baseline = CharNGramBaseline(max_order=2).fit("abababab")

    assert baseline.score_next_token_accuracy("ababab", seq_len=2) == 1.0


def test_validation_and_empty_fit_behaviour() -> None:
    baseline = CharNGramBaseline()

    with pytest.raises(RuntimeError):
        baseline.predict_next("a")
    with pytest.raises(ValueError):
        CharNGramBaseline(max_order=-1)

    baseline.fit("")
    assert baseline.predict_next("a") == ""
    assert baseline.generate("seed", steps=3) == "seed"
    assert baseline.score_next_token_accuracy("abc", seq_len=10) == 0.0
