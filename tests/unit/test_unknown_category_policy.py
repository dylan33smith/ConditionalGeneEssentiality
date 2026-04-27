"""Unit tests for unknown category policy."""
import pytest

from src.data.preprocessing.unknown_category_policy import apply_unk_policy


def test_all_known():
    cats, rate = apply_unk_policy(["LB", "M9"], known_vocab={"LB", "M9"})
    assert rate == 0.0
    assert cats == ["LB", "M9"]


def test_all_unknown():
    cats, rate = apply_unk_policy(["X", "Y"], known_vocab={"LB"})
    assert rate == 1.0
    assert all(c == "<UNK>" for c in cats)


def test_mixed():
    cats, rate = apply_unk_policy(["LB", "NOVEL"], known_vocab={"LB"})
    assert cats[0] == "LB"
    assert cats[1] == "<UNK>"
    assert rate == pytest.approx(0.5)
