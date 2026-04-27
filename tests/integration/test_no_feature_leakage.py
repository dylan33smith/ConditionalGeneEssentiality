"""Integration test: condition vocab contains no val-only media."""
import pytest
from src.data.datasets.dataset_audits import check_no_vocab_leakage


def test_clean_vocab():
    unknown_rate = check_no_vocab_leakage(
        train_media=["LB", "M9"],
        val_media=["LB", "NOVEL"],
        vocab={"LB", "M9"},
    )
    assert unknown_rate == pytest.approx(0.5)


def test_vocab_leakage_raises():
    with pytest.raises(AssertionError, match="Vocab leakage"):
        check_no_vocab_leakage(
            train_media=["LB"],
            val_media=["NOVEL"],
            vocab={"LB", "NOVEL"},   # NOVEL leaked into vocab!
        )
