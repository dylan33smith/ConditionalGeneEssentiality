"""Integration test: split partitions have no organism overlap."""
import pytest
import pandas as pd
from src.data.datasets.dataset_audits import check_no_organism_overlap


def test_no_organism_overlap_simple():
    train = pd.DataFrame({"orgId": ["A", "A", "B"]})
    val   = pd.DataFrame({"orgId": ["C", "C"]})
    test  = pd.DataFrame({"orgId": ["D"]})
    check_no_organism_overlap(train, val, test)  # should not raise


def test_organism_overlap_raises():
    train = pd.DataFrame({"orgId": ["A", "B"]})
    val   = pd.DataFrame({"orgId": ["A"]})   # overlap!
    test  = pd.DataFrame({"orgId": ["C"]})
    with pytest.raises(AssertionError, match="Organism overlap"):
        check_no_organism_overlap(train, val, test)
