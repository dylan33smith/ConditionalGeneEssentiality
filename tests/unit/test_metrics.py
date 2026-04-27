"""Unit tests for metrics module."""
import numpy as np
import pytest
from src.evaluation.metrics import rmse, mae, within_gene_spearman


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 0.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_within_gene_spearman_min_conditions():
    """Genes with fewer than min_conditions are excluded."""
    y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    genes  = np.array(["g1", "g1", "g1", "g2", "g2"])
    result = within_gene_spearman(y_true, y_pred, genes, min_conditions=3)
    # g1 has 3 conditions (eligible), g2 has 2 (excluded)
    assert result["n_genes_used"] == 1
    assert result["mean_spearman"] == pytest.approx(1.0)
