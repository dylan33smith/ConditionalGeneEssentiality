"""C -- the leave-compound-out split."""
import numpy as np
import pandas as pd
import pytest

from src.data.datasets.build_ranking_split import (
    materialize_leave_compound_out,
    assert_no_compound_leakage,
)


def _fixture(n_compounds=20, n_genes=5):
    """One experiment per compound, plus a shared medium component everywhere."""
    rows, chem = [], []
    for c in range(n_compounds):
        exp = f"exp{c}"
        chem.append(dict(experiment_id=exp, canonical_id="glc-D", role="medium"))
        chem.append(dict(experiment_id=exp, canonical_id=f"cmp{c}", role="stressor"))
        for g in range(n_genes):
            rows.append(dict(orgId="org0", gene_key=f"g{g}", experiment_id=exp,
                             expDesc=f"d{c}", media="M", temperature=37,
                             expName=f"e{c}", fit=float(g - c)))
    return pd.DataFrame(rows), pd.DataFrame(chem)


def test_holds_out_whole_compounds_and_never_leaks_them_into_train():
    fit, chem = _fixture()
    split = materialize_leave_compound_out(fit, experiment_chemistry=chem,
                                           fraction=0.25, seed=0)
    exp_to_comp = (chem[chem.role == "stressor"]
                   .groupby("experiment_id")["canonical_id"].agg(set))
    train_exps = set(fit.loc[split.partition == "train", "experiment_id"])
    val_exps = set(fit.loc[split.partition == "val", "experiment_id"])

    train_comps = set().union(*(exp_to_comp[e] for e in train_exps))
    val_comps = set().union(*(exp_to_comp[e] for e in val_exps))

    assert val_comps, "some compounds must be held out"
    assert not (val_comps & train_comps), (
        "a held-out compound appeared in training -- the split leaks, which defeats "
        "its entire purpose")
    assert split.stats["holdout_unit"] == "compound"
    assert split.stats["n_compounds_held_out"] == 5   # 0.25 * 20


def test_medium_components_are_never_held_out():
    fit, chem = _fixture()
    split = materialize_leave_compound_out(fit, experiment_chemistry=chem,
                                           fraction=0.5, seed=1)
    # glc-D is a medium component present in every experiment; if media were eligible
    # for holdout, every row would land in val
    assert (split.partition == "train").sum() > 0
    assert (split.partition == "val").sum() > 0


def test_scaffold_grouping_holds_out_whole_groups():
    fit, chem = _fixture(n_compounds=20)
    # two scaffolds, alternating
    groups = {f"cmp{c}": ("scafA" if c % 2 == 0 else "scafB") for c in range(20)}
    split = materialize_leave_compound_out(
        fit, experiment_chemistry=chem, fraction=0.5, seed=0, compound_groups=groups)
    assert split.stats["holdout_unit"] == "group"
    assert split.stats["n_groups_total"] == 2
    assert split.stats["n_groups_held_out"] == 1

    exp_to_comp = (chem[chem.role == "stressor"]
                   .groupby("experiment_id")["canonical_id"].agg(set))
    val_exps = set(fit.loc[split.partition == "val", "experiment_id"])
    val_comps = set().union(*(exp_to_comp[e] for e in val_exps))
    # every held-out compound belongs to the same scaffold
    assert len({groups[c] for c in val_comps}) == 1


def test_split_is_deterministic_and_hashed():
    fit, chem = _fixture()
    a = materialize_leave_compound_out(fit, experiment_chemistry=chem, seed=3)
    b = materialize_leave_compound_out(fit, experiment_chemistry=chem, seed=3)
    c = materialize_leave_compound_out(fit, experiment_chemistry=chem, seed=4)
    assert a.split_hash == b.split_hash
    assert a.split_hash != c.split_hash


def test_leakage_assert_passes_on_a_clean_split():
    fit, chem = _fixture()
    split = materialize_leave_compound_out(fit, experiment_chemistry=chem, seed=0)
    assert_no_compound_leakage(fit, split, experiment_chemistry=chem)


def test_missing_experiment_id_is_a_loud_error():
    fit, chem = _fixture()
    with pytest.raises(KeyError, match="experiment_id"):
        materialize_leave_compound_out(fit.drop(columns=["experiment_id"]),
                                       experiment_chemistry=chem)


def test_no_stressors_is_a_loud_error():
    fit, chem = _fixture()
    with pytest.raises(ValueError, match="No stressor compounds"):
        materialize_leave_compound_out(
            fit, experiment_chemistry=chem[chem.role == "medium"])
