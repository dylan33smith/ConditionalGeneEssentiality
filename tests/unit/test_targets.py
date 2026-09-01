"""F -- de-meaned and denoised targets."""
import numpy as np
import pandas as pd
import pytest

from src.ranking.targets import (
    fit_additive_effects, apply_demeaning, replicate_average_target,
)


def _additive(n_g=25, n_c=12, interaction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    a = {f"g{i}": rng.normal(0, 1.0) for i in range(n_g)}
    b = {f"c{j}": rng.normal(0, 0.7) for j in range(n_c)}
    mu = 0.5
    rows = []
    for g, av in a.items():
        for c, bv in b.items():
            rows.append(dict(orgId="o", gene_key=g, condition_key=c, expName=f"{c}_r1",
                             fit=mu + av + bv + rng.normal(0, interaction)))
    return pd.DataFrame(rows), mu, a, b


def test_recovers_the_main_effects_it_was_given():
    """mu, a and b are identified only under a sum-to-zero constraint on a and b."""
    df, mu, a, b = _additive(interaction=0.05)
    eff = fit_additive_effects(df)

    # mu is the grand mean; the true generative mu plus the sample means of a and b
    expected_mu = mu + np.mean(list(a.values())) + np.mean(list(b.values()))
    assert eff.mu == pytest.approx(expected_mu, abs=0.05)

    # effects are returned centred, so compare against centred truth
    est_a = np.array([eff.gene_effect[g] for g in a])
    true_a = np.array(list(a.values())); true_a = true_a - true_a.mean()
    assert est_a.mean() == pytest.approx(0.0, abs=1e-9), "effects must be centred"
    assert np.corrcoef(est_a, true_a)[0, 1] > 0.98
    assert np.abs(est_a - true_a).max() < 0.1
    assert eff.stats["var_explained_by_main_effects"] > 0.9


def test_demeaning_removes_gene_effect_variance():
    df, *_ = _additive()
    eff = fit_additive_effects(df)
    out = apply_demeaning(df, eff, components=("gene",))
    # per-gene means should now be far closer to zero than the raw per-gene means
    raw_spread = df.groupby("gene_key").fit.mean().std()
    dem_spread = out.groupby("gene_key").fit_demeaned.mean().std()
    assert dem_spread < 0.3 * raw_spread


def test_demeaning_by_condition_on_cold_columns_is_a_loud_error():
    """The constraint that cannot be engineered away."""
    df, *_ = _additive()
    train = df[df.condition_key.isin([f"c{j}" for j in range(6)])]
    val = df[~df.condition_key.isin([f"c{j}" for j in range(6)])]
    eff = fit_additive_effects(train)
    # gene-only de-meaning is fine on val
    apply_demeaning(val, eff, components=("gene",))
    # condition de-meaning is not -- b_c is unestimable for 100% of val conditions
    with pytest.raises(ValueError, match="not estimable"):
        apply_demeaning(val, eff, components=("gene", "condition"))


def test_unknown_keys_fall_back_to_mu_without_crashing():
    df, *_ = _additive()
    eff = fit_additive_effects(df)
    novel = pd.DataFrame([dict(orgId="o", gene_key="NEW", condition_key="c0",
                               expName="x", fit=1.0)])
    out = apply_demeaning(novel, eff, components=("gene",))
    assert np.isfinite(out.fit_demeaned).all()


def test_replicate_averaging_reduces_noise_and_counts_replicates():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(20):
        true = rng.normal()
        for r in range(4):
            rows.append(dict(orgId="o", gene_key="g", condition_key=f"c{j}",
                             expName=f"c{j}_r{r}", fit=true + rng.normal(0, 1.0),
                             true=true))
    df = pd.DataFrame(rows)
    out = replicate_average_target(df)
    assert len(out) == 20
    assert (out.n_replicates == 4).all()
    truth = df.groupby(["orgId", "gene_key", "condition_key"]).true.first().reset_index()
    m = out.merge(truth, on=["orgId", "gene_key", "condition_key"])
    err_avg = np.std(m.fit_denoised - m.true)
    err_single = np.std(df.fit - df.true)
    assert err_avg < 0.7 * err_single, "averaging 4 replicates should roughly halve noise"


def test_min_replicates_filter():
    rows = [dict(orgId="o", gene_key="g", condition_key="c0", expName="a", fit=1.0),
            dict(orgId="o", gene_key="g", condition_key="c1", expName="b", fit=1.0),
            dict(orgId="o", gene_key="g", condition_key="c1", expName="c", fit=2.0)]
    out = replicate_average_target(pd.DataFrame(rows), min_replicates=2)
    assert len(out) == 1 and out.condition_key.iloc[0] == "c1"


def test_empty_input_is_safe():
    eff = fit_additive_effects(pd.DataFrame(columns=["gene_key", "condition_key", "fit"]))
    assert eff.n_train_rows == 0 and eff.stats.get("empty")


def test_bad_component_is_rejected():
    df, *_ = _additive()
    eff = fit_additive_effects(df)
    with pytest.raises(ValueError, match="unknown component"):
        apply_demeaning(df, eff, components=("bogus",))
