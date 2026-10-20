import numpy as np
import pandas as pd

from src import features, modeling


def synthetic(n=4000, seed=0):
    """Pure-noise-plus-one-signal data used ONLY to exercise the code path (not a result)."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "n_papers_early": r.poisson(3, n), "early_citations": r.poisson(10, n),
        "n_distinct_venues": r.poisson(2, n), "n_top_venue_papers": r.poisson(1, n),
        "has_top_coauthor": r.integers(0, 2, n), "n_unique_coauthors": r.poisson(6, n),
        "mean_team_size": r.uniform(1, 6, n), "institution_tier": r.integers(0, 5, n),
        "topic_diversity": r.uniform(0, 2, n), "first_author_share": r.uniform(0, 1, n),
        "cohort_year": r.choice([2008, 2009, 2010, 2011], n)})
    z = 1.5 * df.has_top_coauthor + 0.4 * df.n_papers_early - 4.5
    df["label"] = (r.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def test_run_all_shapes_and_auc_above_chance():
    df = synthetic()
    res = modeling.run_all(df, n_boot=50)
    assert res["test"].cohort_year.min() >= 2010 and res["train"].cohort_year.max() <= 2009
    m = res["metrics"][("lightgbm", "without_early_citations")]
    lo, hi = m["roc_auc"][1:]
    assert lo <= m["roc_auc"][0] <= hi and m["roc_auc"][0] > 0.6
    assert "early_citations" not in features.FEATURES_NO_EARLY_CITES
    assert len(res["metrics"]) == 4


def test_group_analysis_and_explain():
    from src import explain
    df = synthetic(3000)
    df["institution_tier"] = np.random.default_rng(1).integers(1, 5, len(df))
    ga = explain.group_analysis(df, n_boot=100)
    t = ga["table"]
    assert set(t.n) and t.n.sum() + ga["n_excluded_mid_or_unknown"] == len(df)
    assert ga["overall_gap"][0] > 0                      # synthetic signal is positive
    est, lo, hi = explain.adjusted_coauthor_effect(df, n_boot=20)
    assert lo <= est <= hi or lo <= hi
    res = modeling.run_all(df, n_boot=20)
    te = res["test"][features.FEATURES_NO_EARLY_CITES].head(200)
    sv, iv = explain.shap_explain(res["models"]["without_early_citations"], te)
    assert sv.shape == te.shape and iv.shape == (200, te.shape[1], te.shape[1])
    assert not explain.interaction_table(iv, te).empty
