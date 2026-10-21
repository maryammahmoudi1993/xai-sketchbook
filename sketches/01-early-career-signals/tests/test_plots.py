import numpy as np
from PIL import Image

from src import explain, modeling, plots
from src.features import FEATURES_NO_EARLY_CITES
from tests.test_modeling import synthetic


def test_figures_have_linkedin_size(tmp_path):
    df = synthetic(3000)
    df["institution_tier"] = np.random.default_rng(1).integers(0, 5, len(df))
    res = modeling.run_all(df, n_boot=10)
    X = res["test"][FEATURES_NO_EARLY_CITES].head(150)
    sv, iv = explain.shap_explain(res["models"]["without_early_citations"], X)
    ga = explain.group_analysis(df.assign(institution_tier=df.institution_tier.clip(1, 4)), n_boot=20)
    plots.fig_group_bars(ga, tmp_path / "a.png")
    plots.fig_shap_beeswarm(sv, X, tmp_path / "b.png")
    plots.fig_interaction(explain.shap_by_group(sv, iv, X, n_boot=20), tmp_path / "c.png")
    plots.fig_metrics_card(res["metrics"][("lightgbm", "without_early_citations")], 0.05, 100, 400, tmp_path / "d.png")
    plots.fig_dependence(sv, X, "n_papers_early", tmp_path / "e.png")
    plots.fig_calibration(res["calibration"], tmp_path / "f.png")
    for f in "abcdef":
        assert Image.open(tmp_path / f"{f}.png").size == (1080, 1350)
