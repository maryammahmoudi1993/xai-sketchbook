"""SHAP explanations and the model-free 2x2 comparison (top coauthor x institution tier)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config
from .features import FEATURES_NO_EARLY_CITES
from .modeling import make_logistic


def _pos_class(sv):
    """shap returns a list per class in some versions/configs; keep the positive class."""
    return sv[1] if isinstance(sv, list) else sv


def shap_explain(model, X: pd.DataFrame, interactions: bool = True):
    ex = shap.TreeExplainer(model)
    sv = _pos_class(ex.shap_values(X))
    iv = None
    if interactions:
        iv = ex.shap_interaction_values(X)
        iv = iv[1] if isinstance(iv, list) else iv
    return sv, iv


def interaction_table(iv: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Mean total SHAP interaction (2 * off-diagonal) of has_top_coauthor x institution_tier,
    split by whether the author had a top coauthor, per tier."""
    cols = list(X.columns)
    i, j = cols.index("has_top_coauthor"), cols.index("institution_tier")
    val = 2 * iv[:, i, j]
    d = pd.DataFrame({"tier": X.institution_tier.values, "has_top_coauthor": X.has_top_coauthor.values, "interaction": val})
    return d.groupby(["tier", "has_top_coauthor"]).interaction.agg(["mean", "size"]).reset_index()


def shap_by_group(sv: np.ndarray, iv: np.ndarray, X: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Among test authors WITH a top coauthor: mean SHAP of has_top_coauthor (its total push on the
    prediction) and mean SHAP interaction with institution tier, by tier group, with bootstrap CIs."""
    cols = list(X.columns)
    i, j = cols.index("has_top_coauthor"), cols.index("institution_tier")
    grp = group_assignment(X)
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name in ("Top-ranked institution", "Lower-ranked institution"):
        m = ((grp == name) & (X.has_top_coauthor == 1)).values
        row = {"tier_group": name, "n": int(m.sum())}
        for key, v in (("shap", sv[m, i]), ("interaction", 2 * iv[m, i, j])):
            if len(v) == 0:
                row.update({key: np.nan, key + "_lo": np.nan, key + "_hi": np.nan})
                continue
            b = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
            row.update({key: float(v.mean()), key + "_lo": float(np.percentile(b, 2.5)), key + "_hi": float(np.percentile(b, 97.5))})
        rows.append(row)
    return pd.DataFrame(rows)


def group_assignment(df: pd.DataFrame) -> pd.Series:
    """Tier group: 'Top-ranked' (tier 1), 'Lower-ranked' (tiers 3-4); tier 2 / unknown excluded."""
    g = pd.Series(np.nan, index=df.index, dtype=object)
    g[df.institution_tier == 1] = "Top-ranked institution"
    g[df.institution_tier.isin([3, 4])] = "Lower-ranked institution"
    return g


def _rate_table(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["tier_group", "has_top_coauthor"]).label.agg(["mean", "size", "sum"])


def group_analysis(df: pd.DataFrame, n_boot: int = 2000) -> dict:
    """P(top 5%) by has_top_coauthor x tier group with bootstrap CIs, effect per tier group,
    and the difference-in-differences (does the coauthor gap differ for lower-ranked institutions?)."""
    d = df.assign(tier_group=group_assignment(df)).dropna(subset=["tier_group"])
    rng = np.random.default_rng(config.SEED)
    tiers = ["Top-ranked institution", "Lower-ranked institution"]
    cells = [(t, c) for t in tiers for c in (0, 1)]
    idx = {cell: np.where((d.tier_group == cell[0]) & (d.has_top_coauthor == cell[1]))[0] for cell in cells}
    y = d.label.values
    rate = lambda ii: y[ii].mean() if len(ii) else np.nan
    point = {c: rate(idx[c]) for c in cells}
    boots = {c: [] for c in cells}
    gap_t, gap_l, did = [], [], []
    for _ in range(n_boot):
        r = {c: rate(rng.choice(idx[c], len(idx[c]))) if len(idx[c]) else np.nan for c in cells}
        for c in cells:
            boots[c].append(r[c])
        gt = r[(tiers[0], 1)] - r[(tiers[0], 0)]
        gl = r[(tiers[1], 1)] - r[(tiers[1], 0)]
        gap_t.append(gt); gap_l.append(gl); did.append(gl - gt)
    ci = lambda a: tuple(np.nanpercentile(a, [2.5, 97.5]))
    rows = [{"tier_group": c[0], "has_top_coauthor": c[1], "n": len(idx[c]), "n_top5": int(y[idx[c]].sum()),
             "p_top5": point[c], "ci_low": ci(boots[c])[0], "ci_high": ci(boots[c])[1]} for c in cells]
    gap = lambda t: point[(t, 1)] - point[(t, 0)]
    # overall coauthor gap (all four cells pooled)
    ov = d.groupby("has_top_coauthor").label.mean()
    ov_boot = []
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 31)))
        m = s.groupby("has_top_coauthor").label.mean()
        if {0, 1} <= set(m.index):
            ov_boot.append(m[1] - m[0])
    return {"table": pd.DataFrame(rows),
            "gap_top": (gap(tiers[0]), *ci(gap_t)), "gap_lower": (gap(tiers[1]), *ci(gap_l)),
            "did": (gap(tiers[1]) - gap(tiers[0]), *ci(did)),
            "overall_gap": (float(ov.get(1, np.nan) - ov.get(0, np.nan)), *np.percentile(ov_boot, [2.5, 97.5])),
            "n_excluded_mid_or_unknown": int(len(df) - len(d))}


def adjusted_coauthor_effect(df: pd.DataFrame, n_boot: int = 200) -> tuple[float, float, float]:
    """Standardized-logistic coefficient (log-odds) of has_top_coauthor controlling for the other
    early features (without early_citations), with a bootstrap 95% CI."""
    feats = FEATURES_NO_EARLY_CITES
    rng = np.random.default_rng(config.SEED)

    def coef(d):
        m = make_logistic(feats).fit(d[feats], d.label)
        names = m[0].get_feature_names_out()
        return float(m[-1].coef_[0][list(names).index("num__has_top_coauthor")])

    est, boots = coef(df), []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        if s.label.nunique() == 2:
            boots.append(coef(s))
    return est, *np.percentile(boots, [2.5, 97.5])


def li2019_agreement(lo: float, hi: float, est: float) -> str:
    """Li et al. 2019 (Nat. Commun.) report that early coauthorship with a top scientist is
    associated with higher later success. Agreement = positive association with CI excluding 0."""
    if lo > 0:
        return "agrees (positive association, CI excludes 0)"
    if hi < 0:
        return "disagrees (negative association, CI excludes 0)"
    return "inconclusive (CI crosses 0)"
