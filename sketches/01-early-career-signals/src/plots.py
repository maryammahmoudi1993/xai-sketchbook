"""LinkedIn-portrait (1080x1350) figures. Titles describe the question, never a result."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .features import DATA_DICTIONARY

INK, MUTED, GRID, SURFACE = "#1f2933", "#5f6b7a", "#e4e7eb", "#ffffff"
C_NO, C_YES = "#8a99ad", "#c4531a"      # grey = no top coauthor, burnt orange = top coauthor
C_BLUE = "#2a6fbb"
FOOTER = "Data: OpenAlex (CC0)"
W, H, DPI = 10.8, 13.5, 100             # -> 1080 x 1350 px
LABEL = {k: v[0] for k, v in DATA_DICTIONARY.items()}
TIER_NAME = {0: "Unknown", 1: "Top 25%", 2: "25-50%", 3: "50-75%", 4: "Bottom 25%"}

plt.rcParams.update({"font.size": 20, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def _canvas(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fig.text(0.06, 0.955, title, fontsize=30, fontweight="bold", va="top", wrap=True)
    if subtitle:
        fig.text(0.06, 0.865, subtitle, fontsize=19, color=MUTED, va="top", wrap=True)
    fig.text(0.06, 0.025, FOOTER, fontsize=16, color=MUTED)
    return fig


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def fig_group_bars(ga: dict, path) -> None:
    t = ga["table"]
    fig = _canvas("Early work with a top researcher goes with better odds of top-5% citations",
                  "Share of AI researchers in the top 5% most-cited by year 15 (association, not cause)")
    ax = fig.add_axes([0.12, 0.17, 0.82, 0.60])
    groups = ["Top-ranked institution", "Lower-ranked institution"]
    width = 0.36
    ax.set_ylim(0, max(t.ci_high.max() * 100 * 1.25, 1))
    top = ax.get_ylim()[1]
    for k, (c, lab, col) in enumerate(((0, "No top coauthor", C_NO), (1, "Top coauthor", C_YES))):
        sub = t[t.has_top_coauthor == c].set_index("tier_group").reindex(groups)
        x = np.arange(2) + (k - 0.5) * (width + 0.03)
        y = sub.p_top5.values * 100
        err = np.vstack([y - sub.ci_low.values * 100, sub.ci_high.values * 100 - y])
        ax.bar(x, y, width, color=col, label=lab)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=INK, capsize=6, lw=2)
        for xi, yi, n, hi in zip(x, y, sub.n.values, sub.ci_high.values * 100):
            ax.text(xi, hi + 0.01 * top, f"{yi:.1f}%", ha="center", fontsize=20, fontweight="bold")
            ax.text(xi, 0.012 * top, f"n={int(n):,}", ha="center", fontsize=15, color="white")
    ax.set_xticks(range(2), ["Top-ranked\ninstitution", "Lower-ranked\ninstitutions"])
    ax.set_ylabel("Reached top 5% (%)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=18)
    fig.text(0.06, 0.075, "Bars show 95% confidence ranges. Top-ranked = top quarter of institutions; lower-ranked = bottom half.\n"
             f"Excludes {ga['n_excluded_mid_or_unknown']:,} researchers with a middle-ranked or unknown institution.",
             fontsize=14, color=MUTED)
    _save(fig, path)


def _rename(X: pd.DataFrame) -> pd.DataFrame:
    return X.rename(columns=LABEL)


def _display(X: pd.DataFrame) -> pd.DataFrame:
    """Colour scale for plots only: institution rank high = top-ranked, unknown = missing (grey).
    SHAP values themselves are computed on the model's own encoding."""
    d = X.copy()
    t = d["institution_tier"]
    d["institution_tier"] = np.where(t == 0, np.nan, 5 - t)
    return _rename(d).rename(columns={LABEL["institution_tier"]: "Institution rank (high = top-ranked)"})


def fig_shap_beeswarm(sv, X: pd.DataFrame, path) -> None:
    fig = _canvas("What do the first three years say about a researcher's future?",
                  "Each dot is one researcher. Right = pushes toward 'top 5% by year 15'.")
    plt.sca(fig.add_axes([0.40, 0.14, 0.55, 0.68]))
    shap.summary_plot(sv, _display(X), show=False, plot_size=None, max_display=11)
    ax = plt.gca()
    ax.tick_params(labelsize=17)
    ax.set_xlabel("Effect on prediction (SHAP value)", fontsize=17)
    fig.text(0.06, 0.06, "Red = high value of the signal, blue = low.", fontsize=15, color=MUTED)
    _save(fig, path)


def fig_interaction(tab: pd.DataFrame, path) -> None:
    """Among held-out researchers who HAVE a top coauthor: how much that fact pushed the model's
    prediction up, by institution group. Bars = total push; markers = the part due to the interaction."""
    fig = _canvas("Does a top coauthor push predictions up more at lower-ranked institutions?",
                  "Model view, held-out researchers who had a top coauthor")
    ax = fig.add_axes([0.12, 0.17, 0.82, 0.58])
    x = np.arange(len(tab))
    y, lo, hi = tab["shap"].values, tab["shap_lo"].values, tab["shap_hi"].values
    ax.bar(x, y, 0.5, color=C_YES, label="Total push from having a top coauthor")
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor=INK, capsize=8, lw=2)
    ax.errorbar(x + 0.32, tab["interaction"].values,
                yerr=np.vstack([tab["interaction"] - tab["interaction_lo"], tab["interaction_hi"] - tab["interaction"]]),
                fmt="D", color=C_BLUE, ms=11, capsize=6, lw=2, label="Part due to institution rank (interaction)")
    for xi, n in zip(x, tab["n"]):
        ax.text(xi, 0.03, f"n={int(n)}", ha="center", color="white", fontsize=16)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, ["Top-ranked\ninstitution", "Lower-ranked\ninstitutions"])
    ax.set_ylabel("Effect on prediction (SHAP units)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=15, loc="upper right")
    fig.text(0.06, 0.075, "Bars/markers show 95% confidence ranges. Overlapping ranges = no clear difference.\n"
             "Model explanation, not causal effect. Held-out researchers (start year 2010-2011).", fontsize=14, color=MUTED)
    _save(fig, path)


def fig_metrics_card(metric: dict, base_rate: float, n_test: int, n_total: int, path, model="LightGBM") -> None:
    fig = _canvas("How well can the first 3 years predict the top 5%?",
                  f"{model}, trained on 2008-09 starters, tested on 2010-11 starters")
    auc, lo, hi = metric["roc_auc"]
    ap, alo, ahi = metric["pr_auc"]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    rows = [("Ranking accuracy (AUC)", f"{auc:.2f}", f"95% range {lo:.2f}-{hi:.2f}   (0.50 = coin flip)"),
            ("Precision-recall (PR-AUC)", f"{ap:.2f}", f"95% range {alo:.2f}-{ahi:.2f}   (chance = {base_rate:.2f})"),
            ("Researchers in study", f"{n_total:,}", f"{n_test:,} in the held-out test group"),
            ("Top-5% share in test group", f"{base_rate*100:.1f}%", "the base rate to beat")]
    for i, (lab, big, small) in enumerate(rows):
        y = 0.72 - i * 0.16
        ax.text(0.08, y + 0.045, lab, fontsize=21, color=MUTED)
        ax.text(0.08, y - 0.03, big, fontsize=50, fontweight="bold", color=C_BLUE)
        ax.text(0.40, y - 0.015, small, fontsize=18, color=INK)
    _save(fig, path)


def fig_dependence(sv, X: pd.DataFrame, feature: str, path) -> None:
    fig = _canvas(f"How '{LABEL[feature]}' moves the prediction", "Each dot is one held-out researcher")
    ax = fig.add_axes([0.14, 0.15, 0.80, 0.62])
    shap.dependence_plot(LABEL[feature], sv, _display(X), interaction_index=None, show=False, ax=ax)
    ax.set_ylabel("Effect on prediction (SHAP value)")
    _save(fig, path)


def fig_calibration(cal: dict, path) -> None:
    fig = _canvas("Are the predicted chances realistic?", "Predicted vs actual share reaching the top 5% (test group)")
    ax = fig.add_axes([0.14, 0.15, 0.80, 0.62])
    m = max(max(np.max(p), np.max(f)) for p, f in cal.values()) * 100 * 1.1
    ax.plot([0, m], [0, m], color=MUTED, ls="--", lw=2, label="Perfect")
    for (name, tag), (p, f) in cal.items():
        if tag != "without_early_citations":
            continue
        ax.plot(p * 100, f * 100, marker="o", lw=3, ms=10, color=C_BLUE if name == "lightgbm" else C_YES,
                label="LightGBM" if name == "lightgbm" else "Logistic regression")
    ax.set_xlabel("Predicted chance (%)")
    ax.set_ylabel("Actual share (%)")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=17)
    _save(fig, path)
