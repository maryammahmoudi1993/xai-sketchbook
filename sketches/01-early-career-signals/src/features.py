"""Early-career features computed ONLY from works published in t0..t0+2 (no leakage)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

DATA_DICTIONARY = {
    "n_papers_early": ("Papers in first 3 years", "Distinct papers published in t0..t0+2."),
    "early_citations": ("Citations to early papers", "Current citations to the early papers (overlaps the target; models are run with and without)."),
    "n_distinct_venues": ("Different venues", "Distinct journals/conferences among early papers."),
    "n_top_venue_papers": ("Papers in top venues", "Early papers in top-10% venues (by 2-yr mean citedness) or a hand-listed set of major AI venues."),
    "has_top_coauthor": ("Worked with a top researcher", "1 if any early coauthor is in the top 5% of coauthors by current h-index (approximation: uses today's h-index)."),
    "n_unique_coauthors": ("Different coauthors", "Distinct coauthors on early papers."),
    "mean_team_size": ("Average team size", "Mean number of authors per early paper."),
    "institution_tier": ("Institution rank group", "Quartile of the author's most frequent early institution by institution h-index: 1=top ... 4=bottom, 0=unknown."),
    "topic_diversity": ("Topic variety", "Shannon entropy (nats) of primary topics across early papers."),
    "first_author_share": ("Share as first author", "Fraction of early papers where the author is listed first."),
    "cohort_year": ("Start year", "t0: year of first indexed publication (control)."),
}
FEATURES_ALL = list(DATA_DICTIONARY)
FEATURES_NO_EARLY_CITES = [f for f in FEATURES_ALL if f != "early_citations"]


def entropy(counts) -> float:
    c = np.asarray(list(counts), dtype=float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def top_venues(early: pd.DataFrame, sources: pd.DataFrame) -> tuple[set, dict]:
    """Top-10% venues by citedness within the dataset + hand-listed ones. Returns (ids, match log)."""
    used = sources[sources.source_id.isin(set(early.source_id.dropna()))].dropna(subset=["citedness"])
    thr = used.citedness.quantile(config.TOP_VENUE_QUANTILE)
    stat_ids = set(used[used.citedness >= thr].source_id)
    paper_counts = early.source_id.value_counts()
    log = {"citedness_threshold": float(thr), "n_stat_top": len(stat_ids), "hand": {}}
    hand_ids: set = set()
    for label, pat in config.HAND_VENUES.items():
        rx = re.compile(pat, re.I)
        hit = sources[sources.name.fillna("").apply(lambda n: bool(rx.search(n)))]
        hit = hit[hit.source_id.isin(paper_counts.index)]
        hand_ids |= set(hit.source_id)
        log["hand"][label] = [{"name": r.name, "papers_in_cohort": int(paper_counts.get(r.source_id, 0))}
                              for r in hit.itertuples()]
    log["hand_unmatched"] = [k for k, v in log["hand"].items() if not v]
    return stat_ids | hand_ids, log


def build_features(early: pd.DataFrame, cohort: pd.DataFrame, sources: pd.DataFrame,
                   coauthors: pd.DataFrame, institutions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`early` = full-field author-work rows restricted to t0..t0+2 (one row per author/work)."""
    log: dict = {}
    top_v, log["venues"] = top_venues(early, sources)

    h = coauthors.dropna(subset=["h_index"]).set_index("author_id").h_index
    h_thr = float(h.quantile(config.TOP_COAUTHOR_QUANTILE))
    top_co = set(h[h >= h_thr].index)
    log["top_coauthor_h_threshold"] = h_thr
    log["n_coauthors_with_h"] = int(len(h))

    inst = institutions.dropna(subset=["h_index"]).copy()
    # tier 1 = highest quartile of institution h-index
    inst["tier"] = 4 - pd.qcut(inst.h_index.rank(method="first"), 4, labels=False)
    tier_of = inst.set_index("inst_id").tier.to_dict()
    log["n_institutions_with_h"] = int(len(inst))

    recs = []
    for aid, g in early.groupby("author_id"):
        co = {a for lst in g.coauthors for a in lst}
        insts = pd.Series([i for lst in g.institutions for i in lst])
        main_inst = insts.value_counts().index[0] if len(insts) else None
        recs.append({
            "author_id": aid,
            "n_papers_early": g.work_id.nunique(),
            "early_citations": int(g.cites.sum()),
            "n_distinct_venues": g.source_id.nunique(),
            "n_top_venue_papers": int(g.source_id.isin(top_v).sum()),
            "has_top_coauthor": int(bool(co & top_co)),
            "n_unique_coauthors": len(co),
            "mean_team_size": float(g.team_size.mean()),
            "institution_tier": int(tier_of.get(main_inst, 0)),
            "topic_diversity": entropy(g.topic_id.dropna().value_counts()),
            "first_author_share": float((g.position == "first").mean()),
        })
    f = pd.DataFrame(recs).merge(cohort[["author_id", "t0"]].rename(columns={"t0": "cohort_year"}), on="author_id")
    log["share_unknown_institution"] = float((f.institution_tier == 0).mean())
    log["share_has_top_coauthor"] = float(f.has_top_coauthor.mean())
    return f, log


def write_dictionary(path) -> pd.DataFrame:
    d = pd.DataFrame([(k, *v) for k, v in DATA_DICTIONARY.items()],
                     columns=["column", "plot_label", "description"])
    d.to_csv(path, index=False)
    return d
