"""Outcome variables over works published in t0..t0+14.

Limitation: OpenAlex gives *current* citation totals, so citations received after t0+14
by papers published inside the window are included, and papers published early have had
more time to accumulate citations than the window itself implies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def h_index(cites) -> int:
    c = np.sort(np.asarray(list(cites), dtype=int))[::-1]
    return int(((c >= np.arange(1, len(c) + 1)).sum()))


def build_targets(outcome: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per cohort author. Authors with no works in the window get 0 (never dropped)."""
    g = outcome.groupby("author_id").cites
    agg = pd.DataFrame({"citations_15": g.sum(), "n_papers_15": g.size(),
                        "h_index_15": g.apply(h_index)})
    t = cohort[["author_id", "t0"]].merge(agg, on="author_id", how="left")
    t[["citations_15", "n_papers_15", "h_index_15"]] = t[["citations_15", "n_papers_15", "h_index_15"]].fillna(0).astype(int)
    thr = float(np.quantile(t.citations_15, 1 - config.TOP_FRAC))
    t["label"] = (t.citations_15 >= thr).astype(int)
    t.attrs["top5_threshold"] = thr
    return t
