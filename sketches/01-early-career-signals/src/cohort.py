"""Cohort construction.

Stage 0  seed:     sample AI-subfield works from 2008-2011 -> candidate author IDs.
Stage A  early:    all works up to 2013 for every candidate -> first publication year (t0),
                   cohort filter, early-window features. Authors who later stop publishing
                   are kept (their outcome is simply low), which limits survivorship bias.
Stage B  outcome:  works 2014-2025 for the cohort only -> citations over t0..t0+14.

Choice documented: we require >= MIN_EARLY_WORKS works in t0..t0+2 to exclude one-off authors.
Caveat: t0 is the first *OpenAlex-indexed* publication year, which can be later than the true
first publication if older records are missing or the author profile is split.
"""
from __future__ import annotations

import random

import pandas as pd
from tqdm import tqdm

from . import config
from .api import OpenAlexClient, short_id

WORK_SELECT = "id,publication_year,cited_by_count,primary_location,primary_topic,authorships"
OUTCOME_SELECT = "id,publication_year,cited_by_count,authorships"


# ---------------------------------------------------------------- Stage 0
def collect_candidate_authors(client: OpenAlexClient, dry_run: bool = config.DRY_RUN) -> list[str]:
    """Author IDs appearing on a seeded random sample of AI works per seed year."""
    n_sample = 200 if dry_run else config.SEED_SAMPLE_WORKS
    authors: set[str] = set()
    for year in range(config.SEED_YEARS[0], config.SEED_YEARS[1] + 1):
        filt = (f"primary_topic.subfield.id:{config.SUBFIELD_ID},publication_year:{year},"
                f"type:{config.WORK_TYPES}")
        got, page = 0, 1
        while got < n_sample:
            params = {"filter": filt, "select": "id,authorships", "sample": n_sample,
                      "seed": config.SEED, "per_page": config.PER_PAGE, "page": page}
            res = client.get("works", params).get("results", [])
            if not res:
                break
            for w in res:
                for a in w.get("authorships", []):
                    aid = short_id((a.get("author") or {}).get("id"))
                    if aid:
                        authors.add(aid)
            got += len(res)
            page += 1
    ids = sorted(authors)
    rng = random.Random(config.SEED)
    rng.shuffle(ids)
    cap = config.DRY_RUN_AUTHORS if dry_run else config.CANDIDATE_AUTHOR_CAP
    return ids[:cap]


# ---------------------------------------------------------------- work flattening
def _author_rows(work: dict, wanted: set[str], full: bool) -> list[dict]:
    """One row per (wanted author, work). `full` keeps the fields needed for features."""
    auths = work.get("authorships") or []
    all_ids = [short_id((a.get("author") or {}).get("id")) for a in auths]
    all_ids = [a for a in all_ids if a]
    loc = work.get("primary_location") or {}
    topic = work.get("primary_topic") or {}
    rows = []
    for a in auths:
        aid = short_id((a.get("author") or {}).get("id"))
        if aid not in wanted:
            continue
        row = {"author_id": aid, "work_id": short_id(work["id"]),
               "year": work.get("publication_year"), "cites": work.get("cited_by_count", 0) or 0}
        if full:
            row.update({
                "source_id": short_id((loc.get("source") or {}).get("id")),
                "topic_id": short_id(topic.get("id")),
                "position": a.get("author_position"),
                "team_size": len(all_ids),
                "coauthors": [x for x in all_ids if x != aid],
                "institutions": [short_id(i.get("id")) for i in a.get("institutions", []) if i.get("id")],
            })
        rows.append(row)
    return rows


def _fetch_works_for_authors(client: OpenAlexClient, ids: list[str], year_filter: str,
                             select: str, full: bool, desc: str) -> pd.DataFrame:
    rows: list[dict] = []
    ids = sorted(set(ids))
    for i in tqdm(range(0, len(ids), config.BATCH_IDS), desc=desc):
        chunk = ids[i:i + config.BATCH_IDS]
        wanted = set(chunk)
        params = {"filter": f"authorships.author.id:{'|'.join(chunk)},publication_year:{year_filter},"
                            f"type:{config.WORK_TYPES}", "select": select}
        for w in client.paginate("works", params):
            rows.extend(_author_rows(w, wanted, full))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Stage A / B
def fetch_stage_a(client: OpenAlexClient, candidates: list[str]) -> pd.DataFrame:
    """Works with year <= t0max+2 for all candidates (needed for t0 and early features)."""
    last = config.T0_RANGE[1] + config.EARLY_WINDOW - 1
    return _fetch_works_for_authors(client, candidates, f"<{last + 1}", WORK_SELECT, True, "stage A")


def select_cohort(stage_a: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return per-author t0 table for the cohort plus a log of the funnel."""
    log = {"candidates_with_works": int(stage_a.author_id.nunique())}
    d = stage_a.dropna(subset=["year"]).drop_duplicates(["author_id", "work_id"])
    t0 = d.groupby("author_id").year.min().rename("t0").reset_index()
    log["t0_in_range"] = int(t0.t0.between(*config.T0_RANGE).sum())
    t0 = t0[t0.t0.between(*config.T0_RANGE)]
    m = d.merge(t0, on="author_id")
    early = m[(m.year >= m.t0) & (m.year <= m.t0 + config.EARLY_WINDOW - 1)]
    n_early = early.groupby("author_id").work_id.nunique().rename("n_early").reset_index()
    t0 = t0.merge(n_early, on="author_id", how="left").fillna({"n_early": 0})
    t0 = t0[t0.n_early >= config.MIN_EARLY_WORKS].reset_index(drop=True)
    log["cohort_after_min_early_works"] = int(len(t0))
    return t0, log


def fetch_stage_b(client: OpenAlexClient, cohort_ids: list[str]) -> pd.DataFrame:
    first = config.T0_RANGE[1] + config.EARLY_WINDOW
    return _fetch_works_for_authors(client, cohort_ids, f"{first}-{config.LAST_YEAR}",
                                    OUTCOME_SELECT, False, "stage B")


def outcome_works(stage_a: pd.DataFrame, stage_b: pd.DataFrame, t0: pd.DataFrame) -> pd.DataFrame:
    """All distinct (author, work) rows with year in t0..t0+14 for cohort authors."""
    cols = ["author_id", "work_id", "year", "cites"]
    allw = pd.concat([stage_a[cols], stage_b[cols] if len(stage_b) else stage_b], ignore_index=True)
    allw = allw.merge(t0[["author_id", "t0"]], on="author_id")
    allw = allw[(allw.year >= allw.t0) & (allw.year <= allw.t0 + config.OUTCOME_WINDOW - 1)]
    return allw.drop_duplicates(["author_id", "work_id"])
