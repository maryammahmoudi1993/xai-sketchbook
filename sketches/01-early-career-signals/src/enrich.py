"""Batched lookups of sources, coauthors and institutions (<=50 IDs per call, all cached)."""
from __future__ import annotations

import pandas as pd

from . import config
from .api import OpenAlexClient, short_id


def _lookup(client: OpenAlexClient, endpoint: str, ids, select: str, parse) -> pd.DataFrame:
    ids = sorted({i for i in ids if i})
    rows = [parse(r) for r in client.batch_lookup(endpoint, ids, select)]
    return pd.DataFrame(rows)


def fetch_sources(client, ids) -> pd.DataFrame:
    return _lookup(client, "sources", ids, "id,display_name,summary_stats", lambda r: {
        "source_id": short_id(r["id"]), "name": r.get("display_name"),
        "citedness": (r.get("summary_stats") or {}).get("2yr_mean_citedness")})


def fetch_authors(client, ids) -> pd.DataFrame:
    return _lookup(client, "authors", ids, "id,summary_stats", lambda r: {
        "author_id": short_id(r["id"]), "h_index": (r.get("summary_stats") or {}).get("h_index")})


def fetch_institutions(client, ids) -> pd.DataFrame:
    return _lookup(client, "institutions", ids, "id,display_name,summary_stats", lambda r: {
        "inst_id": short_id(r["id"]), "name": r.get("display_name"),
        "h_index": (r.get("summary_stats") or {}).get("h_index")})


def early_lookup_ids(early: pd.DataFrame) -> dict[str, set]:
    """IDs of sources, coauthors and institutions referenced by the early-window works."""
    return {
        "sources": set(early.source_id.dropna()),
        "authors": {a for lst in early.coauthors for a in lst},
        "institutions": {i for lst in early.institutions for i in lst},
    }
