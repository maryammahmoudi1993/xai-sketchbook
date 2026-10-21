"""OpenAlex client: disk cache for every raw response, global rate limit, backoff, budget guard.

Every response is stored as gzip-JSON keyed by a hash of (endpoint, sorted params without
the API key), so a re-run replays from disk and spends nothing. A budget stop raises
BudgetExhausted; re-running the pipeline the next day resumes from the cache.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

from . import config


class BudgetExhausted(RuntimeError):
    """Raised when the remaining daily OpenAlex budget is below config.MIN_BUDGET_USD."""


def _find_number(obj, keys):
    """Depth-first search for the first numeric value under any of `keys`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            r = _find_number(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_number(v, keys)
            if r is not None:
                return r
    return None


class OpenAlexClient:
    def __init__(self, cache_dir: Path | None = None, api_key: str | None = None,
                 max_req_per_sec: float = config.MAX_REQ_PER_SEC, budget_check_every: int = 200):
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.keys = [api_key] if api_key is not None else config.api_keys()
        self.key_idx = 0
        self.api_key = self.keys[0] if self.keys else None
        self._start_rem: dict[str, float] = {}
        self._last_rem: dict[str, float] = {}
        self.min_interval = 1.0 / max_req_per_sec
        self._last = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.network_calls = 0
        self.cache_hits = 0
        self.budget_check_every = budget_check_every
        self.last_remaining_usd: float | None = None

    # ---------- cache ----------
    @staticmethod
    def cache_key(endpoint: str, params: dict) -> str:
        clean = {k: str(v) for k, v in params.items() if k not in ("api_key", "mailto")}
        blob = json.dumps([endpoint, sorted(clean.items())])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json.gz"

    def _read(self, key: str):
        p = self._path(key)
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write(self, key: str, payload: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(p)

    # ---------- network ----------
    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _http_get(self, url: str, params: dict, retries: int = 6) -> dict:
        if not self.api_key:
            raise RuntimeError(f"Set the {config.API_KEY_ENV} environment variable (OpenAlex API key).")
        full = dict(params, api_key=self.api_key, mailto=config.MAILTO)
        delay = 1.0
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.session.get(url, params=full, timeout=60)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                self.network_calls += 1
                return r.json()
            status = r.status_code if r is not None else None
            if status is not None and status not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex {status} for {url} {params}: {r.text[:300]}")
            retry_after = r.headers.get("Retry-After") if r is not None else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"OpenAlex request failed after {retries} retries: {url} {params}")

    def get(self, endpoint: str, params: dict) -> dict:
        """Cached GET of API_BASE/endpoint. Budget is checked before real network calls."""
        key = self.cache_key(endpoint, params)
        hit = self._read(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        if self.budget_check_every and self.network_calls % self.budget_check_every == 0:
            self.check_budget()
        payload = self._http_get(f"{config.API_BASE}/{endpoint}", params)
        self._write(key, payload)
        return payload

    # ---------- budget ----------
    def rate_limit(self) -> dict:
        """Uncached /rate-limit call (never stored)."""
        return self._http_get(f"{config.API_BASE}/rate-limit", {})

    def _rotate(self) -> bool:
        """Switch to the next API key (each key has its own daily budget). False if none left."""
        with self._lock:
            if self.key_idx + 1 >= len(self.keys):
                return False
            self.key_idx += 1
            self.api_key = self.keys[self.key_idx]
        print(f"Budget of key #{self.key_idx} used up -> switching to key #{self.key_idx + 1} of {len(self.keys)}")
        return True

    def check_budget(self, verbose: bool = False) -> float | None:
        """Raise BudgetExhausted only when every configured key is below MIN_BUDGET_USD."""
        while True:
            info = self.rate_limit()
            remaining = None
            for name in ("daily_remaining_usd", "remaining_usd", "prepaid_remaining_usd"):
                remaining = _find_number(info, {name})
                if remaining is not None:
                    break
            if remaining is not None:
                self._start_rem.setdefault(self.api_key, remaining)
                self._last_rem[self.api_key] = remaining
            self.last_remaining_usd = remaining
            if verbose or remaining is None:
                print(f"OpenAlex /rate-limit (key #{self.key_idx + 1}):", json.dumps(info)[:400])
            if remaining is None or remaining >= config.MIN_BUDGET_USD:
                return remaining
            if not self._rotate():
                raise BudgetExhausted(f"All {len(self.keys)} key(s) below ${config.MIN_BUDGET_USD} remaining. "
                                      "Re-run after the daily reset (00:00 UTC); the cache makes the run resumable.")

    def spent_this_session(self) -> float | None:
        if not self._start_rem:
            return None
        return sum(self._start_rem[k] - self._last_rem[k] for k in self._start_rem)

    # ---------- helpers ----------
    def paginate(self, endpoint: str, params: dict, max_pages: int | None = None):
        """Yield result records following cursor pagination (every page cached)."""
        p = dict(params, per_page=config.PER_PAGE, cursor="*")
        pages = 0
        while True:
            data = self.get(endpoint, p)
            results = data.get("results", [])
            yield from results
            nxt = data.get("meta", {}).get("next_cursor")
            pages += 1
            if not results or not nxt or (max_pages and pages >= max_pages):
                return
            p["cursor"] = nxt

    def batch_lookup(self, endpoint: str, ids: list[str], select: str, id_filter: str = "openalex"):
        """Fetch many entities by ID using OR filters (<=50 IDs per call)."""
        ids = sorted(set(ids))
        for i in range(0, len(ids), config.BATCH_IDS):
            chunk = ids[i:i + config.BATCH_IDS]
            params = {"filter": f"{id_filter}:{'|'.join(chunk)}", "select": select}
            yield from self.paginate(endpoint, params)


def short_id(openalex_url: str | None) -> str | None:
    """'https://openalex.org/A123' -> 'A123'."""
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else None
