"""End-to-end run against a fake in-memory OpenAlex. Synthetic data: validates plumbing only."""
import numpy as np

from src import api, config, pipeline


class FakeOpenAlex(api.OpenAlexClient):
    def __init__(self, n_authors=700, seed=0):
        self.network_calls = self.cache_hits = 0
        self.api_key = "fake"
        self.start_remaining_usd = self.last_remaining_usd = 1.0
        r = np.random.default_rng(seed)
        self.works = []
        self.top_co = {f"A{9000 + i}" for i in range(20)}
        wid = 0
        for a in range(n_authors):
            t0 = int(r.integers(2007, 2013))
            boost = r.random() < 0.3
            for _ in range(int(r.integers(2, 6))):
                y = int(r.integers(t0, t0 + 3))
                self._add(wid, f"A{a}", y, r, boost); wid += 1
            for _ in range(int(r.integers(0, 12))):
                y = int(r.integers(t0 + 3, min(t0 + 15, 2026)))
                self._add(wid, f"A{a}", y, r, boost); wid += 1

    def _add(self, wid, aid, year, r, boost):
        co = [f"A{9000 + int(i)}" for i in r.integers(0, 200, int(r.integers(0, 3)))]
        if boost:
            co.append(sorted(self.top_co)[int(r.integers(0, 20))])
        ids = [aid] + co
        self.works.append({"id": f"https://openalex.org/W{wid}", "publication_year": year,
                           "cited_by_count": int(r.poisson(3 + 20 * boost)),
                           "primary_location": {"source": {"id": f"https://openalex.org/S{int(r.integers(0, 20))}"}},
                           "primary_topic": {"id": f"https://openalex.org/T{int(r.integers(0, 6))}"},
                           "authorships": [{"author": {"id": f"https://openalex.org/{i}"},
                                            "author_position": "first" if k == 0 else "middle",
                                            "institutions": [{"id": f"https://openalex.org/I{int(r.integers(0, 12))}"}]}
                                           for k, i in enumerate(ids)]})

    def check_budget(self, verbose=False):
        return 1.0

    def get(self, endpoint, params):
        self.network_calls += 1
        f = params.get("filter", "")
        if endpoint == "works" and "sample" in params:
            res = [w for w in self.works if 2008 <= w["publication_year"] <= 2011][:3000]
            page = int(params["page"])
            return {"results": res[(page - 1) * 100: page * 100], "meta": {}}
        if endpoint == "works":
            ids = set(f.split("authorships.author.id:")[1].split(",")[0].split("|"))
            yr = f.split("publication_year:")[1].split(",")[0]
            lo, hi = (0, int(yr[1:]) - 1) if yr.startswith("<") else map(int, yr.split("-"))
            res = [w for w in self.works if lo <= w["publication_year"] <= hi and
                   any(a["author"]["id"].rsplit("/", 1)[-1] in ids for a in w["authorships"])]
            return {"results": res, "meta": {"next_cursor": None}}
        ids = f.split("openalex:")[1].split("|")
        mk = {"sources": lambda i: {"id": i, "display_name": "Neural Information Processing Systems" if i.endswith("S1") else f"J {i}",
                                    "summary_stats": {"2yr_mean_citedness": float(hash(i) % 7)}},
              "authors": lambda i: {"id": i, "summary_stats": {"h_index": 90 if i.rsplit("/", 1)[-1] in self.top_co else hash(i) % 30}},
              "institutions": lambda i: {"id": i, "display_name": i, "summary_stats": {"h_index": hash(i) % 200}}}[endpoint]
        return {"results": [mk(f"https://openalex.org/{i}") for i in ids], "meta": {"next_cursor": None}}


def test_full_pipeline_on_fake_api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "DRY_RUN_AUTHORS", 5000)
    monkeypatch.setattr(config, "DRY_RUN_SEED_WORKS", 3000)
    monkeypatch.setattr(config, "TARGET_COHORT", (1, 10))
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(pipeline, "OpenAlexClient", lambda: FakeOpenAlex())
    assert pipeline.main(["--dry-run"]) == 0
    out = tmp_path / "out" / "dry_run"
    assert (out / "results.md").exists() and (out / "limitations.md").exists()
    assert len(list((out / "figures").glob("*.png"))) == 7
    txt = (out / "results.md").read_text(encoding="utf-8")
    assert "DRY RUN" in txt and "not supported" in txt or "supported" in txt
    assert (tmp_path / "data" / "cost_ledger.jsonl").exists()
