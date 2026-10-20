import pytest

from src import api, config


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p, self.headers, self.text = status, payload or {}, {}, ""

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def make_client(tmp_path, responses):
    c = api.OpenAlexClient(cache_dir=tmp_path, api_key="k", max_req_per_sec=1000)
    c.session = FakeSession(responses)
    c.budget_check_every = 0
    return c


def test_cache_key_ignores_key_and_order():
    a = api.OpenAlexClient.cache_key("works", {"filter": "x", "api_key": "1", "select": "id"})
    b = api.OpenAlexClient.cache_key("works", {"select": "id", "filter": "x", "api_key": "2"})
    assert a == b


def test_second_call_is_served_from_disk(tmp_path):
    c = make_client(tmp_path, [FakeResp(200, {"results": [1]})])
    assert c.get("works", {"filter": "a"}) == {"results": [1]}
    assert c.get("works", {"filter": "a"}) == {"results": [1]}
    assert c.network_calls == 1 and c.cache_hits == 1


def test_retries_on_429_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    c = make_client(tmp_path, [FakeResp(429), FakeResp(503), FakeResp(200, {"ok": 1})])
    assert c.get("works", {"filter": "b"}) == {"ok": 1}


def test_client_error_is_not_retried(tmp_path):
    c = make_client(tmp_path, [FakeResp(400)])
    with pytest.raises(RuntimeError):
        c.get("works", {"filter": "c"})


def test_budget_exhausted_raises(tmp_path):
    c = make_client(tmp_path, [FakeResp(200, {"credits": {"remaining_usd": 0.001}})])
    with pytest.raises(api.BudgetExhausted):
        c.check_budget()


def test_paginate_follows_cursor(tmp_path):
    pages = [FakeResp(200, {"results": [1, 2], "meta": {"next_cursor": "n"}}),
             FakeResp(200, {"results": [3], "meta": {"next_cursor": None}})]
    c = make_client(tmp_path, pages)
    assert list(c.paginate("works", {"filter": "d"})) == [1, 2, 3]


def test_batch_lookup_chunks_of_50(tmp_path):
    ids = [f"A{i}" for i in range(120)]
    c = make_client(tmp_path, [FakeResp(200, {"results": [], "meta": {}}) for _ in range(3)])
    list(c.batch_lookup("authors", ids, "id"))
    assert len(c.session.calls) == 3
    assert c.session.calls[0][1]["filter"].count("|") == config.BATCH_IDS - 1
