import types

import pytest

from slitheryn.ai.embedding_service import EmbeddingService


class DummyResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError("error")


@pytest.fixture
def mock_requests(monkeypatch):
    calls = types.SimpleNamespace(posts=[], gets=[])

    def fake_get(url, timeout=0):
        calls.gets.append(url)
        return DummyResponse({"models": [{"name": "qwen3-embedding:4b"}]})

    def fake_post(url, json=None, timeout=0):
        calls.posts.append((url, json))
        return DummyResponse({"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)
    return calls


def test_check_model(mock_requests):
    svc = EmbeddingService()
    assert svc.check_model_availability()
    assert mock_requests.gets


def test_embed_contract(mock_requests):
    svc = EmbeddingService()
    emb = svc.embed_contract("contract code", "Test")
    assert emb and len(emb) == 3
    assert mock_requests.posts

