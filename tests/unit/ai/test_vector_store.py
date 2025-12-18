import os
import tempfile

from slitheryn.ai.vector_store import VectorStore


def test_add_and_search():
    store = VectorStore()
    store.add_contract("A", [1.0, 0.0], {"code_snippet": "code a"})
    store.add_contract("B", [0.9, 0.1], {"code_snippet": "code b"})

    results = store.search_similar([1.0, 0.0], top_k=2)
    assert results[0]["name"] == "A"
    assert len(results) == 2


def test_cache_roundtrip():
    store = VectorStore()
    store.add_contract("A", [0.1, 0.2], {"code": "sample"})

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "emb.json")
        store.save_to_cache(cache_path)

        new_store = VectorStore()
        new_store.load_from_cache(cache_path)
        results = new_store.search_similar([0.1, 0.2], top_k=1)
        assert results[0]["name"] == "A"

