"""
RAG similarity helper utilities.

Uses the VectorStore populated during initialization to find similar contracts
and surface potential duplicated patterns.
"""

from __future__ import annotations

from typing import List, Dict

from slitheryn.ai.vector_store import VectorStore


def find_similar_contracts(
    vector_store: VectorStore, contract_name: str, top_k: int = 5, min_score: float = 0.0
) -> List[Dict]:
    """
    Return similar contracts sorted by similarity score.
    """
    results = vector_store.get_context_for_contract(contract_name, top_k)
    # get_context_for_contract returns snippets; to include scores we re-run search
    search = vector_store.search_similar(
        vector_store._store.get(contract_name, ([], {}))[0], top_k + 1  # type: ignore
    )
    filtered = [
        {"name": r["name"], "score": r["score"], "metadata": r["metadata"]}
        for r in search
        if r["name"] != contract_name and r["score"] >= min_score
    ]
    return filtered[:top_k]


