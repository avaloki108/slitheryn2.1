"""
Hybrid in-memory + file-cache vector store for contract embeddings.

Provides similarity search to support RAG workflows.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple


class VectorStore:
    """
    Minimal vector store to manage contract embeddings.
    """

    def __init__(self):
        # In-memory store: {contract_name: (embedding, metadata)}
        self._store: Dict[str, Tuple[List[float], Dict]] = {}

    # Public API -----------------------------------------------------------
    def add_contract(self, contract_name: str, embedding: List[float], metadata: Dict) -> None:
        self._store[contract_name] = (embedding, metadata)

    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        results = []
        for name, (embedding, metadata) in self._store.items():
            score = self._cosine_similarity(query_embedding, embedding)
            results.append({"name": name, "score": score, "metadata": metadata})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_context_for_contract(self, contract_name: str, top_k: int = 3) -> List[str]:
        if contract_name not in self._store:
            return []
        query_emb, _ = self._store[contract_name]
        sims = self.search_similar(query_emb, top_k + 1)  # include self
        context = []
        for item in sims:
            if item["name"] == contract_name:
                continue
            ctx = item["metadata"].get("code_snippet") or item["metadata"].get("code")
            if ctx:
                context.append(ctx)
            if len(context) >= top_k:
                break
        return context

    def save_to_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = [
            {"name": name, "embedding": emb, "metadata": meta}
            for name, (emb, meta) in self._store.items()
        ]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_from_cache(self, cache_path: str) -> None:
        if not os.path.exists(cache_path):
            return
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            self._store[entry["name"]] = (entry["embedding"], entry.get("metadata", {}))

    def clear_cache(self) -> None:
        self._store.clear()

    # Internal helpers -----------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


