"""
Embedding service for contract code using Ollama qwen3-embedding:4b.

Provides single and batch embedding helpers to support RAG workflows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("Slitheryn.AI.EmbeddingService")


class EmbeddingService:
    """
    Lightweight embedding client built on Ollama's embedding API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def get_embedding_model(self) -> str:
        """Return the embedding model name."""
        return self.model

    def check_model_availability(self) -> bool:
        """Check if the embedding model is available on Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m.get("name") == self.model for m in models)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed checking model availability: %s", exc)
            return False

    def embed_contract(self, contract_code: str, contract_name: str = "") -> Optional[List[float]]:
        """
        Embed a single contract source string.
        """
        prompt = f"Contract: {contract_name}\n\n{contract_code}" if contract_name else contract_code
        return self._embed_text(prompt)

    def embed_batch(self, contracts: List[Dict[str, str]]) -> Dict[str, List[float]]:
        """
        Embed multiple contracts in one call.

        contracts: list of dicts with keys: name, code
        """
        results: Dict[str, List[float]] = {}
        for item in contracts:
            name = item.get("name", "")
            code = item.get("code", "")
            emb = self.embed_contract(code, name)
            if emb:
                results[name or str(len(results))] = emb
        return results

    # Internal helpers -----------------------------------------------------
    def _embed_text(self, text: str) -> Optional[List[float]]:
        payload = {"model": self.model, "prompt": text}
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                return embedding
            logger.warning("Embedding missing in response")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Embedding request failed: %s", exc)
        return None


