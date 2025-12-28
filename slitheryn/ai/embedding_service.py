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
    Supports chunking for large contracts that exceed model context limits.
    """

    # Max characters per chunk (conservative estimate for token limit)
    MAX_CHUNK_SIZE = 8000

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
        timeout: int = 120,
        max_chunk_size: int = 8000,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_chunk_size = max_chunk_size

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
        For large contracts, chunks the code and averages the embeddings.
        """
        prompt = f"Contract: {contract_name}\n\n{contract_code}" if contract_name else contract_code

        # If the prompt is small enough, embed directly
        if len(prompt) <= self.max_chunk_size:
            return self._embed_text(prompt)

        # For large contracts, chunk and average embeddings
        logger.info(f"Contract {contract_name} is large ({len(prompt)} chars), chunking...")
        chunks = self._chunk_text(prompt)
        embeddings = []

        for i, chunk in enumerate(chunks):
            emb = self._embed_text(chunk)
            if emb:
                embeddings.append(emb)
            else:
                logger.warning(f"Failed to embed chunk {i+1}/{len(chunks)} for {contract_name}")

        if not embeddings:
            return None

        # Average the embeddings
        return self._average_embeddings(embeddings)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks, trying to break at natural boundaries.
        """
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + self.max_chunk_size, len(text))

            # If not at end, try to find a natural break point
            if end_pos < len(text):
                # Look for function boundaries, newlines, etc.
                for break_char in ['\n\n', '\n    function ', '\n}', '\n']:
                    break_pos = text.rfind(break_char, current_pos, end_pos)
                    if break_pos > current_pos:
                        end_pos = break_pos + len(break_char)
                        break

            chunk = text[current_pos:end_pos]
            if chunk.strip():
                chunks.append(chunk)
            current_pos = end_pos

        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """
        Average multiple embeddings into one.
        """
        if not embeddings:
            return []
        if len(embeddings) == 1:
            return embeddings[0]

        dim = len(embeddings[0])
        avg = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                avg[i] += val
        for i in range(dim):
            avg[i] /= len(embeddings)
        return avg

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


