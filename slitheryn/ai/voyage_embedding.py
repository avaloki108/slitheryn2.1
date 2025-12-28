"""
Voyage AI embedding service for contract code.

Uses Voyage AI's voyage-3.5 model for high-quality embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("Slitheryn.AI.VoyageEmbedding")


class VoyageEmbeddingService:
    """
    Embedding client using Voyage AI's API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-3.5",
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = model
        self._client = None

        if not self.api_key:
            logger.warning("No Voyage API key provided. Set VOYAGE_API_KEY env var or pass api_key.")

    def _get_client(self):
        """Lazy-load the Voyage client."""
        if self._client is None:
            try:
                import voyageai
                self._client = voyageai.Client(api_key=self.api_key)
            except ImportError:
                logger.error("voyageai package not installed. Run: pip install voyageai")
                raise
        return self._client

    def get_embedding_model(self) -> str:
        """Return the embedding model name."""
        return self.model

    def check_model_availability(self) -> bool:
        """Check if the API is accessible."""
        if not self.api_key:
            logger.warning("No Voyage API key set")
            return False
        try:
            client = self._get_client()
            # Test with a minimal embedding
            result = client.embed(
                texts=["test"],
                model=self.model,
            )
            return result.embeddings is not None and len(result.embeddings) > 0
        except Exception as exc:
            logger.warning("Failed checking Voyage availability: %s", exc)
            return False

    def embed_contract(self, contract_code: str, contract_name: str = "") -> Optional[List[float]]:
        """
        Embed a single contract source string.
        """
        prompt = f"Contract: {contract_name}\n\n{contract_code}" if contract_name else contract_code
        return self._embed_text(prompt)

    def embed_batch(self, contracts: List[Dict[str, str]]) -> Dict[str, List[float]]:
        """
        Embed multiple contracts efficiently using batch API.

        contracts: list of dicts with keys: name, code
        """
        if not contracts:
            return {}

        results: Dict[str, List[float]] = {}

        # Prepare texts and names
        texts = []
        names = []
        for item in contracts:
            name = item.get("name", "")
            code = item.get("code", "")
            prompt = f"Contract: {name}\n\n{code}" if name else code
            texts.append(prompt)
            names.append(name or str(len(names)))

        # Batch embed (Voyage supports batch in single call)
        try:
            client = self._get_client()
            result = client.embed(
                texts=texts,
                model=self.model,
            )

            for i, emb in enumerate(result.embeddings):
                if emb:
                    results[names[i]] = emb

        except Exception as exc:
            logger.error("Batch embedding request failed: %s", exc)
            # Fallback to individual embeddings
            for item in contracts:
                name = item.get("name", "")
                code = item.get("code", "")
                emb = self.embed_contract(code, name)
                if emb:
                    results[name or str(len(results))] = emb

        return results

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text string."""
        try:
            client = self._get_client()
            result = client.embed(
                texts=[text],
                model=self.model,
            )

            if result.embeddings and len(result.embeddings) > 0:
                return result.embeddings[0]

            logger.warning("Embedding missing in response")
        except Exception as exc:
            logger.error("Embedding request failed: %s", exc)
        return None
