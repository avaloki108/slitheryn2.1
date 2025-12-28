"""
OpenAI embedding service for contract code.

Uses OpenAI's text-embedding-3-small/large models for reliable embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("Slitheryn.AI.OpenAIEmbedding")


class OpenAIEmbeddingService:
    """
    Embedding client using OpenAI's API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client = None

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key.")

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                raise
        return self._client

    def get_embedding_model(self) -> str:
        """Return the embedding model name."""
        return self.model

    def check_model_availability(self) -> bool:
        """Check if the API is accessible."""
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            # Test with a minimal embedding
            response = client.embeddings.create(
                model=self.model,
                input="test",
            )
            return response.data is not None and len(response.data) > 0
        except Exception as exc:
            logger.warning("Failed checking OpenAI availability: %s", exc)
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

        # Batch embed (OpenAI supports batch in single call)
        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model,
                input=texts,
            )

            for i, emb_data in enumerate(response.data):
                if emb_data.embedding:
                    results[names[i]] = emb_data.embedding

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
            response = client.embeddings.create(
                model=self.model,
                input=text,
            )

            if response.data and len(response.data) > 0:
                return response.data[0].embedding

            logger.warning("Embedding missing in response")
        except Exception as exc:
            logger.error("Embedding request failed: %s", exc)
        return None
