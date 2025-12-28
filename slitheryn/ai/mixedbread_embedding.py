"""
Mixedbread embedding service for contract code.

Uses the Mixedbread API (mxbai-embed-large-v1) for high-quality embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("Slitheryn.AI.MixedbreadEmbedding")


class MixedbreadEmbeddingService:
    """
    Embedding client using Mixedbread's API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        normalized: bool = True,
    ):
        self.api_key = api_key or os.getenv("MXBAI_API_KEY")
        self.model = model
        self.normalized = normalized
        self._client = None

        if not self.api_key:
            logger.warning("No Mixedbread API key provided. Set MXBAI_API_KEY env var or pass api_key.")

    def _get_client(self):
        """Lazy-load the Mixedbread client."""
        if self._client is None:
            try:
                from mixedbread import Mixedbread
                self._client = Mixedbread(api_key=self.api_key)
            except ImportError:
                logger.error("mixedbread package not installed. Run: pip install mixedbread")
                raise
        return self._client

    def get_embedding_model(self) -> str:
        """Return the embedding model name."""
        return self.model

    def check_model_availability(self) -> bool:
        """Check if the API is accessible."""
        if not self.api_key:
            logger.warning("No Mixedbread API key set")
            return False
        try:
            client = self._get_client()
            # Test with a minimal embedding
            response = client.embed(
                model=self.model,
                input=["test"],
                normalized=self.normalized,
                encoding_format="float",
            )
            return response.data is not None and len(response.data) > 0
        except Exception as exc:
            # Log more details for debugging
            exc_str = str(exc)
            if "503" in exc_str or "504" in exc_str:
                logger.warning("Mixedbread API temporarily unavailable (503/504)")
            elif "401" in exc_str or "403" in exc_str:
                logger.warning("Mixedbread API key invalid or unauthorized")
            else:
                logger.warning("Failed checking Mixedbread availability: %s", exc)
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

        # Batch embed (Mixedbread supports batch in single call)
        try:
            client = self._get_client()
            response = client.embed(
                model=self.model,
                input=texts,
                normalized=self.normalized,
                encoding_format="float",
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
            response = client.embed(
                model=self.model,
                input=[text],
                normalized=self.normalized,
                encoding_format="float",
            )

            if response.data and len(response.data) > 0:
                return response.data[0].embedding

            logger.warning("Embedding missing in response")
        except Exception as exc:
            logger.error("Embedding request failed: %s", exc)
        return None
