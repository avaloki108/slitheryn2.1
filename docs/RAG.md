# RAG Integration Overview

Slitheryn now supports Retrieval-Augmented Generation (RAG) using Ollama's
`qwen3-embedding:4b` model. Contracts are embedded before analysis, enabling
context-aware prompts, similarity search, and cross-contract vulnerability
detection.

## Components
- `EmbeddingService` (`slitheryn/ai/embedding_service.py`): creates embeddings via Ollama.
- `VectorStore` (`slitheryn/ai/vector_store.py`): hybrid in-memory + file cache, similarity search.
- RAG-aware prompts (`slitheryn/ai/ollama_client.py`): include similar contract snippets.
- Cross-contract context in AI detector (`slitheryn/detectors/ai/ai_enhanced_analysis.py`).

## Configuration
Configured in `slitheryn/ai/config.py`:
- `enable_rag` (default: True)
- `embedding_model` (default: `qwen3-embedding:4b`)
- `similarity_threshold`, `max_similar_contracts`
- `cache_embeddings`, `cache_path`

## Workflow
1. Slither embeds contracts during initialization (if RAG enabled) and caches embeddings.
2. AI prompts fetch similar contract snippets from the vector store.
3. AI detector records similar contracts in results for triage.

## Testing
- Unit tests: `tests/unit/ai/test_embedding_service.py`, `tests/unit/ai/test_vector_store.py`
- RAG prompt integration: `tests/e2e/ai/test_rag_integration.py`
- Detector integration: `tests/e2e/detectors/test_ai_detector.py`

