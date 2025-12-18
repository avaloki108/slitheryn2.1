from slitheryn.ai.ollama_client import OllamaClient
from slitheryn.ai.vector_store import VectorStore


class DummyCfg:
    primary_model = "devstral-small-2:24b"
    reasoning_model = "gpt-oss:20b"
    comprehensive_model = "qwen3-coder:30b"


class DummyManager:
    def __init__(self):
        self.config = DummyCfg()

    def get_ollama_url(self):
        return "http://localhost:11434"


def test_rag_context_in_prompt():
    store = VectorStore()
    store.add_contract("A", [1.0, 0.0], {"code_snippet": "contract A { }"})
    store.add_contract("B", [0.9, 0.1], {"code_snippet": "contract B { }"})

    client = OllamaClient(config_manager=DummyManager(), vector_store=store)
    prompt = client._build_security_analysis_prompt("contract A { }", "A", "comprehensive")  # pylint: disable=protected-access
    assert "Context from similar contracts" in prompt

