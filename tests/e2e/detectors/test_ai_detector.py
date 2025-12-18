import logging
import tempfile
from types import SimpleNamespace

from slitheryn.detectors.ai.ai_enhanced_analysis import AIEnhancedAnalysis
from slitheryn.ai.ollama_client import AIAnalysisResult


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.vector_store = None
        self.max_similar_contracts = 3

    def analyze_contract(self, contract_code: str, contract_name: str, analysis_type: str):
        return AIAnalysisResult(
            vulnerabilities=["reentrancy"],
            severity_scores={"reentrancy": "High"},
            attack_scenarios=["step 1, step 2"],
            fix_recommendations=["use reentrancy guard"],
            confidence_score=0.9,
            analysis_time=0.1,
            model_used="test-model",
            raw_response="",
        )


def make_contract(tmp_path):
    solidity = "pragma solidity ^0.8.0; contract A { function f() public {} }"
    path = tmp_path / "A.sol"
    path.write_text(solidity)

    filename = SimpleNamespace(absolute=str(path))
    source_mapping = SimpleNamespace(filename=filename, start=None, length=None)

    contract = SimpleNamespace(
        name="A",
        is_interface=False,
        is_library=False,
        source_mapping=source_mapping,
        state_variables=[],
        functions=[],
    )
    return contract


def test_ai_detector_runs(monkeypatch, tmp_path):
    # Avoid real network calls
    monkeypatch.setattr(
        "slitheryn.detectors.ai.ai_enhanced_analysis.OllamaClient", DummyClient
    )

    contract = make_contract(tmp_path)
    comp_unit = SimpleNamespace(contracts=[contract])
    slither = SimpleNamespace(_vector_store=None)

    detector = AIEnhancedAnalysis(comp_unit, slither, logging.getLogger("test"))
    outputs = detector._detect()  # pylint: disable=protected-access
    assert outputs
    assert "reentrancy" in " ".join(outputs[0].data[0].lower() for _ in [0])

