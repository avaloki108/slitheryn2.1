import asyncio

import pytest

from integrations.web3_audit_system.orchestrator import MultiAgentOrchestrator


class DummyClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.timeout = 5

    def check_model_availability(self, model):
        return True

    def get_best_available_model(self):
        return "test-model"

    def _parse_ai_response(self, response):
        return {
            "vulnerabilities": ["reentrancy"],
            "severity_scores": {"reentrancy": "High"},
            "confidence_score": 0.9,
            "attack_scenarios": ["step 1, step 2"],
            "fix_recommendations": ["use guard"],
        }


@pytest.mark.asyncio
async def test_multi_agent_orchestrator_runs():
    client = DummyClient()
    orchestrator = MultiAgentOrchestrator(client, {"parallel_analysis": False})

    result = await orchestrator.analyze_contract("contract code", "Test")
    assert result
    assert result.consensus_vulnerabilities is not None

