import pytest

from slitheryn.ai.ollama_client import OllamaClient


class DummyCfg:
    primary_model = "devstral-small-2:24b"
    reasoning_model = "gpt-oss:20b"
    comprehensive_model = "qwen3-coder:30b"


class DummyManager:
    def __init__(self):
        self.config = DummyCfg()

    def get_ollama_url(self):
        return "http://localhost:11434"


@pytest.fixture
def client():
    return OllamaClient(config_manager=DummyManager())


def test_parse_ai_response_basic(client):
    response = """
    Vulnerability: Reentrancy
    Severity: High
    Attack scenario: step 1 call withdraw, step 2 reenter
    Fix: use reentrancy guard
    """
    parsed = client._parse_ai_response(response)  # pylint: disable=protected-access
    assert "reentrancy" in parsed["vulnerabilities"]
    assert parsed["severity_scores"].get("reentrancy") == "High"
    assert parsed["attack_scenarios"]
    assert parsed["fix_recommendations"]
    assert 0 <= parsed["confidence_score"] <= 1


def test_extract_attack_scenarios(client):
    response = "Step 1: do X\nThen attacker does Y\nFinally drain funds"
    scenarios = client._extract_attack_scenarios(response)  # pylint: disable=protected-access
    assert scenarios


def test_extract_fix_recommendations(client):
    response = "Fix: add onlyOwner\nRecommend adding reentrancy guard"
    fixes = client._extract_fix_recommendations(response)  # pylint: disable=protected-access
    assert len(fixes) >= 1


def test_confidence_score_bounds(client):
    score = client._calculate_confidence_score("short text", ["reentrancy"])  # pylint: disable=protected-access
    assert 0 <= score <= 1

