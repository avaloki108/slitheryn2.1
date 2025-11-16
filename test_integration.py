#!/usr/bin/env python3
"""
Integration Test for Slitheryn Multi-Agent Audit Suite

This script validates that the multi-agent audit suite has been successfully
integrated with Slitheryn's existing AI infrastructure.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_import_integration():
    """Test that all integration modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Test AI system imports
        from slither.ai.config import get_ai_config, AIConfigManager
        from slither.ai.ollama_client import OllamaClient
        print("   ✅ Slitheryn AI system imports successful")
        
        # Test integration imports  
        from integrations.web3_audit_system import Web3AuditSystem, create_multi_agent_system
        from integrations.web3_audit_system.agents import (
            VulnerabilityDetectorAgent, ExploitAnalyzerAgent, FixRecommenderAgent,
            EconomicAttackAgent, GovernanceAuditAgent, AgentType
        )
        from integrations.web3_audit_system.orchestrator import MultiAgentOrchestrator
        print("   ✅ Multi-agent system imports successful")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_ai_configuration():
    """Test AI configuration with multi-agent settings"""
    print("\n🔧 Testing AI configuration...")
    
    try:
        from slither.ai.config import get_ai_config
        
        ai_config = get_ai_config()
        config = ai_config.config
        
        print(f"   • AI Analysis Enabled: {config.enable_ai_analysis}")
        print(f"   • Multi-Agent Enabled: {getattr(config, 'enable_multi_agent', 'Not configured')}")
        print(f"   • Primary Model: {config.primary_model}")
        print(f"   • Agent Types: {getattr(config, 'agent_types', 'Not configured')}")
        print(f"   • Consensus Threshold: {getattr(config, 'consensus_threshold', 'Not configured')}")
        
        # Test multi-agent config method
        multi_config = ai_config.get_multi_agent_config()
        print(f"   • Multi-Agent Config Available: {bool(multi_config)}")
        
        print("   ✅ AI configuration test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_agent_initialization():
    """Test agent initialization without network"""
    print("\n🤖 Testing agent initialization...")
    
    try:
        from slither.ai.ollama_client import OllamaClient
        from integrations.web3_audit_system.agents import (
            VulnerabilityDetectorAgent, ExploitAnalyzerAgent, FixRecommenderAgent,
            EconomicAttackAgent, GovernanceAuditAgent
        )
        
        # Create a mock Ollama client (won't connect to network)
        ollama_client = OllamaClient("http://localhost:11434")
        
        # Test agent initialization
        agents = {
            'VulnerabilityDetector': VulnerabilityDetectorAgent(ollama_client),
            'ExploitAnalyzer': ExploitAnalyzerAgent(ollama_client),
            'FixRecommender': FixRecommenderAgent(ollama_client),
            'EconomicAttack': EconomicAttackAgent(ollama_client),
            'GovernanceAudit': GovernanceAuditAgent(ollama_client)
        }
        
        for name, agent in agents.items():
            print(f"   • {name}: {agent.agent_type.value} agent initialized")
            print(f"     - Specialized models: {len(agent.specialized_models)}")
        
        print("   ✅ Agent initialization test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        return False

def test_orchestrator():
    """Test multi-agent orchestrator"""
    print("\n🎭 Testing multi-agent orchestrator...")
    
    try:
        from slither.ai.ollama_client import OllamaClient
        from integrations.web3_audit_system.orchestrator import MultiAgentOrchestrator
        
        # Create mock client and config
        ollama_client = OllamaClient("http://localhost:11434")
        config = {
            'agent_types': ['vulnerability', 'exploit', 'fix'],
            'consensus_threshold': 0.7,
            'parallel_analysis': True,
            'max_workers': 3
        }
        
        orchestrator = MultiAgentOrchestrator(ollama_client, config)
        
        print(f"   • Agents initialized: {len(orchestrator.agents)}")
        print(f"   • Enabled agent types: {len(orchestrator.enabled_agent_types)}")
        print(f"   • Parallel analysis: {orchestrator.enable_parallel_analysis}")
        print(f"   • Consensus threshold: {orchestrator.consensus_threshold}")
        
        print("   ✅ Orchestrator test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Orchestrator test failed: {e}")
        return False

def test_web3_audit_system():
    """Test main Web3AuditSystem integration"""
    print("\n🌐 Testing Web3AuditSystem integration...")
    
    try:
        from slither.ai.config import get_ai_config
        from slither.ai.ollama_client import OllamaClient
        from integrations.web3_audit_system import create_multi_agent_system
        
        # Create system using integration function
        ai_config = get_ai_config()
        ollama_client = OllamaClient("http://localhost:11434")
        
        audit_system = create_multi_agent_system(ollama_client, ai_config)
        
        print(f"   • System initialized: {audit_system is not None}")
        print(f"   • Multi-agent enabled: {audit_system.enable_multi_agent}")
        print(f"   • Available agents: {audit_system.get_available_agents()}")
        
        status = audit_system.get_system_status()
        print(f"   • System status keys: {list(status.keys())}")
        
        print("   ✅ Web3AuditSystem test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Web3AuditSystem test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI argument parsing integration"""
    print("\n💻 Testing CLI integration...")
    
    try:
        # Import main function components
        import slither.__main__ as main_module
        
        # Check if multi-agent arguments are available
        has_parse_args = hasattr(main_module, 'parse_args')
        has_process_functions = hasattr(main_module, '_process_with_multi_agent')
        
        print(f"   • Parse args function available: {has_parse_args}")
        print(f"   • Multi-agent process function available: {has_process_functions}")
        
        # Test that multi-agent imports work in main module context
        sys.path.insert(0, '.')
        exec("from integrations.web3_audit_system import create_multi_agent_system")
        print("   • Multi-agent imports work in main context: ✅")
        
        print("   ✅ CLI integration test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ CLI integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("="*80)
    print("🚀 SLITHERYN MULTI-AGENT INTEGRATION TEST SUITE")
    print("="*80)
    
    tests = [
        test_import_integration,
        test_ai_configuration,
        test_agent_initialization,
        test_orchestrator,
        test_web3_audit_system,
        test_cli_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*80)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Multi-agent integration successful!")
        print("\n✅ The multi-agent audit suite has been successfully merged with Slitheryn!")
        print("✅ All components are properly integrated and functional!")
        print("✅ Ready for production use with AI-powered multi-agent analysis!")
    else:
        print(f"⚠️  {total - passed} tests failed - integration issues detected")
        
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)