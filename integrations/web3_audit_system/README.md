# Web3 Multi-Agent Audit System

This directory contains the comprehensive multi-agent audit system that extends Slitheryn's AI-powered security analysis with specialized agent coordination.

## Overview

The Web3 Multi-Agent Audit System implements a sophisticated approach to smart contract security analysis by deploying multiple specialized AI agents, each with expertise in different vulnerability types and attack patterns. This approach provides:

- **Comprehensive Coverage**: Multiple agents analyze different aspects simultaneously
- **Specialized Expertise**: Each agent focuses on specific vulnerability domains
- **Consensus-Based Results**: Cross-validation between agents reduces false positives
- **Parallel Processing**: Simultaneous analysis for faster results
- **Economic & Governance Focus**: Specialized agents for DeFi and DAO security

## Architecture

### Core Components

1. **`agents.py`**: Specialized audit agents
   - `VulnerabilityDetectorAgent`: Common smart contract vulnerabilities
   - `ExploitAnalyzerAgent`: Attack scenario construction
   - `FixRecommenderAgent`: Detailed fix recommendations
   - `EconomicAttackAgent`: DeFi and economic vulnerabilities
   - `GovernanceAuditAgent`: DAO and governance security

2. **`orchestrator.py`**: Multi-agent coordination
   - Agent lifecycle management
   - Parallel/sequential execution control
   - Result aggregation and consensus
   - Confidence scoring and validation

3. **`__init__.py`**: Integration interface
   - Main `Web3AuditSystem` class
   - Integration with existing Slitheryn AI infrastructure
   - Configuration management
   - Compatibility functions

## Agent Specializations

### VulnerabilityDetectorAgent
**Focus**: Common smart contract vulnerabilities
- Reentrancy attacks
- Access control issues
- Integer overflow/underflow
- Uninitialized storage
- Time manipulation
- tx.origin usage
- Delegate call vulnerabilities

**Models**: SmartLLM-OG:latest, phi4-reasoning:latest

### ExploitAnalyzerAgent
**Focus**: Attack scenario construction and impact assessment
- Step-by-step exploit development
- Economic impact calculation
- Attack vector identification
- MEV vulnerability analysis
- Multi-transaction attack sequences

**Models**: phi4-reasoning:latest, qwen3:30b-a3b

### FixRecommenderAgent
**Focus**: Detailed remediation guidance
- Code-level fixes with examples
- Security pattern implementations
- Best practice recommendations
- Gas optimization opportunities
- Testing strategies

**Models**: SmartLLM-OG:latest, qwen3:30b-a3b

### EconomicAttackAgent
**Focus**: DeFi and financial vulnerabilities
- Flash loan attacks
- Price manipulation
- Oracle vulnerabilities
- Liquidity attacks
- Yield farming exploits
- Slippage and sandwich attacks

**Models**: qwen3:30b-a3b, phi4-reasoning:latest

### GovernanceAuditAgent
**Focus**: DAO and governance security
- Voting manipulation
- Proposal attacks
- Admin key risks
- Timelock vulnerabilities
- Quorum manipulation
- Centralization analysis

**Models**: phi4-reasoning:latest, SmartLLM-OG:latest

## Usage

### Basic Usage

```python
from integrations.web3_audit_system import Web3AuditSystem

# Initialize with existing Ollama client
audit_system = Web3AuditSystem(ollama_client, config)

# Perform multi-agent analysis
result = await audit_system.audit(contract_code, "MyContract")

# Generate comprehensive report
report = audit_system.generate_report(result)
print(report)
```

### Integration with Slitheryn

```python
from integrations.web3_audit_system import create_multi_agent_system, run_multi_agent_analysis

# Create system using existing AI infrastructure
audit_system = create_multi_agent_system(ollama_client, ai_config_manager)

# Run analysis
result = await run_multi_agent_analysis(
    contract_code, 
    "MyContract", 
    ollama_client, 
    ai_config_manager
)
```

### Configuration

The multi-agent system extends Slitheryn's AI configuration:

```json
{
  "enable_multi_agent": true,
  "agent_types": ["vulnerability", "exploit", "fix", "economic", "governance"],
  "consensus_threshold": 0.7,
  "parallel_analysis": true,
  "max_workers": 4
}
```

## Analysis Types

### Comprehensive Analysis (Default)
Uses all available agents for thorough security assessment:
```python
result = await audit_system.audit(contract_code, analysis_type="comprehensive")
```

### Quick Analysis
Uses core vulnerability detection and exploit analysis:
```python
result = await audit_system.audit(contract_code, analysis_type="quick")
```

### Specialized Analysis
Focuses on economic and governance vulnerabilities:
```python
result = await audit_system.audit(contract_code, analysis_type="specialized")
```

## Result Structure

The `MultiAgentAnalysisResult` provides:

```python
@dataclass
class MultiAgentAnalysisResult:
    consensus_vulnerabilities: List[str]       # Agreed-upon vulnerabilities
    agent_results: List[AgentResult]          # Individual agent findings
    confidence_matrix: Dict[str, Dict]        # Cross-agent confidence scores
    final_severity_scores: Dict[str, str]     # Consensus severity levels
    attack_scenarios: List[str]               # Detailed attack scenarios
    fix_recommendations: List[str]            # Actionable fix suggestions
    economic_impact_assessment: Dict         # Financial impact analysis
    governance_risks: List[str]              # Governance vulnerabilities
    total_analysis_time: float               # Total processing time
    models_used: List[str]                   # AI models utilized
    consensus_score: float                   # Overall analysis confidence
```

## Consensus Algorithm

The system uses a sophisticated consensus algorithm:

1. **Vulnerability Voting**: Each agent votes on vulnerabilities with confidence weights
2. **Threshold Application**: Only vulnerabilities exceeding consensus threshold are included
3. **Severity Calculation**: Weighted voting determines final severity levels
4. **Confidence Scoring**: Overall analysis confidence based on agent agreement

## Integration Features

- **Backward Compatibility**: Fully compatible with existing Slitheryn AI system
- **Configuration Extension**: Seamlessly extends current AI configuration
- **Model Sharing**: Utilizes existing Ollama client and model infrastructure
- **Logging Integration**: Uses Slitheryn's logging system
- **Error Handling**: Robust error handling with graceful degradation

## Performance Characteristics

- **Parallel Processing**: Up to 5 agents running simultaneously
- **Model Optimization**: Dynamic model selection based on agent requirements
- **Timeout Management**: Configurable timeouts per agent
- **Resource Management**: Controlled resource usage with worker limits

## Security Considerations

- **Model Isolation**: Each agent uses specialized prompts to prevent cross-contamination
- **Result Validation**: Multi-layer validation reduces false positives
- **Confidence Thresholds**: Configurable thresholds for result reliability
- **Audit Trail**: Comprehensive logging of all analysis steps

## Future Enhancements

- **Agent Learning**: Capability to learn from manual audit feedback
- **Dynamic Agent Creation**: Runtime creation of specialized agents
- **Blockchain Integration**: Real-time analysis of deployed contracts
- **Performance Metrics**: Detailed performance and accuracy tracking