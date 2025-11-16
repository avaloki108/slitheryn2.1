# Multi-Agent Audit Suite Integration

This directory contains the multi-agent audit system integration that enhances Slitheryn's AI-powered security analysis capabilities.

## Overview

The multi-agent audit suite provides specialized AI agents that work together to perform comprehensive smart contract security analysis. Each agent specializes in different aspects of security auditing, allowing for more thorough and accurate vulnerability detection.

## Agent Types

### Core Analysis Agents

1. **VulnerabilityDetectorAgent**: Specialized in detecting specific vulnerability types
   - Reentrancy attacks
   - Access control issues
   - Integer overflow/underflow
   - Uninitialized storage
   - Time manipulation

2. **ExploitAnalyzerAgent**: Focuses on attack scenario analysis
   - Step-by-step exploit construction
   - Economic impact assessment
   - Attack vector identification
   - MEV vulnerability analysis

3. **FixRecommenderAgent**: Provides detailed fix recommendations
   - Code-level fixes with examples
   - Best practice recommendations
   - Security pattern suggestions
   - Gas optimization improvements

4. **EconomicAttackAgent**: Analyzes DeFi and economic vulnerabilities
   - Flash loan attacks
   - Price manipulation
   - Liquidity attacks
   - Yield farming exploits
   - Oracle manipulation

5. **GovernanceAuditAgent**: Focuses on governance vulnerabilities
   - Voting manipulation
   - Proposal attacks
   - Admin key risks
   - Decentralization analysis

6. **ConsensusAgent**: Coordinates results from other agents
   - Result aggregation
   - Confidence scoring
   - False positive reduction
   - Final report generation

## Architecture

The multi-agent system is built on top of Slitheryn's existing AI infrastructure, extending the Ollama client and AI configuration system to support:

- **Agent Specialization**: Each agent uses specialized prompts and models
- **Parallel Analysis**: Multiple agents can analyze different aspects simultaneously
- **Result Coordination**: Consensus-based result aggregation
- **Model Selection**: Dynamic model selection based on agent requirements
- **Load Balancing**: Distribute workload across available models

## Usage

The multi-agent system can be enabled through the AI configuration:

```json
{
  "enable_multi_agent": true,
  "agent_types": ["vulnerability", "exploit", "fix", "economic", "governance"],
  "consensus_threshold": 0.7,
  "parallel_analysis": true
}
```

Use with Slitheryn:

```bash
slitheryn contract.sol --multi-agent --agent-types vulnerability,exploit,economic
```

## Integration Notes

- Fully compatible with existing Slitheryn AI system
- Maintains backward compatibility with single-agent analysis
- Extends current AI configuration system
- Uses existing Ollama integration
- Preserves all current detector and printer functionality
# Agents Integration

This file contains information about the agents integration from the scaling-octo-garbanzo repository.

## Overview

This integration brings additional functionality from the scaling-octo-garbanzo project into Slitheryn.

## Structure

- `web3-audit-system/` - Contains the web3 audit system components
- `commands/` - Contains command-line tools and utilities
- `AGENTS.md` - This documentation file

## Usage

[To be documented based on the actual content from scaling-octo-garbanzo repository]

## Notes

This is a placeholder structure created for the integration. The actual content would be copied from the `avaloki108/scaling-octo-garbanzo` repository once it becomes accessible.
