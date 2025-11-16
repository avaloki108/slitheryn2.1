"""
Multi-Agent Audit System for Slitheryn

This module implements a specialized multi-agent system for smart contract security auditing.
Each agent has expertise in specific vulnerability types and attack patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
import concurrent.futures
import time
import logging

logger = logging.getLogger("Slitheryn.MultiAgent")

class AgentType(Enum):
    """Types of specialized audit agents"""
    VULNERABILITY_DETECTOR = "vulnerability"
    EXPLOIT_ANALYZER = "exploit"
    FIX_RECOMMENDER = "fix"
    ECONOMIC_ATTACK = "economic"
    GOVERNANCE_AUDIT = "governance"
    CONSENSUS = "consensus"

@dataclass
class AgentResult:
    """Result from an individual agent analysis"""
    agent_type: AgentType
    vulnerabilities: List[str]
    severity_scores: Dict[str, str]
    confidence_score: float
    analysis_time: float
    model_used: str
    specialized_findings: Dict[str, Any]
    raw_response: str

@dataclass
class MultiAgentAnalysisResult:
    """Aggregated result from multi-agent analysis"""
    consensus_vulnerabilities: List[str]
    agent_results: List[AgentResult]
    confidence_matrix: Dict[str, Dict[str, float]]
    final_severity_scores: Dict[str, str]
    attack_scenarios: List[str]
    fix_recommendations: List[str]
    economic_impact_assessment: Dict[str, Any]
    governance_risks: List[str]
    total_analysis_time: float
    models_used: List[str]
    consensus_score: float

class BaseAuditAgent(ABC):
    """Base class for specialized audit agents"""
    
    def __init__(self, ollama_client, agent_type: AgentType):
        self.ollama_client = ollama_client
        self.agent_type = agent_type
        self.specialized_models = self._get_specialized_models()
        
    @abstractmethod
    def _get_specialized_models(self) -> List[str]:
        """Get list of models best suited for this agent type"""
        pass
    
    @abstractmethod
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        """Build specialized prompt for this agent's expertise"""
        pass
    
    @abstractmethod
    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        """Parse agent-specific findings from response"""
        pass
    
    async def analyze(self, contract_code: str, contract_name: str = "Unknown") -> Optional[AgentResult]:
        """Perform specialized analysis on the contract"""
        model = self._select_best_model()
        if not model:
            logger.warning(f"No suitable model available for {self.agent_type.value} agent")
            return None
            
        prompt = self._build_specialized_prompt(contract_code, contract_name)
        start_time = time.time()
        
        try:
            # Use existing ollama client but with specialized prompt
            response = await self._query_model(model, prompt)
            analysis_time = time.time() - start_time
            
            if response:
                # Parse general vulnerability info
                general_findings = self.ollama_client._parse_ai_response(response)
                # Parse specialized findings
                specialized_findings = self._parse_specialized_response(response)
                
                return AgentResult(
                    agent_type=self.agent_type,
                    vulnerabilities=general_findings['vulnerabilities'],
                    severity_scores=general_findings['severity_scores'],
                    confidence_score=general_findings['confidence_score'],
                    analysis_time=analysis_time,
                    model_used=model,
                    specialized_findings=specialized_findings,
                    raw_response=response
                )
            return None
            
        except Exception as e:
            logger.error(f"Error in {self.agent_type.value} agent analysis: {e}")
            return None
    
    def _select_best_model(self) -> Optional[str]:
        """Select the best available model for this agent"""
        for model in self.specialized_models:
            if self.ollama_client.check_model_availability(model):
                return model
        return self.ollama_client.get_best_available_model()
    
    async def _query_model(self, model: str, prompt: str) -> Optional[str]:
        """Query the model with the specialized prompt"""
        import requests
        
        try:
            response = requests.post(
                f"{self.ollama_client.base_url}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'top_p': 0.9,
                        'num_predict': 2000
                    }
                },
                timeout=self.ollama_client.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            return None
            
        except Exception as e:
            logger.error(f"Error querying model {model}: {e}")
            return None

class VulnerabilityDetectorAgent(BaseAuditAgent):
    """Agent specialized in detecting specific vulnerability types"""
    
    def __init__(self, ollama_client):
        super().__init__(ollama_client, AgentType.VULNERABILITY_DETECTOR)
    
    def _get_specialized_models(self) -> List[str]:
        return ["SmartLLM-OG:latest", "phi4-reasoning:latest"]
    
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        return f"""You are a specialized vulnerability detection agent focusing on common smart contract security issues.

Contract Name: {contract_name}
Contract Code:
{contract_code}

Your specific expertise covers:
1. Reentrancy vulnerabilities (external calls before state updates)
2. Access control issues (missing modifiers, incorrect permissions)
3. Integer overflow/underflow (arithmetic operations)
4. Uninitialized storage pointers
5. Time manipulation vulnerabilities
6. tx.origin usage instead of msg.sender
7. Delegate call vulnerabilities
8. Unsafe external calls

For each vulnerability type, provide:
- Exact function and line number where it occurs
- Severity level (Critical/High/Medium/Low)
- Brief technical explanation
- Simple proof of concept if applicable

Focus only on these specific vulnerability types. Be precise and technical."""

    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        # Parse vulnerability-specific information
        response_lower = response.lower()
        
        vulnerability_locations = {}
        technical_details = {}
        
        # Extract function-specific vulnerabilities
        lines = response.split('\n')
        for line in lines:
            if 'function' in line.lower() and any(vuln in line.lower() for vuln in ['reentrancy', 'access', 'overflow', 'uninitialized']):
                # Extract function name and vulnerability type
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'function' in part.lower() and i+1 < len(parts):
                        func_name = parts[i+1]
                        vulnerability_locations[func_name] = line.strip()
                        break
        
        return {
            'vulnerability_locations': vulnerability_locations,
            'technical_details': technical_details,
            'focus_area': 'specific_vulnerabilities'
        }

class ExploitAnalyzerAgent(BaseAuditAgent):
    """Agent specialized in analyzing attack scenarios and exploits"""
    
    def __init__(self, ollama_client):
        super().__init__(ollama_client, AgentType.EXPLOIT_ANALYZER)
    
    def _get_specialized_models(self) -> List[str]:
        return ["phi4-reasoning:latest", "qwen3:30b-a3b"]
    
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        return f"""You are an expert exploit analyst specializing in constructing detailed attack scenarios for smart contracts.

Contract Name: {contract_name}
Contract Code:
{contract_code}

Your expertise focuses on:
1. Step-by-step exploit construction
2. Economic impact assessment
3. Attack vector identification
4. MEV (Maximal Extractable Value) vulnerabilities
5. Cross-contract interaction exploits
6. Flash loan attack patterns
7. Multi-transaction attack sequences

For each potential exploit:
- Provide detailed step-by-step attack scenario
- Calculate potential economic damage
- Identify required conditions for the attack
- Estimate attack complexity and likelihood
- Consider real-world feasibility

Think like an attacker. Show the complete exploit chain."""

    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        # Parse exploit-specific information
        attack_steps = []
        economic_impact = {}
        attack_complexity = {}
        
        lines = response.split('\n')
        current_scenario = []
        
        for line in lines:
            line_stripped = line.strip()
            if any(indicator in line.lower() for indicator in ['step 1', 'step 2', 'first', 'then', 'attack scenario']):
                if current_scenario:
                    attack_steps.append(' '.join(current_scenario))
                    current_scenario = []
                current_scenario.append(line_stripped)
            elif current_scenario and line_stripped:
                current_scenario.append(line_stripped)
            
            # Extract economic impact
            if any(term in line.lower() for term in ['economic', 'damage', 'loss', 'profit', 'steal']):
                economic_impact['potential_loss'] = line_stripped
        
        if current_scenario:
            attack_steps.append(' '.join(current_scenario))
        
        return {
            'attack_scenarios': attack_steps,
            'economic_impact': economic_impact,
            'attack_complexity': attack_complexity,
            'focus_area': 'exploit_analysis'
        }

class FixRecommenderAgent(BaseAuditAgent):
    """Agent specialized in providing detailed fix recommendations"""
    
    def __init__(self, ollama_client):
        super().__init__(ollama_client, AgentType.FIX_RECOMMENDER)
    
    def _get_specialized_models(self) -> List[str]:
        return ["SmartLLM-OG:latest", "qwen3:30b-a3b"]
    
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        return f"""You are a smart contract security specialist focused on providing detailed fix recommendations.

Contract Name: {contract_name}
Contract Code:
{contract_code}

Your expertise includes:
1. Code-level fixes with specific examples
2. Security pattern implementations
3. Best practice recommendations
4. Gas optimization improvements
5. Access control implementations
6. SafeMath usage and alternatives
7. Reentrancy protection patterns
8. Upgrade-safe code patterns

For each issue found, provide:
- Specific code changes with before/after examples
- Implementation of security patterns (ReentrancyGuard, etc.)
- Best practice recommendations
- Gas optimization opportunities
- Testing recommendations

Provide actionable, implementable solutions."""

    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        # Parse fix-specific information
        code_fixes = []
        best_practices = []
        gas_optimizations = []
        
        lines = response.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if any(indicator in line.lower() for indicator in ['fix', 'replace', 'use', 'implement', 'add require']):
                if len(line_stripped) > 20:  # Filter out short lines
                    code_fixes.append(line_stripped)
            elif any(indicator in line.lower() for indicator in ['best practice', 'recommend', 'should']):
                best_practices.append(line_stripped)
            elif any(indicator in line.lower() for indicator in ['gas', 'optimization', 'efficient']):
                gas_optimizations.append(line_stripped)
        
        return {
            'code_fixes': code_fixes,
            'best_practices': best_practices,
            'gas_optimizations': gas_optimizations,
            'focus_area': 'fix_recommendations'
        }

class EconomicAttackAgent(BaseAuditAgent):
    """Agent specialized in DeFi and economic vulnerabilities"""
    
    def __init__(self, ollama_client):
        super().__init__(ollama_client, AgentType.ECONOMIC_ATTACK)
    
    def _get_specialized_models(self) -> List[str]:
        return ["qwen3:30b-a3b", "phi4-reasoning:latest"]
    
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        return f"""You are a DeFi security expert specializing in economic attacks and financial vulnerabilities.

Contract Name: {contract_name}
Contract Code:
{contract_code}

Your DeFi expertise covers:
1. Flash loan attacks and arbitrage exploits
2. Price manipulation and oracle attacks
3. Liquidity pool attacks and impermanent loss exploits
4. Yield farming vulnerabilities
5. Slippage and sandwich attacks
6. Governance token attacks
7. MEV extraction opportunities
8. Cross-protocol composability risks

For each economic vulnerability:
- Analyze the economic model and tokenomics
- Identify arbitrage opportunities
- Assess oracle dependencies and manipulation risks
- Calculate potential financial impact
- Consider market conditions and liquidity requirements

Focus on financial and economic attack vectors."""

    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        # Parse economic attack specific information
        economic_vulnerabilities = []
        financial_impact = {}
        defi_risks = {}
        
        response_lower = response.lower()
        
        # Extract DeFi-specific vulnerabilities
        defi_keywords = ['flash loan', 'arbitrage', 'oracle', 'liquidity', 'yield', 'slippage', 'mev']
        for keyword in defi_keywords:
            if keyword in response_lower:
                economic_vulnerabilities.append(keyword)
        
        # Extract financial impact information
        lines = response.split('\n')
        for line in lines:
            if any(term in line.lower() for term in ['profit', 'loss', 'drain', 'extract', 'manipulate']):
                financial_impact['potential_impact'] = line.strip()
        
        return {
            'economic_vulnerabilities': economic_vulnerabilities,
            'financial_impact': financial_impact,
            'defi_risks': defi_risks,
            'focus_area': 'economic_attacks'
        }

class GovernanceAuditAgent(BaseAuditAgent):
    """Agent specialized in governance vulnerabilities"""
    
    def __init__(self, ollama_client):
        super().__init__(ollama_client, AgentType.GOVERNANCE_AUDIT)
    
    def _get_specialized_models(self) -> List[str]:
        return ["phi4-reasoning:latest", "SmartLLM-OG:latest"]
    
    def _build_specialized_prompt(self, contract_code: str, contract_name: str) -> str:
        return f"""You are a governance security specialist focusing on DAO and protocol governance vulnerabilities.

Contract Name: {contract_name}
Contract Code:
{contract_code}

Your governance expertise includes:
1. Voting manipulation and vote buying
2. Proposal attacks and malicious governance proposals
3. Admin key risks and centralization points
4. Timelock vulnerabilities
5. Quorum manipulation
6. Delegation attacks
7. Governance token distribution issues
8. Multi-sig security analysis

For each governance issue:
- Analyze voting mechanisms and potential manipulation
- Identify centralization risks
- Assess admin privileges and potential abuse
- Review timelock implementations
- Consider token distribution and voting power concentration

Focus on governance security and decentralization risks."""

    def _parse_specialized_response(self, response: str) -> Dict[str, Any]:
        # Parse governance-specific information
        governance_risks = []
        centralization_points = []
        voting_vulnerabilities = []
        
        response_lower = response.lower()
        
        # Extract governance-specific vulnerabilities
        gov_keywords = ['voting', 'proposal', 'admin', 'owner', 'governance', 'timelock', 'quorum', 'delegation']
        for keyword in gov_keywords:
            if keyword in response_lower:
                governance_risks.append(keyword)
        
        # Extract centralization points
        lines = response.split('\n')
        for line in lines:
            if any(term in line.lower() for term in ['admin', 'owner', 'centralized', 'single point']):
                centralization_points.append(line.strip())
        
        return {
            'governance_risks': governance_risks,
            'centralization_points': centralization_points,
            'voting_vulnerabilities': voting_vulnerabilities,
            'focus_area': 'governance_security'
        }