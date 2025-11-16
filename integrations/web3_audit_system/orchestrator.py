"""
Multi-Agent Orchestrator for Slitheryn

This module coordinates multiple specialized audit agents to perform comprehensive
smart contract security analysis.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .agents import (
    BaseAuditAgent,
    VulnerabilityDetectorAgent,
    ExploitAnalyzerAgent,
    FixRecommenderAgent,
    EconomicAttackAgent,
    GovernanceAuditAgent,
    AgentType,
    AgentResult,
    MultiAgentAnalysisResult
)

logger = logging.getLogger("Slitheryn.MultiAgent.Orchestrator")

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized audit agents for comprehensive analysis"""
    
    def __init__(self, ollama_client, config: Optional[Dict] = None):
        self.ollama_client = ollama_client
        self.config = config or {}
        self.agents = self._initialize_agents()
        
        # Configuration options
        self.enable_parallel_analysis = self.config.get('parallel_analysis', True)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        self.max_workers = self.config.get('max_workers', 4)
        self.enabled_agent_types = self._parse_enabled_agents()
        
    def _initialize_agents(self) -> Dict[AgentType, BaseAuditAgent]:
        """Initialize all specialized audit agents"""
        agents = {}
        
        try:
            agents[AgentType.VULNERABILITY_DETECTOR] = VulnerabilityDetectorAgent(self.ollama_client)
            agents[AgentType.EXPLOIT_ANALYZER] = ExploitAnalyzerAgent(self.ollama_client)
            agents[AgentType.FIX_RECOMMENDER] = FixRecommenderAgent(self.ollama_client)
            agents[AgentType.ECONOMIC_ATTACK] = EconomicAttackAgent(self.ollama_client)
            agents[AgentType.GOVERNANCE_AUDIT] = GovernanceAuditAgent(self.ollama_client)
            
            logger.info(f"Initialized {len(agents)} specialized audit agents")
            return agents
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            return {}
    
    def _parse_enabled_agents(self) -> Set[AgentType]:
        """Parse enabled agent types from configuration"""
        enabled = self.config.get('agent_types', ['vulnerability', 'exploit', 'fix', 'economic', 'governance'])
        
        agent_type_map = {
            'vulnerability': AgentType.VULNERABILITY_DETECTOR,
            'exploit': AgentType.EXPLOIT_ANALYZER,
            'fix': AgentType.FIX_RECOMMENDER,
            'economic': AgentType.ECONOMIC_ATTACK,
            'governance': AgentType.GOVERNANCE_AUDIT
        }
        
        enabled_types = set()
        for agent_name in enabled:
            if agent_name in agent_type_map:
                enabled_types.add(agent_type_map[agent_name])
        
        return enabled_types or set(agent_type_map.values())  # Default to all if none specified
    
    async def analyze_contract(self, 
                             contract_code: str, 
                             contract_name: str = "Unknown") -> Optional[MultiAgentAnalysisResult]:
        """
        Perform comprehensive multi-agent analysis of a smart contract
        
        Args:
            contract_code: Solidity contract source code
            contract_name: Name of the contract
            
        Returns:
            MultiAgentAnalysisResult with aggregated findings from all agents
        """
        start_time = time.time()
        
        logger.info(f"Starting multi-agent analysis of contract: {contract_name}")
        logger.info(f"Enabled agents: {[agent.value for agent in self.enabled_agent_types]}")
        
        # Filter agents to only those enabled
        active_agents = {
            agent_type: agent for agent_type, agent in self.agents.items()
            if agent_type in self.enabled_agent_types
        }
        
        if not active_agents:
            logger.error("No active agents available for analysis")
            return None
        
        # Perform analysis with enabled agents
        if self.enable_parallel_analysis:
            agent_results = await self._parallel_analysis(active_agents, contract_code, contract_name)
        else:
            agent_results = await self._sequential_analysis(active_agents, contract_code, contract_name)
        
        if not agent_results:
            logger.error("No analysis results from any agent")
            return None
        
        # Aggregate and analyze results
        total_time = time.time() - start_time
        
        logger.info(f"Multi-agent analysis completed in {total_time:.2f} seconds")
        logger.info(f"Results from {len(agent_results)} agents")
        
        return self._aggregate_results(agent_results, total_time)
    
    async def _parallel_analysis(self, 
                                agents: Dict[AgentType, BaseAuditAgent], 
                                contract_code: str, 
                                contract_name: str) -> List[AgentResult]:
        """Perform parallel analysis using multiple agents"""
        logger.info(f"Starting parallel analysis with {len(agents)} agents")
        
        # Create tasks for async execution
        tasks = []
        for agent_type, agent in agents.items():
            task = asyncio.create_task(
                agent.analyze(contract_code, contract_name),
                name=f"{agent_type.value}_analysis"
            )
            tasks.append(task)
        
        # Wait for all agents to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    agent_type = list(agents.keys())[i]
                    logger.error(f"Agent {agent_type.value} failed with error: {result}")
                elif result is not None:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            return []
    
    async def _sequential_analysis(self, 
                                 agents: Dict[AgentType, BaseAuditAgent], 
                                 contract_code: str, 
                                 contract_name: str) -> List[AgentResult]:
        """Perform sequential analysis using multiple agents"""
        logger.info(f"Starting sequential analysis with {len(agents)} agents")
        
        results = []
        for agent_type, agent in agents.items():
            try:
                logger.info(f"Running {agent_type.value} agent analysis...")
                result = await agent.analyze(contract_code, contract_name)
                if result:
                    results.append(result)
                    logger.info(f"{agent_type.value} agent completed in {result.analysis_time:.2f}s")
                else:
                    logger.warning(f"{agent_type.value} agent returned no results")
                    
            except Exception as e:
                logger.error(f"Error in {agent_type.value} agent: {e}")
        
        return results
    
    def _aggregate_results(self, 
                          agent_results: List[AgentResult], 
                          total_time: float) -> MultiAgentAnalysisResult:
        """Aggregate results from multiple agents into a comprehensive report"""
        
        logger.info("Aggregating results from all agents...")
        
        # Collect all findings
        all_vulnerabilities = set()
        all_severity_scores = {}
        confidence_matrix = {}
        attack_scenarios = []
        fix_recommendations = []
        economic_impact = {}
        governance_risks = []
        models_used = []
        
        # Process each agent's results
        for result in agent_results:
            # Collect vulnerabilities
            all_vulnerabilities.update(result.vulnerabilities)
            
            # Collect severity scores
            all_severity_scores.update(result.severity_scores)
            
            # Build confidence matrix
            confidence_matrix[result.agent_type.value] = {
                'confidence': result.confidence_score,
                'vulnerabilities': result.vulnerabilities,
                'model': result.model_used
            }
            
            # Collect models used
            if result.model_used not in models_used:
                models_used.append(result.model_used)
            
            # Extract specialized findings
            specialized = result.specialized_findings
            
            if result.agent_type == AgentType.EXPLOIT_ANALYZER:
                attack_scenarios.extend(specialized.get('attack_scenarios', []))
                economic_impact.update(specialized.get('economic_impact', {}))
                
            elif result.agent_type == AgentType.FIX_RECOMMENDER:
                fix_recommendations.extend(specialized.get('code_fixes', []))
                fix_recommendations.extend(specialized.get('best_practices', []))
                
            elif result.agent_type == AgentType.ECONOMIC_ATTACK:
                economic_impact.update(specialized.get('financial_impact', {}))
                
            elif result.agent_type == AgentType.GOVERNANCE_AUDIT:
                governance_risks.extend(specialized.get('governance_risks', []))
        
        # Apply consensus algorithm
        consensus_vulnerabilities = self._apply_consensus(agent_results, all_vulnerabilities)
        
        # Calculate final severity scores
        final_severity_scores = self._calculate_final_severity(agent_results, consensus_vulnerabilities)
        
        # Calculate overall consensus score
        consensus_score = self._calculate_consensus_score(agent_results, consensus_vulnerabilities)
        
        logger.info(f"Consensus analysis complete:")
        logger.info(f"  - Consensus vulnerabilities: {len(consensus_vulnerabilities)}")
        logger.info(f"  - Overall consensus score: {consensus_score:.2f}")
        logger.info(f"  - Attack scenarios: {len(attack_scenarios)}")
        logger.info(f"  - Fix recommendations: {len(fix_recommendations)}")
        
        return MultiAgentAnalysisResult(
            consensus_vulnerabilities=list(consensus_vulnerabilities),
            agent_results=agent_results,
            confidence_matrix=confidence_matrix,
            final_severity_scores=final_severity_scores,
            attack_scenarios=attack_scenarios,
            fix_recommendations=fix_recommendations,
            economic_impact_assessment=economic_impact,
            governance_risks=governance_risks,
            total_analysis_time=total_time,
            models_used=models_used,
            consensus_score=consensus_score
        )
    
    def _apply_consensus(self, 
                        agent_results: List[AgentResult], 
                        all_vulnerabilities: Set[str]) -> Set[str]:
        """Apply consensus algorithm to determine final vulnerability list"""
        
        vulnerability_votes = {}
        total_agents = len(agent_results)
        
        # Count votes for each vulnerability
        for vulnerability in all_vulnerabilities:
            votes = 0
            confidence_sum = 0.0
            
            for result in agent_results:
                if vulnerability in result.vulnerabilities:
                    votes += 1
                    confidence_sum += result.confidence_score
            
            # Calculate weighted vote based on confidence
            if votes > 0:
                avg_confidence = confidence_sum / votes
                weighted_vote = (votes / total_agents) * avg_confidence
                vulnerability_votes[vulnerability] = weighted_vote
        
        # Select vulnerabilities above consensus threshold
        consensus_vulnerabilities = set()
        for vulnerability, score in vulnerability_votes.items():
            if score >= self.consensus_threshold:
                consensus_vulnerabilities.add(vulnerability)
                logger.debug(f"Consensus reached for {vulnerability}: {score:.2f}")
            else:
                logger.debug(f"Consensus not reached for {vulnerability}: {score:.2f}")
        
        return consensus_vulnerabilities
    
    def _calculate_final_severity(self, 
                                 agent_results: List[AgentResult], 
                                 consensus_vulnerabilities: Set[str]) -> Dict[str, str]:
        """Calculate final severity scores for consensus vulnerabilities"""
        
        final_scores = {}
        severity_weights = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        
        for vulnerability in consensus_vulnerabilities:
            severity_votes = {}
            total_weight = 0
            
            for result in agent_results:
                if vulnerability in result.severity_scores:
                    severity = result.severity_scores[vulnerability]
                    weight = result.confidence_score
                    
                    if severity in severity_votes:
                        severity_votes[severity] += weight
                    else:
                        severity_votes[severity] = weight
                    
                    total_weight += weight
            
            # Select severity with highest weighted vote
            if severity_votes:
                best_severity = max(severity_votes.items(), key=lambda x: x[1])[0]
                final_scores[vulnerability] = best_severity
        
        return final_scores
    
    def _calculate_consensus_score(self, 
                                 agent_results: List[AgentResult], 
                                 consensus_vulnerabilities: Set[str]) -> float:
        """Calculate overall consensus score for the analysis"""
        
        if not agent_results:
            return 0.0
        
        # Base score from agent confidence
        avg_confidence = sum(result.confidence_score for result in agent_results) / len(agent_results)
        
        # Agreement score based on vulnerability consensus
        if len(consensus_vulnerabilities) > 0:
            total_vulnerabilities = set()
            for result in agent_results:
                total_vulnerabilities.update(result.vulnerabilities)
            
            if total_vulnerabilities:
                agreement_ratio = len(consensus_vulnerabilities) / len(total_vulnerabilities)
            else:
                agreement_ratio = 1.0
        else:
            agreement_ratio = 1.0 if not any(result.vulnerabilities for result in agent_results) else 0.0
        
        # Model diversity bonus
        unique_models = len(set(result.model_used for result in agent_results))
        diversity_bonus = min(0.1, unique_models * 0.05)
        
        # Final consensus score
        consensus_score = min(1.0, avg_confidence * 0.6 + agreement_ratio * 0.3 + diversity_bonus)
        
        return consensus_score
    
    def get_analysis_summary(self, result: MultiAgentAnalysisResult) -> str:
        """Generate a human-readable summary of the multi-agent analysis"""
        
        summary_lines = [
            "=== SLITHERYN MULTI-AGENT AUDIT REPORT ===",
            f"Analysis Time: {result.total_analysis_time:.2f} seconds",
            f"Consensus Score: {result.consensus_score:.2f}",
            f"Models Used: {', '.join(result.models_used)}",
            f"Agents Deployed: {len(result.agent_results)}",
            "",
            "=== CONSENSUS VULNERABILITIES ===",
        ]
        
        if result.consensus_vulnerabilities:
            for vuln in result.consensus_vulnerabilities:
                severity = result.final_severity_scores.get(vuln, 'Unknown')
                summary_lines.append(f"[{severity}] {vuln}")
        else:
            summary_lines.append("No consensus vulnerabilities found.")
        
        if result.attack_scenarios:
            summary_lines.extend([
                "",
                "=== ATTACK SCENARIOS ===",
            ])
            for i, scenario in enumerate(result.attack_scenarios[:3], 1):  # Show top 3
                summary_lines.append(f"{i}. {scenario}")
        
        if result.fix_recommendations:
            summary_lines.extend([
                "",
                "=== FIX RECOMMENDATIONS ===",
            ])
            for i, fix in enumerate(result.fix_recommendations[:5], 1):  # Show top 5
                summary_lines.append(f"{i}. {fix}")
        
        if result.governance_risks:
            summary_lines.extend([
                "",
                "=== GOVERNANCE RISKS ===",
            ])
            for risk in result.governance_risks:
                summary_lines.append(f"- {risk}")
        
        return "\n".join(summary_lines)