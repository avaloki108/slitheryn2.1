"""
Web3 Multi-Agent Audit System Integration for Slitheryn

This module provides the main integration point for the multi-agent audit system,
extending Slitheryn's existing AI capabilities with specialized agent coordination.
"""

__version__ = "1.0.0"

import logging
import asyncio
from typing import Optional, Dict, Any

from .orchestrator import MultiAgentOrchestrator
from .agents import AgentType, MultiAgentAnalysisResult

logger = logging.getLogger("Slitheryn.MultiAgent")

class Web3AuditSystem:
    """
    Main class for Web3 multi-agent audit system functionality.
    
    Integrates with Slitheryn's existing AI infrastructure to provide
    specialized multi-agent security analysis capabilities.
    """
    
    def __init__(self, ollama_client=None, config: Optional[Dict] = None):
        """
        Initialize the Web3 audit system.
        
        Args:
            ollama_client: Existing Ollama client from Slitheryn's AI system
            config: Configuration dictionary for multi-agent system
        """
        self.ollama_client = ollama_client
        self.config = config or {}
        self.orchestrator = None
        
        # Multi-agent configuration
        self.enable_multi_agent = self.config.get('enable_multi_agent', True)
        self.agent_types = self.config.get('agent_types', [
            'vulnerability', 'exploit', 'fix', 'economic', 'governance'
        ])
        
        if self.enable_multi_agent and self.ollama_client:
            self._initialize_orchestrator()
    
    def _initialize_orchestrator(self):
        """Initialize the multi-agent orchestrator"""
        try:
            self.orchestrator = MultiAgentOrchestrator(self.ollama_client, self.config)
            logger.info("Multi-agent orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize multi-agent orchestrator: {e}")
            self.orchestrator = None
    
    async def audit(self, 
                   contract_code: str, 
                   contract_name: str = "Unknown",
                   analysis_type: str = "comprehensive") -> Optional[MultiAgentAnalysisResult]:
        """
        Perform multi-agent audit of a smart contract.
        
        Args:
            contract_code: Solidity contract source code
            contract_name: Name of the contract being audited
            analysis_type: Type of analysis ('quick', 'comprehensive', 'specialized')
            
        Returns:
            MultiAgentAnalysisResult with comprehensive findings from all agents
        """
        if not self.enable_multi_agent:
            logger.warning("Multi-agent analysis is disabled")
            return None
        
        if not self.orchestrator:
            logger.error("Multi-agent orchestrator not available")
            return None
        
        if not self.ollama_client:
            logger.error("Ollama client not available for multi-agent analysis")
            return None
        
        try:
            logger.info(f"Starting multi-agent audit of {contract_name}")
            logger.info(f"Analysis type: {analysis_type}")
            
            # Configure orchestrator based on analysis type
            if analysis_type == "quick":
                # For quick analysis, use fewer agents
                original_agents = self.orchestrator.enabled_agent_types.copy()
                self.orchestrator.enabled_agent_types = {
                    AgentType.VULNERABILITY_DETECTOR,
                    AgentType.EXPLOIT_ANALYZER
                }
            elif analysis_type == "specialized":
                # For specialized analysis, use economic and governance agents
                original_agents = self.orchestrator.enabled_agent_types.copy()
                self.orchestrator.enabled_agent_types = {
                    AgentType.ECONOMIC_ATTACK,
                    AgentType.GOVERNANCE_AUDIT,
                    AgentType.FIX_RECOMMENDER
                }
            else:
                original_agents = None  # Use all agents for comprehensive analysis
            
            # Perform the analysis
            result = await self.orchestrator.analyze_contract(contract_code, contract_name)
            
            # Restore original agent configuration
            if original_agents is not None:
                self.orchestrator.enabled_agent_types = original_agents
            
            if result:
                logger.info(f"Multi-agent audit completed successfully")
                logger.info(f"Found {len(result.consensus_vulnerabilities)} consensus vulnerabilities")
                logger.info(f"Consensus score: {result.consensus_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during multi-agent audit: {e}")
            return None
    
    def generate_report(self, audit_results: MultiAgentAnalysisResult) -> str:
        """
        Generate a comprehensive audit report from multi-agent results.
        
        Args:
            audit_results: Results from multi-agent analysis
            
        Returns:
            Formatted audit report string
        """
        if not audit_results:
            return "No audit results available."
        
        if not self.orchestrator:
            return "Multi-agent orchestrator not available for report generation."
        
        try:
            return self.orchestrator.get_analysis_summary(audit_results)
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return f"Error generating report: {e}"
    
    def is_available(self) -> bool:
        """Check if the multi-agent audit system is available and configured"""
        return (
            self.enable_multi_agent and 
            self.ollama_client is not None and 
            self.orchestrator is not None
        )
    
    def get_available_agents(self) -> list:
        """Get list of available agent types"""
        if not self.orchestrator:
            return []
        
        return [agent_type.value for agent_type in self.orchestrator.enabled_agent_types]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the multi-agent system"""
        status = {
            'multi_agent_enabled': self.enable_multi_agent,
            'ollama_client_available': self.ollama_client is not None,
            'orchestrator_available': self.orchestrator is not None,
            'available_agents': self.get_available_agents(),
            'system_ready': self.is_available()
        }
        
        if self.orchestrator:
            status.update({
                'parallel_analysis': self.orchestrator.enable_parallel_analysis,
                'consensus_threshold': self.orchestrator.consensus_threshold,
                'max_workers': self.orchestrator.max_workers
            })
        
        return status
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration"""
        self.config.update(new_config)
        
        # Update enable flag
        self.enable_multi_agent = self.config.get('enable_multi_agent', True)
        
        # Re-initialize orchestrator if needed
        if self.enable_multi_agent and self.ollama_client and not self.orchestrator:
            self._initialize_orchestrator()
        elif self.orchestrator:
            # Update orchestrator config
            self.orchestrator.config.update(new_config)
            self.orchestrator.enabled_agent_types = self.orchestrator._parse_enabled_agents()
        
        logger.info("Multi-agent system configuration updated")

# Compatibility functions for integration with existing Slitheryn AI system
def create_multi_agent_system(ollama_client, ai_config_manager) -> Web3AuditSystem:
    """
    Create a multi-agent system using existing Slitheryn AI infrastructure
    
    Args:
        ollama_client: Existing Ollama client from Slitheryn
        ai_config_manager: AI configuration manager from Slitheryn
        
    Returns:
        Initialized Web3AuditSystem
    """
    # Extract multi-agent configuration from AI config
    ai_config = ai_config_manager.config
    
    multi_agent_config = {
        'enable_multi_agent': getattr(ai_config, 'enable_multi_agent', True),
        'agent_types': getattr(ai_config, 'agent_types', [
            'vulnerability', 'exploit', 'fix', 'economic', 'governance'
        ]),
        'consensus_threshold': getattr(ai_config, 'consensus_threshold', 0.7),
        'parallel_analysis': getattr(ai_config, 'parallel_analysis', True),
        'max_workers': getattr(ai_config, 'max_workers', 4)
    }
    
    return Web3AuditSystem(ollama_client, multi_agent_config)

async def run_multi_agent_analysis(contract_code: str, 
                                 contract_name: str,
                                 ollama_client,
                                 ai_config_manager) -> Optional[MultiAgentAnalysisResult]:
    """
    Convenience function to run multi-agent analysis with existing Slitheryn AI infrastructure
    
    Args:
        contract_code: Solidity contract source code
        contract_name: Name of the contract
        ollama_client: Existing Ollama client
        ai_config_manager: AI configuration manager
        
    Returns:
        Multi-agent analysis results
    """
    audit_system = create_multi_agent_system(ollama_client, ai_config_manager)
    
    if not audit_system.is_available():
        logger.warning("Multi-agent audit system not available")
        return None
    
    return await audit_system.audit(contract_code, contract_name)

# Export main classes and functions
__all__ = [
    'Web3AuditSystem',
    'MultiAgentAnalysisResult',
    'AgentType',
    'create_multi_agent_system',
    'run_multi_agent_analysis'
]