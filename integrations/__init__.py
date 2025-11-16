"""
Slitheryn Integrations Package

This package contains additional integrations and extensions for Slitheryn,
including the multi-agent audit system and command-line utilities.
"""

__version__ = "1.0.0"

# Import main integration modules for easier access
try:
    from .web3_audit_system import Web3AuditSystem, create_multi_agent_system, run_multi_agent_analysis
    __all__ = ['Web3AuditSystem', 'create_multi_agent_system', 'run_multi_agent_analysis']
except ImportError:
    # Integrations not fully available
    __all__ = []