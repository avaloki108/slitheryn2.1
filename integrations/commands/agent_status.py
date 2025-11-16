#!/usr/bin/env python3
"""
Multi-Agent System Status Command

This command provides information about the multi-agent audit system status,
configuration, and available models.
"""

import argparse
import sys
import json
from pathlib import Path
import logging

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slither.ai.config import get_ai_config, setup_ai_logging
from slither.ai.ollama_client import OllamaClient

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent System Status and Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to AI configuration file'
    )
    
    parser.add_argument(
        '--check-models',
        action='store_true',
        help='Check availability of all configured models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr
    )
    
    setup_ai_logging()

def check_system_status(config_manager, check_models=False):
    """Check multi-agent system status"""
    logger = logging.getLogger("AgentStatus")
    
    status = {
        'multi_agent_system': {},
        'ai_configuration': {},
        'ollama_connection': {},
        'model_availability': {}
    }
    
    # Check AI configuration
    ai_config = config_manager.config
    status['ai_configuration'] = {
        'enable_ai_analysis': ai_config.enable_ai_analysis,
        'enable_multi_agent': getattr(ai_config, 'enable_multi_agent', True),
        'primary_model': ai_config.primary_model,
        'reasoning_model': ai_config.reasoning_model,
        'comprehensive_model': ai_config.comprehensive_model,
        'ollama_base_url': ai_config.ollama_base_url,
        'timeout': ai_config.timeout,
        'confidence_threshold': ai_config.confidence_threshold
    }
    
    # Check multi-agent configuration
    multi_agent_config = config_manager.get_multi_agent_config()
    status['multi_agent_system'] = multi_agent_config
    
    # Initialize Ollama client and check connection
    ollama_client = OllamaClient(config_manager.get_ollama_url())
    
    try:
        # Test Ollama connection
        response = ollama_client.check_model_availability("test")  # This will test connection
        status['ollama_connection'] = {
            'url': config_manager.get_ollama_url(),
            'accessible': True,
            'error': None
        }
        
        # Check model availability if requested
        if check_models:
            models_to_check = [
                ai_config.primary_model,
                ai_config.reasoning_model,
                ai_config.comprehensive_model
            ]
            
            for model in models_to_check:
                available = ollama_client.check_model_availability(model)
                status['model_availability'][model] = {
                    'available': available,
                    'model_name': model
                }
                
                if available:
                    logger.info(f"✅ Model available: {model}")
                else:
                    logger.warning(f"❌ Model not available: {model}")
        
    except Exception as e:
        status['ollama_connection'] = {
            'url': config_manager.get_ollama_url(),
            'accessible': False,
            'error': str(e)
        }
        logger.error(f"❌ Ollama connection failed: {e}")
    
    # Try to import and check multi-agent system
    try:
        from integrations.web3_audit_system import create_multi_agent_system
        
        audit_system = create_multi_agent_system(ollama_client, config_manager)
        system_status = audit_system.get_system_status()
        
        status['multi_agent_system'].update({
            'integration_available': True,
            'system_ready': audit_system.is_available(),
            'available_agents': audit_system.get_available_agents(),
            'system_status': system_status
        })
        
        logger.info("✅ Multi-agent system integration available")
        
    except ImportError as e:
        status['multi_agent_system'].update({
            'integration_available': False,
            'error': f"Integration not found: {e}",
            'system_ready': False
        })
        logger.warning("❌ Multi-agent system integration not available")
        
    except Exception as e:
        status['multi_agent_system'].update({
            'integration_available': False,
            'error': f"Integration error: {e}",
            'system_ready': False
        })
        logger.error(f"❌ Multi-agent system error: {e}")
    
    return status

def format_status_text(status):
    """Format status information as text"""
    lines = [
        "=== SLITHERYN MULTI-AGENT SYSTEM STATUS ===",
        ""
    ]
    
    # AI Configuration
    lines.extend([
        "🔧 AI CONFIGURATION:",
        f"  AI Analysis Enabled: {'✅' if status['ai_configuration']['enable_ai_analysis'] else '❌'}",
        f"  Multi-Agent Enabled: {'✅' if status['ai_configuration']['enable_multi_agent'] else '❌'}",
        f"  Primary Model: {status['ai_configuration']['primary_model']}",
        f"  Reasoning Model: {status['ai_configuration']['reasoning_model']}",
        f"  Comprehensive Model: {status['ai_configuration']['comprehensive_model']}",
        f"  Ollama URL: {status['ai_configuration']['ollama_base_url']}",
        f"  Timeout: {status['ai_configuration']['timeout']}s",
        f"  Confidence Threshold: {status['ai_configuration']['confidence_threshold']}",
        ""
    ])
    
    # Ollama Connection
    ollama = status['ollama_connection']
    lines.extend([
        "🔗 OLLAMA CONNECTION:",
        f"  URL: {ollama['url']}",
        f"  Status: {'✅ Connected' if ollama['accessible'] else '❌ Failed'}",
    ])
    
    if not ollama['accessible']:
        lines.append(f"  Error: {ollama['error']}")
    
    lines.append("")
    
    # Multi-Agent System
    ma_system = status['multi_agent_system']
    lines.extend([
        "🤖 MULTI-AGENT SYSTEM:",
        f"  Integration Available: {'✅' if ma_system.get('integration_available', False) else '❌'}",
        f"  System Ready: {'✅' if ma_system.get('system_ready', False) else '❌'}",
    ])
    
    if ma_system.get('integration_available'):
        lines.extend([
            f"  Parallel Analysis: {'✅' if ma_system.get('parallel_analysis', False) else '❌'}",
            f"  Consensus Threshold: {ma_system.get('consensus_threshold', 'N/A')}",
            f"  Max Workers: {ma_system.get('max_workers', 'N/A')}",
            f"  Available Agents: {', '.join(ma_system.get('available_agents', []))}",
        ])
    else:
        if 'error' in ma_system:
            lines.append(f"  Error: {ma_system['error']}")
    
    lines.append("")
    
    # Model Availability
    if status['model_availability']:
        lines.extend([
            "🎯 MODEL AVAILABILITY:",
        ])
        
        for model, info in status['model_availability'].items():
            status_icon = '✅' if info['available'] else '❌'
            lines.append(f"  {status_icon} {model}")
        
        lines.append("")
    
    # System Recommendations
    lines.extend([
        "💡 RECOMMENDATIONS:",
    ])
    
    recommendations = []
    
    if not status['ai_configuration']['enable_ai_analysis']:
        recommendations.append("  • Enable AI analysis in configuration")
    
    if not ollama['accessible']:
        recommendations.append("  • Check Ollama server is running and accessible")
        recommendations.append(f"  • Verify Ollama URL: {ollama['url']}")
    
    if not ma_system.get('integration_available'):
        recommendations.append("  • Install multi-agent system integration")
    
    if status['model_availability']:
        unavailable_models = [
            model for model, info in status['model_availability'].items() 
            if not info['available']
        ]
        if unavailable_models:
            recommendations.append(f"  • Install missing models: {', '.join(unavailable_models)}")
    
    if not recommendations:
        recommendations.append("  • System appears to be properly configured ✅")
    
    lines.extend(recommendations)
    
    return "\n".join(lines)

def format_status_json(status):
    """Format status information as JSON"""
    return json.dumps(status, indent=2)

def main():
    """Main entry point"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = logging.getLogger("AgentStatus")
    
    try:
        # Setup AI configuration
        if args.config:
            config_manager = get_ai_config(args.config)
        else:
            config_manager = get_ai_config()
        
        logger.info("🔍 Checking multi-agent system status...")
        
        # Check system status
        status = check_system_status(config_manager, args.check_models)
        
        # Format and output results
        if args.format == 'json':
            output = format_status_json(status)
        else:
            output = format_status_text(status)
        
        print(output)
        
        # Set exit code based on system readiness
        if status['multi_agent_system'].get('system_ready', False):
            logger.info("✅ Multi-agent system is ready")
            sys.exit(0)
        else:
            logger.warning("⚠️  Multi-agent system has issues")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()