#!/usr/bin/env python3
"""
Multi-Agent Smart Contract Audit Command

This command provides a standalone interface for running multi-agent security audits
on smart contracts using Slitheryn's enhanced AI capabilities.
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
import logging

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slither.ai.config import get_ai_config, setup_ai_logging
from slither.ai.ollama_client import OllamaClient

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Smart Contract Security Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s contract.sol                          # Full multi-agent audit
  %(prog)s contract.sol --quick                  # Quick vulnerability scan
  %(prog)s contract.sol --agents economic,governance  # Specialized analysis
  %(prog)s contract.sol --output report.json    # JSON output
  %(prog)s contract.sol --parallel false        # Sequential analysis
        """
    )
    
    parser.add_argument(
        'contract_file',
        help='Solidity contract file to audit'
    )
    
    parser.add_argument(
        '--agents',
        default='vulnerability,exploit,fix,economic,governance',
        help='Comma-separated list of agents to use (default: all)'
    )
    
    parser.add_argument(
        '--analysis-type',
        choices=['quick', 'comprehensive', 'specialized'],
        default='comprehensive',
        help='Type of analysis to perform (default: comprehensive)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Shortcut for --analysis-type quick'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for results (stdout if not specified)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--parallel',
        type=bool,
        default=True,
        help='Enable parallel agent execution (default: true)'
    )
    
    parser.add_argument(
        '--consensus-threshold',
        type=float,
        default=0.7,
        help='Consensus threshold for vulnerability agreement (default: 0.7)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout for each agent in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        help='Path to AI configuration file'
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr
    )
    
    setup_ai_logging()

def read_contract_file(file_path: str) -> tuple[str, str]:
    """Read contract file and return code and name"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            contract_code = f.read()
        
        contract_name = Path(file_path).stem
        return contract_code, contract_name
        
    except FileNotFoundError:
        print(f"Error: Contract file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading contract file: {e}", file=sys.stderr)
        sys.exit(1)

async def run_multi_agent_audit(args):
    """Run the multi-agent audit"""
    logger = logging.getLogger("MultiAgentAudit")
    
    # Read contract file
    contract_code, contract_name = read_contract_file(args.contract_file)
    logger.info(f"Loaded contract: {contract_name}")
    
    # Setup AI configuration
    if args.config:
        # Load custom config if specified
        config_manager = get_ai_config(args.config)
    else:
        config_manager = get_ai_config()
    
    # Override config with command line arguments
    config_updates = {}
    if hasattr(args, 'consensus_threshold'):
        config_updates['consensus_threshold'] = args.consensus_threshold
    if hasattr(args, 'parallel'):
        config_updates['parallel_analysis'] = args.parallel
    if hasattr(args, 'timeout'):
        config_updates['timeout'] = args.timeout
    
    # Parse agent types
    agent_types = [agent.strip() for agent in args.agents.split(',')]
    config_updates['agent_types'] = agent_types
    
    if config_updates:
        config_manager.update_config(**config_updates)
    
    # Initialize Ollama client
    ollama_client = OllamaClient(config_manager.get_ollama_url())
    
    # Check if multi-agent analysis is available
    analysis_type = 'quick' if args.quick else args.analysis_type
    
    try:
        logger.info("🚀 Starting multi-agent security audit...")
        logger.info(f"Contract: {contract_name}")
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Agents: {', '.join(agent_types)}")
        
        # Attempt multi-agent analysis
        result = await ollama_client.analyze_contract_multi_agent(
            contract_code, 
            contract_name, 
            analysis_type
        )
        
        if result and result.get('multi_agent'):
            logger.info("✅ Multi-agent analysis completed successfully")
            
            # Format and output results
            output_text = format_results(result, args.format)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                logger.info(f"Results saved to: {args.output}")
            else:
                print(output_text)
        
        else:
            logger.warning("Multi-agent analysis not available, falling back to single-agent")
            
            # Fallback to single-agent analysis
            fallback_result = ollama_client.analyze_contract(
                contract_code,
                contract_name,
                analysis_type
            )
            
            if fallback_result:
                logger.info("✅ Single-agent analysis completed")
                output_text = format_single_agent_results(fallback_result, args.format)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output_text)
                    logger.info(f"Results saved to: {args.output}")
                else:
                    print(output_text)
            else:
                logger.error("❌ Analysis failed")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"❌ Analysis failed with error: {e}")
        sys.exit(1)

def format_results(result: dict, format_type: str) -> str:
    """Format multi-agent analysis results"""
    if format_type == 'json':
        return json.dumps(result, indent=2)
    
    elif format_type == 'markdown':
        return format_markdown_report(result)
    
    else:  # text format
        return result.get('full_report', 'No report available')

def format_single_agent_results(result, format_type: str) -> str:
    """Format single-agent analysis results"""
    if format_type == 'json':
        return json.dumps({
            'multi_agent': False,
            'vulnerabilities': result.vulnerabilities,
            'severity_scores': result.severity_scores,
            'attack_scenarios': result.attack_scenarios,
            'fix_recommendations': result.fix_recommendations,
            'confidence_score': result.confidence_score,
            'analysis_time': result.analysis_time,
            'model_used': result.model_used
        }, indent=2)
    
    elif format_type == 'markdown':
        return format_single_agent_markdown(result)
    
    else:  # text format
        return format_single_agent_text(result)

def format_markdown_report(result: dict) -> str:
    """Format multi-agent results as markdown"""
    lines = [
        "# Multi-Agent Smart Contract Security Audit Report",
        "",
        f"**Analysis Time:** {result.get('analysis_time', 0):.2f} seconds",
        f"**Consensus Score:** {result.get('consensus_score', 0):.2f}",
        f"**Models Used:** {', '.join(result.get('models_used', []))}",
        "",
        "## Consensus Vulnerabilities",
        ""
    ]
    
    vulnerabilities = result.get('consensus_vulnerabilities', [])
    severity_scores = result.get('final_severity_scores', {})
    
    if vulnerabilities:
        for vuln in vulnerabilities:
            severity = severity_scores.get(vuln, 'Unknown')
            lines.append(f"- **[{severity}]** {vuln}")
    else:
        lines.append("No consensus vulnerabilities found.")
    
    lines.extend([
        "",
        "## Attack Scenarios",
        ""
    ])
    
    scenarios = result.get('attack_scenarios', [])
    if scenarios:
        for i, scenario in enumerate(scenarios[:3], 1):
            lines.append(f"### Scenario {i}")
            lines.append(scenario)
            lines.append("")
    else:
        lines.append("No attack scenarios identified.")
    
    lines.extend([
        "",
        "## Fix Recommendations",
        ""
    ])
    
    fixes = result.get('fix_recommendations', [])
    if fixes:
        for i, fix in enumerate(fixes[:5], 1):
            lines.append(f"{i}. {fix}")
    else:
        lines.append("No specific fix recommendations.")
    
    return "\n".join(lines)

def format_single_agent_markdown(result) -> str:
    """Format single-agent results as markdown"""
    lines = [
        "# Smart Contract Security Audit Report",
        "",
        f"**Analysis Time:** {result.analysis_time:.2f} seconds",
        f"**Confidence Score:** {result.confidence_score:.2f}",
        f"**Model Used:** {result.model_used}",
        "",
        "## Vulnerabilities Found",
        ""
    ]
    
    if result.vulnerabilities:
        for vuln in result.vulnerabilities:
            severity = result.severity_scores.get(vuln, 'Unknown')
            lines.append(f"- **[{severity}]** {vuln}")
    else:
        lines.append("No vulnerabilities found.")
    
    return "\n".join(lines)

def format_single_agent_text(result) -> str:
    """Format single-agent results as text"""
    lines = [
        "=== SLITHERYN SECURITY AUDIT REPORT ===",
        f"Analysis Time: {result.analysis_time:.2f} seconds",
        f"Confidence Score: {result.confidence_score:.2f}",
        f"Model Used: {result.model_used}",
        "",
        "=== VULNERABILITIES ===",
    ]
    
    if result.vulnerabilities:
        for vuln in result.vulnerabilities:
            severity = result.severity_scores.get(vuln, 'Unknown')
            lines.append(f"[{severity}] {vuln}")
    else:
        lines.append("No vulnerabilities found.")
    
    if result.attack_scenarios:
        lines.extend([
            "",
            "=== ATTACK SCENARIOS ===",
        ])
        for i, scenario in enumerate(result.attack_scenarios[:3], 1):
            lines.append(f"{i}. {scenario}")
    
    if result.fix_recommendations:
        lines.extend([
            "",
            "=== FIX RECOMMENDATIONS ===",
        ])
        for i, fix in enumerate(result.fix_recommendations[:5], 1):
            lines.append(f"{i}. {fix}")
    
    return "\n".join(lines)

async def main():
    """Main entry point"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        await run_multi_agent_audit(args)
    except KeyboardInterrupt:
        print("\n🛑 Audit interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())