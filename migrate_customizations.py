#!/usr/bin/env python3
"""
Script to migrate Slytheryn customizations to slitheryn2.1
"""

import os
import shutil
import re
from pathlib import Path

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_ai_components():
    """Copy AI-related components"""
    print("📦 Copying AI components...")
    
    # Copy AI module
    src_ai = "Slitheryn/slither/ai"
    dst_ai = "slitheryn2.1/slither/ai"
    if os.path.exists(src_ai):
        ensure_directory(dst_ai)
        shutil.copytree(src_ai, dst_ai, dirs_exist_ok=True)
        print(f"  ✅ Copied AI module to {dst_ai}")
    
    # Copy AI detectors
    src_ai_detector = "Slitheryn/slither/detectors/ai"
    dst_ai_detector = "slitheryn2.1/slither/detectors/ai"
    if os.path.exists(src_ai_detector):
        ensure_directory(dst_ai_detector)
        shutil.copytree(src_ai_detector, dst_ai_detector, dirs_exist_ok=True)
        print(f"  ✅ Copied AI detectors to {dst_ai_detector}")
    
    # Copy integrations folder
    src_integrations = "Slitheryn/integrations"
    dst_integrations = "slitheryn2.1/integrations"
    if os.path.exists(src_integrations):
        ensure_directory(dst_integrations)
        shutil.copytree(src_integrations, dst_integrations, dirs_exist_ok=True)
        print(f"  ✅ Copied integrations to {dst_integrations}")

def update_main_file():
    """Update __main__.py with AI enhancements"""
    print("🔧 Updating __main__.py...")
    
    main_file = "slitheryn2.1/slither/__main__.py"
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Add multi-agent analysis imports and functions
    ai_imports = """
async def _process_with_multi_agent(
    slither: Slither,
    detector_classes: List[Type[AbstractDetector]],
    printer_classes: List[Type[AbstractPrinter]],
    multi_agent_args: dict,
) -> Tuple[Slither, List[Dict], List[Output], int, Optional[Dict]]:
    \"\"\"
    Enhanced process function that supports multi-agent AI analysis
    \"\"\"
    # Run standard analysis first
    slither_result, results_detectors, results_printers, analyzed_contracts_count = _process(
        slither, detector_classes, printer_classes
    )
    
    multi_agent_result = None
    
    # Run multi-agent analysis if enabled
    if multi_agent_args.get('enable_multi_agent', False):
        try:
            from slither.ai.config import get_ai_config
            from slither.ai.ollama_client import OllamaClient
            
            logger.info("🤖 Starting multi-agent AI analysis...")
            
            ai_config = get_ai_config()
            
            # Update config with CLI arguments
            config_updates = {}
            if 'agent_types' in multi_agent_args:
                agent_types = [t.strip() for t in multi_agent_args['agent_types'].split(',')]
                config_updates['agent_types'] = agent_types
            if 'consensus_threshold' in multi_agent_args:
                config_updates['consensus_threshold'] = multi_agent_args['consensus_threshold']
            if 'parallel_analysis' in multi_agent_args:
                config_updates['parallel_analysis'] = multi_agent_args['parallel_analysis']
            
            if config_updates:
                ai_config.update_config(**config_updates)
            
            # Initialize Ollama client
            ollama_client = OllamaClient(ai_config.get_ollama_url())
            
            # Prepare contract code for analysis
            contracts_analyzed = []
            
            for compilation_unit in slither.compilation_units:
                for contract in compilation_unit.contracts:
                    if contract.is_interface or not contract.source_mapping:
                        continue
                    
                    try:
                        # Get contract source code
                        with open(str(contract.source_mapping.filename.absolute), 'r') as f:
                            contract_code = f.read()
                        
                        # Run multi-agent analysis
                        analysis_type = multi_agent_args.get('analysis_type', 'comprehensive')
                        
                        result = await ollama_client.analyze_contract_multi_agent(
                            contract_code,
                            contract.name,
                            analysis_type
                        )
                        
                        if result and result.get('multi_agent'):
                            contracts_analyzed.append({
                                'contract_name': contract.name,
                                'file_path': str(contract.source_mapping.filename.absolute),
                                'multi_agent_result': result
                            })
                            
                            logger.info(f"✅ Multi-agent analysis completed for {contract.name}")
                            logger.info(f"   Found {len(result['consensus_vulnerabilities'])} consensus vulnerabilities")
                            
                    except Exception as e:
                        logger.warning(f"Multi-agent analysis failed for {contract.name}: {e}")
                        continue
            
            if contracts_analyzed:
                multi_agent_result = {
                    'total_contracts_analyzed': len(contracts_analyzed),
                    'contracts': contracts_analyzed,
                    'analysis_type': analysis_type,
                    'enabled_agents': multi_agent_args.get('agent_types', '').split(',')
                }
                
                logger.info(f"🎯 Multi-agent analysis summary:")
                logger.info(f"   Contracts analyzed: {len(contracts_analyzed)}")
                
                # Add summary to results
                total_consensus_vulns = sum(
                    len(c['multi_agent_result']['consensus_vulnerabilities']) 
                    for c in contracts_analyzed
                )
                logger.info(f"   Total consensus vulnerabilities: {total_consensus_vulns}")
            
        except ImportError:
            logger.warning("Multi-agent system not available (integration not found)")
        except Exception as e:
            logger.error(f"Multi-agent analysis error: {e}")
    
    return slither_result, results_detectors, results_printers, analyzed_contracts_count, multi_agent_result
"""
    
    # Find the right place to insert the function (after _process function)
    if "_process_with_multi_agent" not in content:
        # Find where to insert
        insert_pos = content.find("def process_from_asts(")
        if insert_pos > 0:
            content = content[:insert_pos] + ai_imports + "\n\n" + content[insert_pos:]
    
    # Update process_all to use multi-agent when needed
    process_replacement = """
    # Check if multi-agent analysis is requested
    if getattr(args, 'multi_agent', False):
        # Prepare multi-agent arguments
        multi_agent_args = {
            'enable_multi_agent': True,
            'agent_types': getattr(args, 'agent_types', 'vulnerability,exploit,fix,economic,governance'),
            'analysis_type': getattr(args, 'analysis_type', 'comprehensive'),
            'consensus_threshold': getattr(args, 'consensus_threshold', 0.7),
            'parallel_analysis': not getattr(args, 'no_parallel_agents', False)
        }
        
        # Run async multi-agent analysis
        import asyncio
        
        async def run_multi_agent():
            return await _process_with_multi_agent(
                slither, detector_classes, printer_classes, multi_agent_args
            )
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_multi_agent())
            loop.close()
            
            # Extract results (ignore multi_agent_result for now in standard return)
            slither_result, results_detectors, results_printers, analyzed_contracts_count, multi_agent_result = result
            
            # Store multi-agent results in slither object for later use
            if multi_agent_result:
                slither._multi_agent_results = multi_agent_result
            
            return slither_result, results_detectors, results_printers, analyzed_contracts_count
            
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}")
            logger.info("Falling back to standard analysis...")
            return _process(slither, detector_classes, printer_classes)
    else:
        return _process(slither, detector_classes, printer_classes)"""
    
    # Replace the simple _process call
    simple_process = "return _process(slither, detector_classes, printer_classes)"
    if simple_process in content:
        content = content.replace(simple_process, process_replacement, 1)
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("  ✅ Updated __main__.py with AI enhancements")

def add_cli_arguments():
    """Add multi-agent CLI arguments"""
    print("🔧 Adding CLI arguments...")
    
    main_file = "slitheryn2.1/slither/__main__.py"
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    cli_args = '''
    # Multi-Agent AI Analysis options
    group_multi_agent = parser.add_argument_group("Multi-Agent AI Analysis")
    group_multi_agent.add_argument(
        "--multi-agent",
        help="Enable multi-agent AI analysis (requires AI system)",
        action="store_true",
        default=defaults_flag_in_config.get("multi_agent", False),
    )
    
    group_multi_agent.add_argument(
        "--agent-types",
        help="Comma-separated list of agent types to use: vulnerability,exploit,fix,economic,governance",
        action="store",
        default=defaults_flag_in_config.get("agent_types", "vulnerability,exploit,fix,economic,governance"),
    )
    
    group_multi_agent.add_argument(
        "--analysis-type",
        help="Type of multi-agent analysis: quick, comprehensive, specialized",
        choices=["quick", "comprehensive", "specialized"],
        default=defaults_flag_in_config.get("analysis_type", "comprehensive"),
    )
    
    group_multi_agent.add_argument(
        "--consensus-threshold",
        help="Consensus threshold for vulnerability agreement (0.0-1.0)",
        type=float,
        default=defaults_flag_in_config.get("consensus_threshold", 0.7),
    )
    
    group_multi_agent.add_argument(
        "--no-parallel-agents",
        help="Disable parallel agent execution",
        action="store_true",
        default=False,
    )
'''
    
    # Find where to insert CLI args (after other argument groups)
    marker = "group_misc = parser.add_argument_group"
    if marker in content and "Multi-Agent AI Analysis" not in content:
        # Find the end of the misc group
        pos = content.find(marker)
        # Find the next group or return statement
        next_group = content.find("return parser", pos)
        if next_group > 0:
            content = content[:next_group] + cli_args + "\n" + content[next_group:]
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("  ✅ Added CLI arguments")

def update_detector_imports():
    """Update all_detectors.py to include AI detector"""
    print("🔧 Updating detector imports...")
    
    detector_file = "slitheryn2.1/slither/detectors/all_detectors.py"
    
    with open(detector_file, 'r') as f:
        content = f.read()
    
    # Add AI detector import at the end
    if "ai_enhanced_analysis" not in content:
        content += "\n# AI-enhanced detectors\nfrom .ai.ai_enhanced_analysis import AIEnhancedAnalysis\n"
    
    with open(detector_file, 'w') as f:
        f.write(content)
    
    print("  ✅ Updated detector imports")

def update_branding(keep_slither=False):
    """Update branding from Slither to Slitheryn"""
    if keep_slither:
        print("🎨 Keeping Slither branding...")
        return
    
    print("🎨 Updating branding to Slitheryn...")
    
    files_to_update = [
        "slitheryn2.1/README.md",
        "slitheryn2.1/setup.py"
    ]
    
    replacements = {
        "Slither": "Slitheryn",
        "slither": "slitheryn",
        "crytic/slither": "avaloki108/slitheryn2.1",
        "slither-analyzer": "slitheryn-analyzer"
    }
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"  ✅ Updated {file_path}")

def copy_test_files():
    """Copy test contracts"""
    print("📄 Copying test files...")
    
    test_files = [
        "test_contract.sol",
        "test_vulnerable_contract.sol",
        "test_integration.py"
    ]
    
    for file in test_files:
        src = f"Slitheryn/{file}"
        dst = f"slitheryn2.1/{file}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {file}")

def main():
    print("🚀 Starting Slitheryn customization migration to slitheryn2.1")
    print("="*60)
    
    # Check if both directories exist
    if not os.path.exists("Slitheryn"):
        print("❌ Error: Slitheryn directory not found")
        return
    
    if not os.path.exists("slitheryn2.1"):
        print("❌ Error: slitheryn2.1 directory not found")
        return
    
    try:
        # Step 1: Copy AI components
        copy_ai_components()
        
        # Step 2: Update main file
        update_main_file()
        
        # Step 3: Add CLI arguments
        add_cli_arguments()
        
        # Step 4: Update detector imports
        update_detector_imports()
        
        # Step 5: Update branding (optional)
        response = input("\n📝 Do you want to keep Slitheryn branding? (y/n): ")
        update_branding(keep_slither=(response.lower() != 'y'))
        
        # Step 6: Copy test files
        copy_test_files()
        
        print("\n" + "="*60)
        print("✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Review the changes in slitheryn2.1")
        print("2. Test the integration: cd slitheryn2.1 && python -m pytest")
        print("3. Test AI features: python -m slither . --multi-agent")
        print("4. Commit the changes: git add . && git commit -m 'Merge Slitheryn customizations'")
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        print("Please review the error and try again.")

if __name__ == "__main__":
    main()