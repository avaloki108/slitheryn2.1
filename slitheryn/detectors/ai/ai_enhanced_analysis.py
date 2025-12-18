"""
AI-Enhanced Smart Contract Analysis Detector
Integrates SmartLLM-OG with traditional static analysis
"""

from typing import List, Dict, Any
import logging

from slitheryn.detectors.abstract_detector import AbstractDetector, DetectorClassification
from slitheryn.utils.output import Output
from slitheryn.ai.ollama_client import OllamaClient, AIAnalysisResult
from slitheryn.ai.config import get_ai_config

logger = logging.getLogger("Slitheryn.AIDetector")

class AIEnhancedAnalysis(AbstractDetector):
    """
    AI-Enhanced vulnerability detection using SmartLLM-OG
    Combines traditional static analysis with AI-powered security analysis
    """

    ARGUMENT = 'ai-analysis'
    HELP = 'AI-powered smart contract security analysis using SmartLLM-OG'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    WIKI = 'https://github.com/jasidok/slitheryn/wiki/AI-Enhanced-Analysis'
    WIKI_TITLE = 'AI-Enhanced Security Analysis'
    WIKI_DESCRIPTION = """
    Uses SmartLLM-OG and other specialized AI models to perform comprehensive 
    smart contract security analysis, detecting vulnerabilities that traditional 
    static analysis might miss.
    """
    WIKI_EXPLOIT_SCENARIO = """
    The AI analyzer examines contract code for:
    - Complex reentrancy patterns
    - DeFi-specific attack vectors  
    - Governance vulnerabilities
    - Economic exploits
    - Advanced attack scenarios
    """
    WIKI_RECOMMENDATION = """
    Review all AI-identified vulnerabilities carefully. The AI provides:
    - Detailed attack scenarios
    - Severity assessments
    - Specific fix recommendations
    - Economic impact analysis
    """

    def __init__(self, compilation_unit, slither_instance, logger):
        super().__init__(compilation_unit, slither_instance, logger)
        self.ai_config = get_ai_config()
        self.ollama_client = None
        
        # Initialize AI client if enabled
        if self.ai_config.is_ai_enabled():
            try:
                self.ollama_client = OllamaClient(
                    self.ai_config.get_ollama_url(),
                    self.ai_config,
                    vector_store=getattr(slither_instance, "_vector_store", None),
                    similarity_threshold=getattr(slither_instance, "_similarity_threshold", 0.7),
                    max_similar_contracts=getattr(slither_instance, "_max_similar_contracts", 3),
                )
                logger.info("AI-Enhanced Analysis detector initialized")
            except Exception as e:
                logger.warning(f"Could not initialize AI client: {e}")
                self.ollama_client = None
        else:
            logger.info("AI analysis disabled in configuration")

    def _detect(self) -> List[Output]:
        """Run AI-enhanced vulnerability detection"""
        
        if not self.ollama_client:
            logger.warning("AI client not available, skipping AI analysis")
            return []

        results = []
        
        # Analyze each contract
        for contract in self.compilation_unit.contracts:
            if contract.is_interface or contract.is_library:
                continue
                
            logger.info(f"Running AI analysis on contract: {contract.name}")
            
            try:
                # Get contract source code
                contract_source = self._get_contract_source(contract)
                if not contract_source:
                    logger.warning(f"Could not get source for contract {contract.name}")
                    continue
                
                # Run AI analysis
                ai_result = self.ollama_client.analyze_contract(
                    contract_code=contract_source,
                    contract_name=contract.name,
                    analysis_type="comprehensive"
                )
                
                if ai_result:
                    # Convert AI results to Slitheryn output format
                    output = self._create_output_from_ai_result(contract, ai_result)
                    if output:
                        results.extend(output)
                        logger.info(f"AI analysis completed for {contract.name}: "
                                  f"{len(ai_result.vulnerabilities)} vulnerabilities found")
                else:
                    logger.warning(f"AI analysis failed for contract {contract.name}")
                    
            except Exception as e:
                logger.error(f"Error during AI analysis of {contract.name}: {e}")
                continue
        
        return results

    def _get_contract_source(self, contract) -> str:
        """Extract source code for a contract"""
        try:
            if hasattr(contract, 'source_mapping') and contract.source_mapping:
                source_mapping = contract.source_mapping
                if hasattr(source_mapping, 'filename') and source_mapping.filename:
                    # Try to read the source file
                    with open(source_mapping.filename.absolute, 'r', encoding='utf-8') as f:
                        full_source = f.read()
                    
                    # Extract contract-specific code if possible
                    if hasattr(source_mapping, 'start') and hasattr(source_mapping, 'length'):
                        start = source_mapping.start
                        length = source_mapping.length
                        contract_source = full_source[start:start + length]
                        return contract_source
                    else:
                        # Return full source if we can't isolate the contract
                        return full_source
        except Exception as e:
            logger.debug(f"Error extracting source for {contract.name}: {e}")
        
        # Fallback: try to reconstruct from Slitheryn's representation
        return self._reconstruct_contract_source(contract)

    def _reconstruct_contract_source(self, contract) -> str:
        """Reconstruct contract source from Slitheryn's internal representation"""
        try:
            lines = []
            lines.append(f"contract {contract.name} {{")
            
            # Add state variables
            for var in contract.state_variables:
                visibility = var.visibility if hasattr(var, 'visibility') else 'public'
                var_type = str(var.type) if hasattr(var, 'type') else 'uint256'
                lines.append(f"    {var_type} {visibility} {var.name};")
            
            # Add functions
            for func in contract.functions:
                if func.name in ['constructor', 'fallback', 'receive']:
                    func_name = func.name
                else:
                    func_name = f"function {func.name}"
                
                visibility = func.visibility if hasattr(func, 'visibility') else 'public'
                lines.append(f"    {func_name}() {visibility} {{")
                lines.append(f"        // Function implementation")
                lines.append(f"    }}")
            
            lines.append("}")
            return "\n".join(lines)
            
        except Exception as e:
            logger.debug(f"Error reconstructing source for {contract.name}: {e}")
            return f"contract {contract.name} {{ /* Could not reconstruct source */ }}"

    def _create_output_from_ai_result(self, contract, ai_result: AIAnalysisResult) -> List[Output]:
        """Convert AI analysis result to Slitheryn output format"""
        
        if not ai_result.vulnerabilities:
            return []
        
        outputs = []
        
        for vuln_type in ai_result.vulnerabilities:
            try:
                # Get severity for this vulnerability
                severity = ai_result.severity_scores.get(vuln_type, 'Medium')
                
                # Get relevant attack scenarios
                relevant_scenarios = [
                    scenario for scenario in ai_result.attack_scenarios
                    if vuln_type.replace('_', ' ') in scenario.lower()
                ]
                
                # Get relevant fixes
                relevant_fixes = [
                    fix for fix in ai_result.fix_recommendations
                    if vuln_type.replace('_', ' ') in fix.lower()
                ]
                
                # Create the output
                info = [
                    f"AI-detected vulnerability: {vuln_type.replace('_', ' ').title()}",
                    f"Contract: {contract.name}",
                    f"Severity: {severity}",
                    f"Confidence: {ai_result.confidence_score:.2f}",
                    f"Analysis time: {ai_result.analysis_time:.2f}s",
                    f"Model used: {ai_result.model_used}"
                ]
                
                # Add attack scenarios if available
                if relevant_scenarios:
                    info.append("Attack scenarios:")
                    for i, scenario in enumerate(relevant_scenarios[:2], 1):  # Limit to 2 scenarios
                        info.append(f"  {i}. {scenario[:200]}...")
                
                # Add fix recommendations if available
                if relevant_fixes:
                    info.append("Fix recommendations:")
                    for i, fix in enumerate(relevant_fixes[:3], 1):  # Limit to 3 fixes
                        info.append(f"  {i}. {fix[:150]}...")
                
                # Create output
                similar_contracts = []
                if self.ollama_client and self.ollama_client.vector_store:
                    try:
                        similar = self.ollama_client.vector_store.search_similar(
                            self.ollama_client.vector_store._store.get(contract.name, ([], {}))[0],  # type: ignore
                            top_k=getattr(self.ollama_client, "max_similar_contracts", 3),
                        )
                        similar_contracts = [
                            {"name": s["name"], "score": s["score"]} for s in similar if s["name"] != contract.name
                        ]
                    except Exception:
                        similar_contracts = []

                locations = getattr(ai_result, "vulnerability_locations", {})

                output = Output(
                    info,
                    additional_fields={
                        'vulnerability_type': vuln_type,
                        'ai_confidence': ai_result.confidence_score,
                        'ai_model': ai_result.model_used,
                        'severity': severity,
                        'analysis_time': ai_result.analysis_time,
                        'similar_contracts': similar_contracts,
                        'vulnerability_locations': locations,
                    }
                )
                
                outputs.append(output)
                
            except Exception as e:
                logger.error(f"Error creating output for vulnerability {vuln_type}: {e}")
                continue
        
        return outputs

    def _get_impact_from_severity(self, severity: str) -> DetectorClassification:
        """Convert AI severity to Slitheryn impact classification"""
        severity_lower = severity.lower()
        if severity_lower == 'critical':
            return DetectorClassification.HIGH
        elif severity_lower == 'high':
            return DetectorClassification.HIGH
        elif severity_lower == 'medium':
            return DetectorClassification.MEDIUM
        elif severity_lower == 'low':
            return DetectorClassification.LOW
        else:
            return DetectorClassification.MEDIUM