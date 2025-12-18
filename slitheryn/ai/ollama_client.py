"""
Ollama API client for SmartLLM integration with Slitheryn
Provides AI-powered smart contract security analysis with multi-agent capabilities
"""

import json
import time
import requests
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("Slitheryn.AI")

@dataclass
class AIAnalysisResult:
    """Result from AI vulnerability analysis"""
    vulnerabilities: List[str]
    severity_scores: Dict[str, str]
    attack_scenarios: List[str]
    fix_recommendations: List[str]
    confidence_score: float
    analysis_time: float
    model_used: str
    raw_response: str

class OllamaClient:
    """Client for interacting with Ollama models for security analysis"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        config_manager=None,
        vector_store=None,
        similarity_threshold: float = 0.7,
        max_similar_contracts: int = 3,
    ):
        from slitheryn.ai.config import get_ai_config

        self.base_url = base_url
        self.timeout = 120
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.max_similar_contracts = max_similar_contracts

        # Get configuration from config manager
        if config_manager is None:
            self.ai_config = get_ai_config()
        else:
            self.ai_config = config_manager

        # Load model names from configuration
        self.primary_model = self.ai_config.config.primary_model
        self.reasoning_model = self.ai_config.config.reasoning_model
        self.comprehensive_model = self.ai_config.config.comprehensive_model
        
    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)  # Increased timeout for large models
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                return model_name in available_models
            return False
        except Exception as e:
            logger.warning(f"Could not check model availability: {e}")
            return False
    
    def get_best_available_model(self) -> str:
        """Get the best available model for analysis"""
        models_priority = [
            self.primary_model,
            self.reasoning_model, 
            self.comprehensive_model
        ]
        
        for model in models_priority:
            if self.check_model_availability(model):
                logger.info(f"Using model: {model}")
                return model
        
        # Fallback - check what models are actually available
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    fallback_model = models[0]['name']
                    logger.warning(f"Primary models not available, using fallback: {fallback_model}")
                    return fallback_model
        except Exception as e:
            logger.error(f"Could not get available models: {e}")
        
        logger.error("No Ollama models available!")
        return None
    
    def analyze_contract(self, 
                        contract_code: str, 
                        contract_name: str = "Unknown",
                        analysis_type: str = "comprehensive") -> Optional[AIAnalysisResult]:
        """
        Analyze smart contract code for security vulnerabilities
        
        Args:
            contract_code: Solidity contract source code
            contract_name: Name of the contract
            analysis_type: Type of analysis ('quick', 'comprehensive', 'reasoning')
        """
        
        model = self._select_model_for_analysis_type(analysis_type)
        if not model:
            logger.error("No suitable model available for analysis")
            return None
            
        prompt = self._build_security_analysis_prompt(contract_code, contract_name, analysis_type)
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Low temperature for consistent analysis
                        'top_p': 0.9,
                        'num_predict': 2000
                    }
                },
                timeout=self.timeout
            )
            
            analysis_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                # Parse the AI response
                parsed_result = self._parse_ai_response(raw_response)
                
                return AIAnalysisResult(
                    vulnerabilities=parsed_result['vulnerabilities'],
                    severity_scores=parsed_result['severity_scores'],
                    attack_scenarios=parsed_result['attack_scenarios'],
                    fix_recommendations=parsed_result['fix_recommendations'],
                    confidence_score=parsed_result['confidence_score'],
                    analysis_time=analysis_time,
                    model_used=model,
                    raw_response=raw_response
                )
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Analysis timeout after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            return None
    
    def _select_model_for_analysis_type(self, analysis_type: str) -> Optional[str]:
        """Select the best model based on analysis type"""
        if analysis_type == "quick":
            # For quick analysis, prefer faster models
            candidates = [self.reasoning_model, self.primary_model]
        elif analysis_type == "reasoning":
            # For complex reasoning, prefer reasoning-specialized models
            candidates = [self.reasoning_model, self.comprehensive_model, self.primary_model]
        else:  # comprehensive
            # For comprehensive analysis, prefer accuracy
            candidates = [self.primary_model, self.comprehensive_model, self.reasoning_model]
        
        for model in candidates:
            if self.check_model_availability(model):
                return model
        
        return self.get_best_available_model()
    
    def _build_security_analysis_prompt(self, contract_code: str, contract_name: str, analysis_type: str) -> str:
        """Build a comprehensive prompt for security analysis"""
        
        base_prompt = f"""You are an expert Web3 security auditor specializing in smart contract vulnerabilities, DeFi exploits, and governance attacks.

Analyze this Solidity smart contract for security vulnerabilities:

Contract Name: {contract_name}
Contract Code:
{contract_code}

Provide comprehensive security analysis focusing on:
1. Smart contract vulnerabilities (reentrancy, access control, integer overflow, etc.)
2. DeFi-specific attack vectors (flash loans, price manipulation, MEV)
3. Governance vulnerabilities (voting manipulation, proposal attacks)
4. Economic exploits and tokenomics issues
5. Gas optimization and DoS vulnerabilities

For each vulnerability found, provide:
- Vulnerability name and type
- Severity level (Critical/High/Medium/Low)
- Affected functions and line numbers
- Step-by-step attack scenario
- Economic impact assessment
- Specific fix recommendations with code examples

Format your response as structured analysis with clear sections."""

        if analysis_type == "quick":
            base_prompt += "\n\nProvide a concise but thorough analysis focusing on the most critical vulnerabilities."
        elif analysis_type == "reasoning":
            base_prompt += "\n\nShow your step-by-step reasoning process and logical analysis for each vulnerability."
        else:  # comprehensive
            base_prompt += "\n\nProvide exhaustive analysis covering all possible security issues and attack vectors."
        
        rag_context = self._get_rag_context(contract_code, contract_name)
        if rag_context:
            base_prompt += "\n\nContext from similar contracts in this codebase:\n"
            base_prompt += rag_context

        base_prompt += """
\n\nReturn results in JSON with the following keys:
- vulnerabilities: array of vulnerability identifiers
- severity_scores: object mapping vulnerability -> Critical/High/Medium/Low
- attack_scenarios: array of concise step-by-step scenarios
- fix_recommendations: array of actionable fixes
- confidence_score: float between 0 and 1
"""

        return base_prompt

    def _get_rag_context(self, contract_code: str, contract_name: str) -> str:
        """
        Retrieve similar contract snippets using the vector store.
        """
        if not self.vector_store:
            return ""
        # If we have an embedding for this contract already, use similarity search
        if contract_name in getattr(self.vector_store, "_store", {}):
            contexts = self.vector_store.get_context_for_contract(
                contract_name, self.max_similar_contracts
            )
        else:
            # Fallback: embed on the fly using the embedding service is not available here.
            contexts = []

        if not contexts:
            return ""

        joined = []
        for idx, ctx in enumerate(contexts, 1):
            joined.append(f"[Similar #{idx}]\n{ctx[:1200]}")
        return "\n\n".join(joined)
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response to extract structured vulnerability information"""
        
        # Try structured JSON first
        structured = self._parse_json_response(response)
        if structured:
            return structured

        response_lower = response.lower()
        
        # Extract vulnerabilities mentioned
        vulnerability_keywords = {
            'reentrancy': ['reentrancy', 're-entrancy', 'reentrant'],
            'access_control': ['access control', 'unauthorized', 'permission', 'onlyowner'],
            'integer_overflow': ['overflow', 'underflow', 'arithmetic'],
            'flash_loan_attack': ['flash loan', 'flashloan'],
            'price_manipulation': ['price manipulation', 'oracle', 'price oracle'],
            'governance_attack': ['governance', 'voting', 'proposal'],
            'dos': ['denial of service', 'dos', 'gas limit'],
            'front_running': ['front-run', 'mev', 'sandwich'],
            'time_manipulation': ['timestamp', 'block.timestamp', 'time'],
            'uninitialized_storage': ['uninitialized', 'storage'],
            'delegatecall': ['delegatecall', 'delegate call'],
            'tx_origin': ['tx.origin', 'tx origin'],
            'mev': ['mev', 'sandwich', 'front run'],
            'oracle': ['oracle', 'price feed', 'manipulation'],
            'upgradeability': ['proxy', 'upgrade', 'delegatecall upgrade'],
            'authorization': ['admin', 'owner', 'privileged'],
        }
        
        found_vulnerabilities = []
        severity_scores = {}
        vulnerability_locations: Dict[str, List[str]] = {}

        lines = response.splitlines()
        
        for vuln_type, keywords in vulnerability_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    found_vulnerabilities.append(vuln_type)
                    # Try to extract severity
                    severity = self._extract_severity_for_vulnerability(response_lower, keyword)
                    if severity:
                        severity_scores[vuln_type] = severity
                    # Capture potential function context
                    for line in lines:
                        lower = line.lower()
                        if keyword in lower and "function" in lower:
                            vulnerability_locations.setdefault(vuln_type, []).append(line.strip())
                    break
        
        # Extract attack scenarios
        attack_scenarios = self._extract_attack_scenarios(response)
        
        # Extract fix recommendations
        fix_recommendations = self._extract_fix_recommendations(response)
        
        # Calculate confidence score based on response quality
        confidence_score = self._calculate_confidence_score(response, found_vulnerabilities)
        
        # Filter out weak findings
        validated = []
        for vuln in found_vulnerabilities:
            if self._validate_vulnerability_finding(
                vuln, severity_scores, attack_scenarios, fix_recommendations
            ):
                validated.append(vuln)

        return {
            'vulnerabilities': list(set(validated)),
            'severity_scores': severity_scores,
            'attack_scenarios': attack_scenarios,
            'fix_recommendations': fix_recommendations,
            'vulnerability_locations': vulnerability_locations,
            'confidence_score': confidence_score
        }

    def _validate_vulnerability_finding(
        self,
        vuln_type: str,
        severity_scores: Dict[str, str],
        attack_scenarios: List[str],
        fix_recommendations: List[str],
    ) -> bool:
        """
        Basic validation to reduce false positives.
        """
        has_severity = vuln_type in severity_scores
        has_scenario = any(vuln_type.replace("_", " ") in s.lower() for s in attack_scenarios)
        has_fix = any(vuln_type.replace("_", " ") in f.lower() for f in fix_recommendations)
        similar_context = False
        if self.vector_store:
            similar_context = bool(
                self.vector_store.search_similar(
                    self.vector_store._store.get(vuln_type, ([], {}))[0] if hasattr(self.vector_store, "_store") else [],  # type: ignore
                    top_k=1,
                )
            )
        return has_severity or has_scenario or has_fix or similar_context

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Attempt to parse a structured JSON response.
        Expected keys: vulnerabilities, severity_scores, attack_scenarios, fix_recommendations, confidence_score
        """
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return None
            vulnerabilities = data.get("vulnerabilities") or data.get("issues")
            if vulnerabilities is None:
                return None
            severity_scores = data.get("severity_scores", {})
            attack_scenarios = data.get("attack_scenarios", [])
            fix_recommendations = data.get("fix_recommendations", [])
            confidence = data.get("confidence_score", 0.5)
            return {
                "vulnerabilities": vulnerabilities,
                "severity_scores": severity_scores,
                "attack_scenarios": attack_scenarios,
                "fix_recommendations": fix_recommendations,
                "confidence_score": confidence,
            }
        except Exception:
            return None
    
    def _extract_severity_for_vulnerability(self, response_lower: str, vulnerability_keyword: str) -> Optional[str]:
        """Extract severity level for a specific vulnerability"""
        # Look for severity keywords near the vulnerability mention
        severity_keywords = ['critical', 'high', 'medium', 'low']
        
        # Find position of vulnerability keyword
        vuln_pos = response_lower.find(vulnerability_keyword)
        if vuln_pos == -1:
            return None
        
        # Look in a window around the vulnerability mention
        window_start = max(0, vuln_pos - 200)
        window_end = min(len(response_lower), vuln_pos + 200)
        window_text = response_lower[window_start:window_end]
        
        for severity in severity_keywords:
            if severity in window_text:
                return severity.capitalize()
        
        return None
    
    def _extract_attack_scenarios(self, response: str) -> List[str]:
        """Extract attack scenarios from the response"""
        scenarios = []
        
        # Look for common attack scenario indicators
        scenario_indicators = [
            'attack scenario', 'exploit', 'attacker', 'malicious',
            'step 1', 'step 2', 'first', 'then', 'finally'
        ]
        impact_terms = ['profit', 'loss', 'drain', 'steal', 'impact', 'cost']
        prereq_terms = ['requires', 'assumes', 'needs', 'prerequisite']
        
        lines = response.split('\n')
        current_scenario = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(indicator in line_lower for indicator in scenario_indicators):
                if current_scenario:
                    scenarios.append(' '.join(current_scenario))
                    current_scenario = []
                current_scenario.append(line.strip())
            elif current_scenario and line.strip():
                current_scenario.append(line.strip())
                if len(current_scenario) > 5:  # Limit scenario length
                    scenarios.append(' '.join(current_scenario))
                    current_scenario = []
            elif any(term in line_lower for term in impact_terms):
                current_scenario.append(f"Impact: {line.strip()}")
            elif any(term in line_lower for term in prereq_terms):
                current_scenario.append(f"Prerequisite: {line.strip()}")
        
        if current_scenario:
            scenarios.append(' '.join(current_scenario))
        
        return scenarios[:5]  # Limit to 5 scenarios
    
    def _extract_fix_recommendations(self, response: str) -> List[str]:
        """Extract fix recommendations from the response"""
        recommendations = []
        code_block = []
        in_code = False
        
        # Look for fix recommendation indicators
        fix_indicators = [
            'fix', 'recommend', 'solution', 'prevent', 'mitigate',
            'use', 'implement', 'add', 'require', 'check'
        ]
        
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("```"):
                if in_code and code_block:
                    recommendations.append("CODE:" + "\n".join(code_block))
                    code_block = []
                in_code = not in_code
                continue
            if in_code:
                code_block.append(line.rstrip())
                continue
            if any(indicator in line_lower for indicator in fix_indicators) and len(line.strip()) > 20:
                recommendations.append(line.strip())
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def _calculate_confidence_score(self, response: str, vulnerabilities: List[str]) -> float:
        """Calculate confidence score based on response quality"""
        score = 0.0
        
        # Base score for finding vulnerabilities
        if vulnerabilities:
            score += 0.3
        
        # Score for response length (comprehensive analysis)
        if len(response) > 1000:
            score += 0.2
        elif len(response) > 500:
            score += 0.1
        
        # Score for technical terms
        technical_terms = [
            'call', 'state', 'balance', 'msg.sender', 'require',
            'modifier', 'function', 'contract', 'solidity'
        ]
        
        terms_found = sum(1 for term in technical_terms if term in response.lower())
        score += min(0.3, terms_found * 0.05)
        
        # Score for fix recommendations
        if 'recommend' in response.lower() or 'fix' in response.lower():
            score += 0.2
        
        return min(1.0, score)
    
    async def analyze_contract_multi_agent(self, 
                                         contract_code: str, 
                                         contract_name: str = "Unknown",
                                         analysis_type: str = "comprehensive") -> Optional[Dict]:
        """
        Perform multi-agent analysis of smart contract code
        
        Args:
            contract_code: Solidity contract source code
            contract_name: Name of the contract
            analysis_type: Type of analysis ('quick', 'comprehensive', 'specialized')
            
        Returns:
            Multi-agent analysis results or None if not available
        """
        try:
            # Try to import and use multi-agent system
            from integrations.web3_audit_system import create_multi_agent_system
            from slitheryn.ai.config import get_ai_config
            
            ai_config = get_ai_config()
            
            # Check if multi-agent is enabled
            if not ai_config.is_multi_agent_enabled():
                logger.info("Multi-agent analysis disabled, falling back to single-agent")
                return None
            
            # Create multi-agent system
            audit_system = create_multi_agent_system(self, ai_config)
            
            if not audit_system.is_available():
                logger.warning("Multi-agent system not available, using single-agent analysis")
                return None
            
            logger.info(f"🤖 Starting multi-agent analysis with {len(audit_system.get_available_agents())} agents")
            
            # Perform multi-agent analysis
            result = await audit_system.audit(contract_code, contract_name, analysis_type)
            
            if result:
                logger.info(f"✅ Multi-agent analysis completed:")
                logger.info(f"   - Consensus vulnerabilities: {len(result.consensus_vulnerabilities)}")
                logger.info(f"   - Consensus score: {result.consensus_score:.2f}")
                logger.info(f"   - Analysis time: {result.total_analysis_time:.2f}s")
                
                # Convert to dictionary for compatibility
                return {
                    'multi_agent': True,
                    'consensus_vulnerabilities': result.consensus_vulnerabilities,
                    'final_severity_scores': result.final_severity_scores,
                    'attack_scenarios': result.attack_scenarios,
                    'fix_recommendations': result.fix_recommendations,
                    'economic_impact': result.economic_impact_assessment,
                    'governance_risks': result.governance_risks,
                    'consensus_score': result.consensus_score,
                    'analysis_time': result.total_analysis_time,
                    'models_used': result.models_used,
                    'agent_results': [
                        {
                            'agent_type': agent_result.agent_type.value,
                            'vulnerabilities': agent_result.vulnerabilities,
                            'confidence': agent_result.confidence_score,
                            'model_used': agent_result.model_used
                        }
                        for agent_result in result.agent_results
                    ],
                    'full_report': audit_system.generate_report(result)
                }
            
            return None
            
        except ImportError:
            logger.debug("Multi-agent system not available (integrations not found)")
            return None
        except Exception as e:
            logger.error(f"Error in multi-agent analysis: {e}")
            return None