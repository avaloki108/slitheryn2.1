"""
Ollama API client for SmartLLM integration with Slitheryn
Provides AI-powered smart contract security analysis with multi-agent capabilities
Enhanced with tool-calling support for devstral-2:123b-cloud model
"""

import json
import time
import requests
import asyncio
from typing import Dict, List, Optional, Tuple, Callable, Any
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
    """Client for interacting with Ollama models for security analysis with tool-calling support"""

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
        
        # Tool-calling support
        self.enable_tool_calling = True
        self.available_tools = self._initialize_tools()

    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize available tools for the AI model"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_vulnerability",
                    "description": "Analyze a specific vulnerability type in the contract",
                    "parameters": {
                        "type": "object",
                        "required": ["vulnerability_type", "contract_code"],
                        "properties": {
                            "vulnerability_type": {
                                "type": "string",
                                "description": "Type of vulnerability to analyze (e.g., reentrancy, access_control, integer_overflow)"
                            },
                            "contract_code": {
                                "type": "string", 
                                "description": "The contract code to analyze"
                            },
                            "target_functions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific functions to focus on"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_attack_scenario",
                    "description": "Generate a detailed attack scenario for a vulnerability",
                    "parameters": {
                        "type": "object",
                        "required": ["vulnerability", "contract_code"],
                        "properties": {
                            "vulnerability": {
                                "type": "string",
                                "description": "The vulnerability to create an attack scenario for"
                            },
                            "contract_code": {
                                "type": "string",
                                "description": "The contract code containing the vulnerability"
                            },
                            "attacker_capabilities": {
                                "type": "string",
                                "description": "Assumed attacker capabilities and resources"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend_fix",
                    "description": "Provide specific fix recommendations for a vulnerability",
                    "parameters": {
                        "type": "object",
                        "required": ["vulnerability", "contract_code"],
                        "properties": {
                            "vulnerability": {
                                "type": "string",
                                "description": "The vulnerability to fix"
                            },
                            "contract_code": {
                                "type": "string",
                                "description": "The contract code containing the vulnerability"
                            },
                            "fix_approach": {
                                "type": "string",
                                "enum": ["minimal", "comprehensive", "defensive"],
                                "description": "Approach to take for the fix"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_economic_impact",
                    "description": "Assess the economic impact of a vulnerability",
                    "parameters": {
                        "type": "object",
                        "required": ["vulnerability", "contract_code"],
                        "properties": {
                            "vulnerability": {
                                "type": "string",
                                "description": "The vulnerability to assess"
                            },
                            "contract_code": {
                                "type": "string",
                                "description": "The contract code"
                            },
                            "protocol_context": {
                                "type": "string",
                                "description": "Context about the protocol and tokenomics"
                            }
                        }
                    }
                }
            }
        ]
        return tools

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a specific tool with given arguments"""
        try:
            if tool_name == "analyze_vulnerability":
                return self._tool_analyze_vulnerability(**arguments)
            elif tool_name == "generate_attack_scenario":
                return self._tool_generate_attack_scenario(**arguments)
            elif tool_name == "recommend_fix":
                return self._tool_recommend_fix(**arguments)
            elif tool_name == "assess_economic_impact":
                return self._tool_assess_economic_impact(**arguments)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_analyze_vulnerability(self, vulnerability_type: str, contract_code: str, target_functions: List[str] = None) -> str:
        """Tool implementation for vulnerability analysis"""
        # Create a focused prompt for this specific vulnerability
        prompt = f"""Analyze the {vulnerability_type} vulnerability in this contract:

Contract Code:
{contract_code[:2000]}...

Focus on:
- How the {vulnerability_type} vulnerability manifests
- Which functions are affected
- Conditions that trigger the vulnerability
- Potential impact severity

Provide a concise technical analysis."""
        
        if target_functions:
            prompt += f"\n\nFocus specifically on these functions: {', '.join(target_functions)}"
        
        # Use the model to analyze this specific vulnerability
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.primary_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Analysis failed')
            else:
                return f"Analysis failed: {response.status_code}"
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def _tool_generate_attack_scenario(self, vulnerability: str, contract_code: str, attacker_capabilities: str = "standard") -> str:
        """Tool implementation for attack scenario generation"""
        prompt = f"""Generate a detailed attack scenario for the {vulnerability} vulnerability:

Contract Code:
{contract_code[:2000]}...

Attacker capabilities: {attacker_capabilities}

Provide:
1. Step-by-step attack sequence
2. Required conditions and prerequisites
3. Expected outcome and impact
4. Detection and mitigation challenges"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.primary_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.2,
                        'num_predict': 800
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Scenario generation failed')
            else:
                return f"Scenario generation failed: {response.status_code}"
        except Exception as e:
            return f"Scenario generation error: {str(e)}"

    def _tool_recommend_fix(self, vulnerability: str, contract_code: str, fix_approach: str = "comprehensive") -> str:
        """Tool implementation for fix recommendations"""
        prompt = f"""Provide fix recommendations for the {vulnerability} vulnerability:

Contract Code:
{contract_code[:2000]}...

Fix approach: {fix_approach}

Provide:
1. Specific code changes needed
2. Implementation examples
3. Additional security measures
4. Testing recommendations"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.primary_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 800
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Fix recommendation failed')
            else:
                return f"Fix recommendation failed: {response.status_code}"
        except Exception as e:
            return f"Fix recommendation error: {str(e)}"

    def _tool_assess_economic_impact(self, vulnerability: str, contract_code: str, protocol_context: str = "") -> str:
        """Tool implementation for economic impact assessment"""
        prompt = f"""Assess the economic impact of the {vulnerability} vulnerability:

Contract Code:
{contract_code[:2000]}...

Protocol context: {protocol_context}

Provide:
1. Financial loss potential
2. Market impact assessment
3. User risk exposure
4. Systemic risk considerations"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.primary_model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 600
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'Economic impact assessment failed')
            else:
                return f"Economic impact assessment failed: {response.status_code}"
        except Exception as e:
            return f"Economic impact assessment error: {str(e)}"

    def chat_with_tools(self, messages: List[Dict], model: str = None, tools: List[Dict] = None, max_iterations: int = 5) -> Dict:
        """
        Chat with the model using tool-calling capabilities
        Based on Ollama tool-calling documentation
        """
        if model is None:
            model = self.primary_model
        if tools is None:
            tools = self.available_tools
        
        current_messages = messages.copy()
        
        for iteration in range(max_iterations):
            try:
                # Make request to Ollama chat API with tools
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        'model': model,
                        'messages': current_messages,
                        'stream': False,
                        'tools': tools,
                        'options': {
                            'temperature': self.ai_config.config.temperature,
                            'num_predict': self.ai_config.config.max_tokens
                        }
                    },
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                    break
                
                result = response.json()
                assistant_message = result.get('message', {})
                
                # Add assistant message to conversation
                current_messages.append(assistant_message)
                
                # Check if there are tool calls to execute
                tool_calls = assistant_message.get('tool_calls', [])
                
                if not tool_calls:
                    # No more tool calls, conversation complete
                    break
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function'].get('arguments', {})
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Add tool result to conversation
                    current_messages.append({
                        'role': 'tool',
                        'tool_name': tool_name,
                        'content': str(tool_result)
                    })
                
            except Exception as e:
                logger.error(f"Error in tool-calling iteration {iteration}: {e}")
                break
        
        # Return the final conversation
        return {
            'messages': current_messages,
            'final_message': current_messages[-1] if current_messages else None,
            'iterations': iteration + 1
        }
        
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
        Analyze smart contract code for security vulnerabilities with tool-calling support
        
        Args:
            contract_code: Solidity contract source code
            contract_name: Name of the contract
            analysis_type: Type of analysis ('quick', 'comprehensive', 'reasoning')
        """
        
        model = self._select_model_for_analysis_type(analysis_type)
        if not model:
            logger.error("No suitable model available for analysis")
            return None
        
        start_time = time.time()
        
        # Try tool-calling approach first for devstral-2:123b-cloud
        if self.enable_tool_calling and "devstral-2:123b-cloud" in model:
            try:
                return self._analyze_with_tools(contract_code, contract_name, model, analysis_type, start_time)
            except Exception as e:
                logger.warning(f"Tool-calling analysis failed, falling back to standard analysis: {e}")
        
        # Fallback to standard analysis
        return self._analyze_standard(contract_code, contract_name, model, analysis_type, start_time)
    
    def _analyze_with_tools(self, contract_code: str, contract_name: str, model: str, analysis_type: str, start_time: float) -> AIAnalysisResult:
        """Analyze contract using tool-calling capabilities"""
        
        # Build initial message with tools
        system_prompt = f"""You are an expert Web3 security auditor specializing in smart contract vulnerabilities. 

You have access to specialized tools for comprehensive security analysis. Use these tools to:
1. Analyze specific vulnerability types in detail
2. Generate realistic attack scenarios
3. Provide specific fix recommendations
4. Assess economic impact

Contract to analyze: {contract_name}
Contract code:
{contract_code}

Please perform a comprehensive security analysis using the available tools. Start by identifying potential vulnerabilities, then use the tools to analyze each one in detail."""
        
        messages = [
            {
                "role": "user",
                "content": system_prompt
            }
        ]
        
        # Execute tool-calling analysis
        tool_result = self.chat_with_tools(messages, model, self.available_tools)
        
        analysis_time = time.time() - start_time
        
        # Extract final response
        final_message = tool_result.get('final_message', {})
        raw_response = final_message.get('content', '')
        
        # Parse tool results for structured output
        parsed_result = self._parse_tool_analysis_result(tool_result, raw_response)
        
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
    
    def _analyze_standard(self, contract_code: str, contract_name: str, model: str, analysis_type: str, start_time: float) -> AIAnalysisResult:
        """Standard analysis without tool-calling"""
        
        prompt = self._build_security_analysis_prompt(contract_code, contract_name, analysis_type)
        
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
                raise Exception(f"API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error(f"Analysis timeout after {self.timeout} seconds")
            raise Exception("Analysis timeout")
        except Exception as e:
            logger.error(f"Error during standard AI analysis: {e}")
            raise e
    
    def _parse_tool_analysis_result(self, tool_result: Dict, raw_response: str) -> Dict:
        """Parse results from tool-based analysis"""
        
        # Extract tool results from conversation
        messages = tool_result.get('messages', [])
        tool_responses = []
        
        for message in messages:
            if message.get('role') == 'tool':
                tool_responses.append(message.get('content', ''))
        
        # Combine tool responses with final analysis
        combined_text = raw_response + "\n\n" + "\n\n".join(tool_responses)
        
        # Parse the combined text using existing parser
        return self._parse_ai_response(combined_text)
    
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