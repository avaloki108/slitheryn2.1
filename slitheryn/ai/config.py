"""
AI configuration system for Slitheryn
Manages model selection and AI analysis settings
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("Slitheryn.AI.Config")

@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    primary_model: str = "SmartLLM-OG:latest"
    reasoning_model: str = "phi4-reasoning:latest"
    comprehensive_model: str = "qwen3:30b-a3b"
    ollama_base_url: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.1
    max_tokens: int = 2000
    enable_ai_analysis: bool = True
    confidence_threshold: float = 0.7
    analysis_types: Dict[str, bool] = None
    
    # Multi-Agent System Configuration
    enable_multi_agent: bool = True
    agent_types: list = None
    consensus_threshold: float = 0.7
    parallel_analysis: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.analysis_types is None:
            self.analysis_types = {
                "vulnerability_detection": True,
                "attack_scenarios": True,
                "fix_recommendations": True,
                "severity_assessment": True,
                "false_positive_reduction": True
            }
        
        if self.agent_types is None:
            self.agent_types = [
                "vulnerability",
                "exploit", 
                "fix",
                "economic",
                "governance"
            ]

class AIConfigManager:
    """Manages AI configuration for Slitheryn"""
    
    DEFAULT_CONFIG_PATH = ".slitheryn/ai_config.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self._config = AIModelConfig(**config_data)
                logger.info(f"Loaded AI config from {self.config_path}")
            else:
                self._config = AIModelConfig()
                self._save_config()
                logger.info(f"Created default AI config at {self.config_path}")
        except Exception as e:
            logger.warning(f"Error loading AI config: {e}, using defaults")
            self._config = AIModelConfig()
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(asdict(self._config), f, indent=2)
            logger.debug(f"Saved AI config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving AI config: {e}")
    
    @property
    def config(self) -> AIModelConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated AI config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        self._save_config()
    
    def get_model_priority_list(self, analysis_type: str = "comprehensive") -> list:
        """Get prioritized list of models for analysis type"""
        if analysis_type == "quick":
            return [
                self._config.reasoning_model,
                self._config.primary_model,
                self._config.comprehensive_model
            ]
        elif analysis_type == "reasoning":
            return [
                self._config.reasoning_model,
                self._config.comprehensive_model,
                self._config.primary_model
            ]
        else:  # comprehensive
            return [
                self._config.primary_model,
                self._config.comprehensive_model,
                self._config.reasoning_model
            ]
    
    def is_ai_enabled(self) -> bool:
        """Check if AI analysis is enabled"""
        return self._config.enable_ai_analysis
    
    def get_ollama_url(self) -> str:
        """Get Ollama base URL"""
        return self._config.ollama_base_url
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """Get settings for AI analysis"""
        return {
            'temperature': self._config.temperature,
            'max_tokens': self._config.max_tokens,
            'timeout': self._config.timeout,
            'confidence_threshold': self._config.confidence_threshold
        }
    
    def is_multi_agent_enabled(self) -> bool:
        """Check if multi-agent analysis is enabled"""
        return getattr(self._config, 'enable_multi_agent', True)
    
    def get_multi_agent_config(self) -> Dict[str, Any]:
        """Get multi-agent system configuration"""
        return {
            'enable_multi_agent': getattr(self._config, 'enable_multi_agent', True),
            'agent_types': getattr(self._config, 'agent_types', [
                'vulnerability', 'exploit', 'fix', 'economic', 'governance'
            ]),
            'consensus_threshold': getattr(self._config, 'consensus_threshold', 0.7),
            'parallel_analysis': getattr(self._config, 'parallel_analysis', True),
            'max_workers': getattr(self._config, 'max_workers', 4)
        }

# Global config manager instance
_config_manager = None

def get_ai_config() -> AIConfigManager:
    """Get global AI configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = AIConfigManager()
    return _config_manager

def setup_ai_logging():
    """Setup logging for AI components"""
    ai_logger = logging.getLogger("Slitheryn.AI")
    ai_logger.setLevel(logging.INFO)
    
    # Only add handler if none exists
    if not ai_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[AI] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        ai_logger.addHandler(handler)