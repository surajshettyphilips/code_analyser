"""
Configuration loader for the application.
Loads settings from config.yaml and environment variables.
"""
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the config.yaml file
        """
        # Load environment variables
        load_dotenv()
        
        # Load YAML config
        self.config = self._load_yaml_config(config_path)
        
        # Override with environment variables
        self._override_with_env()
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def _override_with_env(self):
        """Override configuration with environment variables."""
        # Chunking settings
        if os.getenv('CHUNK_SIZE'):
            self.config['chunking']['chunk_size'] = int(os.getenv('CHUNK_SIZE'))
        if os.getenv('CHUNK_OVERLAP'):
            self.config['chunking']['overlap_size'] = int(os.getenv('CHUNK_OVERLAP'))
        
        # ChromaDB settings
        if os.getenv('CHROMADB_PERSIST_DIR'):
            self.config['chromadb']['persist_directory'] = os.getenv('CHROMADB_PERSIST_DIR')
        
        # LLM settings
        if os.getenv('OLLAMA_MODEL'):
            self.config['llm']['model'] = os.getenv('OLLAMA_MODEL')
        if os.getenv('OLLAMA_BASE_URL'):
            self.config['llm']['base_url'] = os.getenv('OLLAMA_BASE_URL')
        
        # DSPy settings
        if os.getenv('DSPY_CACHE_DIR'):
            self.config['dspy']['cache_dir'] = os.getenv('DSPY_CACHE_DIR')
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to the config key (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self.config.get('chunking', {})
    
    def get_chromadb_config(self) -> Dict[str, Any]:
        """Get ChromaDB configuration."""
        return self.config.get('chromadb', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get('llm', {})
    
    def get_dspy_config(self) -> Dict[str, Any]:
        """Get DSPy configuration."""
        return self.config.get('dspy', {})


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
