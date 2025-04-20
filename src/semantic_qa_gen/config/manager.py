
"""Configuration management for SemanticQAGen."""

import os
import yaml
import logging
from typing import Optional, Dict, Any, Union, List, Set
from pydantic import ValidationError

from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.utils.error import ConfigurationError


class ConfigManager:
    """
    Manages the configuration for SemanticQAGen.
    
    This class handles loading, validating, and accessing configuration settings
    from YAML files and environment variables.
    """
    
    # Environment variable prefix for SemanticQAGen configs
    ENV_PREFIX = "SEMANTICQAGEN_"
    
    def __init__(self, config_path: Optional[str] = None, 
                 config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file.
            config_dict: Dictionary containing configuration values.
            
        Raises:
            ConfigurationError: If both config_path and config_dict are provided,
                or if the configuration fails validation.
        """
        self._config: Optional[SemanticQAGenConfig] = None
        self.logger = logging.getLogger(__name__)
        
        if config_path and config_dict:
            raise ConfigurationError(
                "Both config_path and config_dict were provided. "
                "Please provide only one."
            )
        
        # Check environment variable for config path if not provided
        if not config_path and not config_dict:
            env_config_path = os.environ.get(f"{self.ENV_PREFIX}CONFIG_PATH")
            if env_config_path:
                self.logger.info(f"Using config path from environment: {env_config_path}")
                config_path = env_config_path
        
        if config_path:
            self._load_from_file(config_path)
        elif config_dict:
            self._load_from_dict(config_dict)
        else:
            # Load default configuration
            self._config = SemanticQAGenConfig()
            
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Raises:
            ConfigurationError: If the file cannot be loaded or contains invalid configuration.
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)
                
            self._load_from_dict(config_dict)
            self.logger.info(f"Configuration loaded from {config_path}")
                
        except FileNotFoundError as e:
            raise ConfigurationError(f"Failed to load configuration file: {str(e)}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {str(e)}")
    
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Raises:
            ConfigurationError: If the configuration fails validation.
        """
        try:
            # Process environment variable interpolation
            config_dict = self._process_env_vars(config_dict)
            
            # Validate configuration against schema
            self._config = SemanticQAGenConfig(**config_dict)
            
        except ValidationError as e:
            # Format validation errors in a more user-friendly way
            errors = []
            for error in e.errors():
                loc = '.'.join(str(l) for l in error['loc'])
                msg = error['msg']
                errors.append(f"- {loc}: {msg}")
                
            error_msg = "Invalid configuration:\n" + "\n".join(errors)
            raise ConfigurationError(error_msg)

    def _process_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment variable interpolation in configuration values.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            Dictionary with environment variables interpolated.
        """
        result = {}

        for key, value in config_dict.items():
            if isinstance(value, dict):
                result[key] = self._process_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Basic env var syntax: ${ENV_VAR}
                env_var = value[2:-1]
                env_val = os.environ.get(env_var)

                if env_val is not None:
                    # Try to interpret the value as int, float, or bool if possible
                    if env_val.isdigit():
                        result[key] = int(env_val)
                    elif env_val.replace('.', '', 1).isdigit() and env_val.count('.') < 2:
                        result[key] = float(env_val)
                    elif env_val.lower() in ('true', 'false'):
                        result[key] = env_val.lower() == 'true'
                    else:
                        result[key] = env_val
                else:
                    # Keep the original value if env var not found
                    result[key] = value
            elif isinstance(value, list):
                # Process lists for env vars
                processed_list = []
                for item in value:
                    if isinstance(item, dict):
                        processed_list.append(self._process_env_vars(item))
                    elif isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                        env_var = item[2:-1]
                        env_val = os.environ.get(env_var)
                        if env_val is not None:
                            # Type conversion for env values
                            if env_val.isdigit():
                                processed_list.append(int(env_val))
                            elif env_val.replace('.', '', 1).isdigit() and env_val.count('.') < 2:
                                processed_list.append(float(env_val))
                            elif env_val.lower() in ('true', 'false'):
                                processed_list.append(env_val.lower() == 'true')
                            else:
                                processed_list.append(env_val)
                        else:
                            processed_list.append(item)
                    else:
                        processed_list.append(item)
                result[key] = processed_list
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self) -> None:
        """
        Apply configuration overrides from environment variables.
        
        This allows users to override any configuration option using environment 
        variables with the pattern SEMANTICQAGEN_SECTION_KEY=value.
        """
        # Find all relevant environment variables
        prefix = self.ENV_PREFIX
        overrides = {}
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
                
            # Remove prefix and split into parts
            config_path = env_key[len(prefix):].lower().split('_')
            
            # Skip if empty path
            if not config_path or config_path[0] == 'config_path':
                continue
                
            # Convert value to appropriate type
            if env_value.isdigit():
                typed_value = int(env_value)
            elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') < 2:
                typed_value = float(env_value)
            elif env_value.lower() in ('true', 'false'):
                typed_value = env_value.lower() == 'true'
            else:
                typed_value = env_value
                
            # Build nested structure
            current = overrides
            for i, part in enumerate(config_path[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set the value at the last level
            current[config_path[-1]] = typed_value
        
        # Apply overrides
        if overrides:
            self._apply_overrides_recursive(overrides, self._config.dict())
            
            # Recreate config with overrides
            try:
                updated_dict = self._config.dict()
                self._config = SemanticQAGenConfig(**updated_dict)
                self.logger.debug(f"Applied {len(overrides)} environment variable overrides")
            except ValidationError as e:
                self.logger.error(f"Failed to apply environment overrides: {e}")
    
    def _apply_overrides_recursive(self, overrides: Dict[str, Any], target: Dict[str, Any]) -> None:
        """Apply nested overrides recursively."""
        for key, value in overrides.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    # Recursively update nested dictionaries
                    self._apply_overrides_recursive(value, target[key])
                else:
                    # Update the value directly
                    target[key] = value
    
    @property
    def config(self) -> SemanticQAGenConfig:
        """
        Get the validated configuration object.
        
        Returns:
            The configuration object.
            
        Raises:
            ConfigurationError: If the configuration has not been loaded.
        """
        if self._config is None:
            raise ConfigurationError("Configuration has not been loaded.")
        return self._config
    
    def get_section(self, section_name: str) -> Any:
        """
        Get a specific section of the configuration.
        
        Args:
            section_name: Name of the configuration section.
            
        Returns:
            The requested configuration section.
            
        Raises:
            ConfigurationError: If the section does not exist.
        """
        if not hasattr(self.config, section_name):
            raise ConfigurationError(f"Configuration section not found: {section_name}")
        
        return getattr(self.config, section_name)
    
    def save_config(self, file_path: str, include_derived: bool = False) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            file_path: Path where to save the configuration.
            include_derived: Whether to include derived values.
            
        Raises:
            ConfigurationError: If the configuration cannot be saved.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Get config as dict
            config_dict = self.config.dict(exclude_none=True, exclude_unset=not include_derived)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
                
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def create_default_config_file(self, file_path: str, include_comments: bool = True) -> None:
        """
        Create a default configuration file with comments.
        
        Args:
            file_path: Path where to save the configuration.
            include_comments: Whether to include descriptive comments.
            
        Raises:
            ConfigurationError: If the configuration file cannot be created.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Get default config as dict
            config_dict = SemanticQAGenConfig().dict(exclude_none=True)
            
            if include_comments:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write("# SemanticQAGen Configuration\n\n")
                    
                    # Write each section with comments
                    for section, section_data in config_dict.items():
                        file.write(f"# {section.replace('_', ' ').title()} settings\n")
                        file.write(f"{section}:\n")
                        
                        # Write section content with proper indentation
                        section_yaml = yaml.dump({section: section_data}, default_flow_style=False)
                        indented_section = '\n'.join(section_yaml.split('\n')[1:])  # Skip the section key
                        file.write(indented_section)
                        file.write("\n")
            else:
                # Simple YAML dump without comments
                with open(file_path, 'w', encoding='utf-8') as file:
                    yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
                    
            self.logger.info(f"Default configuration saved to {file_path}")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to create default configuration: {str(e)}")
