# filename: semantic_qa_gen/config/manager.py

"""Configuration management for SemanticQAGen."""

import os
import re
import yaml # Standard PyYAML for loading
import logging
from typing import Optional, Dict, Any, Union, List, Type
# Use Pydantic V2 imports
from pydantic import BaseModel, ValidationError
from copy import deepcopy
import warnings

# Use ruamel.yaml for comment-preserving dump
try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarstring import PreservedScalarString
    RUAMEL_YAML_AVAILABLE = True
except ImportError:
    RUAMEL_YAML_AVAILABLE = False
    PreservedScalarString = str # Fallback type
    CommentedMap = dict # Fallback type
    CommentedSeq = list # Fallback type
    YAML = None # Fallback type

from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.utils.error import ConfigurationError

class ConfigManager:
    """
    Manages the configuration for SemanticQAGen using Pydantic V2.

    Handles loading, validation, merging, defaults, and environment variables.
    Uses ruamel.yaml (optional) for comment preservation when saving.
    """
    ENV_PREFIX = "SEMANTICQAGEN_"

    def __init__(self, config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        """Initializes the ConfigManager."""
        self._config: Optional[SemanticQAGenConfig] = None
        self.logger = logging.getLogger(__name__)

        if config_path and config_dict:
            raise ConfigurationError("Provide either config_path or config_dict, not both.")

        load_source: Optional[str] = None
        initial_config_dict: Optional[Dict[str, Any]] = None

        # Determine config source
        if not config_path and not config_dict:
            env_config_path = os.environ.get(f"{self.ENV_PREFIX}CONFIG_PATH")
            if env_config_path:
                self.logger.info(f"Using config path from env var {self.ENV_PREFIX}CONFIG_PATH: {env_config_path}")
                config_path = env_config_path

        if config_path:
            initial_config_dict = self._load_dict_from_file(config_path)
            load_source = f"file: {config_path}"
        elif config_dict:
            initial_config_dict = deepcopy(config_dict) # Use a copy
            load_source = "dictionary"
        else:
            # Get defaults using Pydantic V2's model_construct or default factory on instantiation
            initial_config_dict = SemanticQAGenConfig().model_dump(exclude_unset=False) # Get all defaults
            load_source = "defaults"
            self.logger.info("No configuration source provided, using default settings.")

        if initial_config_dict is None:
             raise ConfigurationError("Failed to determine any configuration source.")

        # Process environment variable interpolation (${VAR})
        interpolated_config = self._process_env_vars_interpolation(initial_config_dict)

        # Apply environment variable overrides (SEMANTICQAGEN_SECTION_KEY=value)
        final_config_dict = self._apply_env_overrides(interpolated_config)

        # Validate the final configuration dictionary using Pydantic V2
        try:
            self._config = SemanticQAGenConfig(**final_config_dict)
            self.logger.info(f"Configuration successfully loaded and validated from {load_source}.")
            self.logger.debug(f"Final configuration: {self._config.model_dump_json(indent=2)}")
        except ValidationError as e:
            error_msg = self._format_validation_error(e)
            raise ConfigurationError(f"Invalid configuration from {load_source}:\n{error_msg}")
        except Exception as e: # Catch other potential model creation errors
             self.logger.exception(f"Unexpected error parsing final config from {load_source}: {e}", exc_info=True)
             raise ConfigurationError(f"Unexpected error creating config model from {load_source}: {e}")

    def _load_dict_from_file(self, config_path: str) -> Dict[str, Any]:
        """Loads dict from YAML file."""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        if not os.path.isfile(config_path):
             raise ConfigurationError(f"Configuration path is not a file: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Basic check for empty file
                if not content.strip():
                    self.logger.warning(f"Configuration file {config_path} is empty. Using defaults might occur.")
                    return {}
                config_dict = yaml.safe_load(content)
                if not isinstance(config_dict, dict):
                     # Allow empty dict if file was empty after stripping comments etc.
                     if config_dict is None:
                          return {}
                     raise ConfigurationError(f"Config file {config_path} root is not a YAML dictionary (type: {type(config_dict)}).")
                return config_dict if config_dict else {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        except IOError as e:
            raise ConfigurationError(f"Cannot read config file {config_path}: {e}")
        except Exception as e:
             raise ConfigurationError(f"Unexpected error loading config file {config_path}: {e}")

    def _process_env_vars_interpolation(self, data: Any) -> Any:
        """Recursively replaces ${VAR} patterns with environment variable values."""
        if isinstance(data, dict):
            return {k: self._process_env_vars_interpolation(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_env_vars_interpolation(item) for item in data]
        elif isinstance(data, str):
            # Use regex to find all ${VAR} patterns, not just start/end
            def replace_env(match):
                env_var_name = match.group(1)
                env_value = os.environ.get(env_var_name)
                if env_value is not None:
                    self.logger.debug(f"Interpolating env var '{env_var_name}'")
                    # Attempt type conversion for overrides later, return string here
                    return env_value
                else:
                    self.logger.warning(f"Env var '{env_var_name}' not found for interpolation. Keeping placeholder.")
                    return match.group(0) # Keep original ${VAR} string

            # Regex to find ${VAR_NAME} - basic name matching
            # Allows for VAR_NAME or VAR_NAME_1 etc.
            interpolated_str = re.sub(r'\$\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}', replace_env, data)
            return interpolated_str
        else:
            return data

    def _apply_env_overrides(self, base_config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Applies SEMANTICQAGEN_SECTION_KEY=value overrides."""
        overridden_config = deepcopy(base_config_dict)
        overrides_applied = 0

        for env_key, raw_env_value in os.environ.items():
            if not env_key.startswith(self.ENV_PREFIX): continue

            config_path_parts = env_key[len(self.ENV_PREFIX):].lower().split('_')
            if not config_path_parts or config_path_parts == ['config', 'path']: continue

            # Attempt to infer type from raw value
            try:
                 if raw_env_value.lower() == 'true': typed_value = True
                 elif raw_env_value.lower() == 'false': typed_value = False
                 elif '.' in raw_env_value: typed_value = float(raw_env_value)
                 else: typed_value = int(raw_env_value)
            except ValueError:
                 typed_value = raw_env_value # Keep as string

            current_level = overridden_config
            try:
                 # Navigate or create path in the config dict
                 for i, part in enumerate(config_path_parts[:-1]):
                     if part not in current_level or not isinstance(current_level.get(part), dict):
                         # If path intermediate doesn't exist or is not dict, create it
                         current_level[part] = {}
                     current_level = current_level[part]
                 final_key = config_path_parts[-1]
                 current_level[final_key] = typed_value # Set the value
                 self.logger.debug(f"Applying environment override: {'_'.join(config_path_parts)} = {typed_value} (type: {type(typed_value).__name__})")
                 overrides_applied += 1
            except Exception as e:
                 self.logger.error(f"Error applying env override {env_key}={raw_env_value}: {e}")

        if overrides_applied > 0:
             self.logger.info(f"Applied {overrides_applied} environment variable override(s).")
        return overridden_config

    def _format_validation_error(self, error: ValidationError) -> str:
         """Formats Pydantic V2 validation errors."""
         error_lines = []
         for err in error.errors():
             loc = " -> ".join(map(str, err['loc']))
             msg = err['msg']
             # Include input value if helpful and not too large
             inp = str(err.get('input'))
             inp_str = f" (Input: {inp[:50]}{'...' if len(inp) > 50 else ''})" if inp else ""
             error_lines.append(f"  - {msg}, Location: [{loc}]{inp_str}")
         return "Configuration validation failed:\n" + "\n".join(error_lines)

    @property
    def config(self) -> SemanticQAGenConfig:
        """Gets the validated configuration object."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded or validated.")
        return self._config

    def get_section(self, section_name: str) -> Any:
        """Gets a specific top-level config section."""
        if self._config is None:
             raise ConfigurationError("Configuration not loaded.")
        try:
            # Access section directly as attribute
            section = getattr(self.config, section_name)
            return section
        except AttributeError:
             raise AttributeError(f"Configuration section '{section_name}' not found.")

    def save_config(self, file_path: str, include_comments: bool = True) -> None:
        """Saves the *current* config state to YAML file."""
        if self._config is None:
            raise ConfigurationError("Cannot save: Configuration not loaded.")

        # Use model_dump for Pydantic V2
        config_dict = self.config.model_dump(mode='python', exclude_none=True)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        except OSError as e:
            raise ConfigurationError(f"Failed to create directory for config {file_path}: {e}")

        try:
            if include_comments and RUAMEL_YAML_AVAILABLE and YAML is not None:
                 yaml_dumper = YAML()
                 yaml_dumper.indent(mapping=2, sequence=4, offset=2)
                 yaml_dumper.preserve_quotes = True
                 commented_config = self._dict_to_commented_map(config_dict, SemanticQAGenConfig)
                 with open(file_path, 'w', encoding='utf-8') as file:
                      yaml_dumper.dump(commented_config, file)
                 self.logger.info(f"Configuration with comments saved to {file_path}")
            else:
                 if include_comments and not RUAMEL_YAML_AVAILABLE:
                      self.logger.warning("ruamel.yaml not installed. Saving config without comments.")
                 with open(file_path, 'w', encoding='utf-8') as file:
                      yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False, indent=2)
                 self.logger.info(f"Configuration saved to {file_path} (no comments).")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {str(e)}")

    def create_default_config_file(self, file_path: str, include_comments: bool = True) -> None:
        """Creates a default configuration YAML file."""
        try:
            default_config_instance = SemanticQAGenConfig()
            # Use model_dump for Pydantic V2, include defaults explicitly
            default_config_dict = default_config_instance.model_dump(mode='python', exclude_unset=False, exclude_none=True)

            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            if include_comments and RUAMEL_YAML_AVAILABLE and YAML is not None:
                 yaml_dumper = YAML()
                 yaml_dumper.indent(mapping=2, sequence=4, offset=2)
                 yaml_dumper.preserve_quotes = True
                 # Pass the dict and the model class to helper for comment extraction
                 commented_defaults = self._dict_to_commented_map(default_config_dict, SemanticQAGenConfig)
                 with open(file_path, 'w', encoding='utf-8') as file:
                      file.write("# SemanticQAGen Default Configuration\n")
                      file.write("# Modify values as needed. Env vars can override settings (e.g., SEMANTICQAGEN_LOCAL_ENABLED=true).\n\n")
                      yaml_dumper.dump(commented_defaults, file)
                 self.logger.info(f"Default config with comments saved to {file_path}")
            else:
                 if include_comments and not RUAMEL_YAML_AVAILABLE:
                      self.logger.warning("ruamel.yaml not installed. Saving default config without comments.")
                 with open(file_path, 'w', encoding='utf-8') as file:
                      yaml.dump(default_config_dict, file, default_flow_style=False, sort_keys=False, indent=2)
                 self.logger.info(f"Default config saved to {file_path} (no comments).")
        except Exception as e:
             self.logger.exception(f"Failed creating default config at {file_path}: {e}", exc_info=True)
             raise ConfigurationError(f"Failed to create default configuration: {str(e)}")

    def _get_field_description(self, model_class: Type[BaseModel], field_name: str) -> Optional[str]:
        """Safely extracts field description from Pydantic V2 model."""
        if not model_class or not hasattr(model_class, 'model_fields'):
            return None
        field_info = model_class.model_fields.get(field_name)
        return field_info.description if field_info else None

    def _get_nested_model_class(self, parent_model_class: Type[BaseModel], field_name: str) -> Optional[Type[BaseModel]]:
        """Determines the model class for nested fields (dict values, list items)."""
        if not parent_model_class or not hasattr(parent_model_class, 'model_fields'):
            return None
        field_info = parent_model_class.model_fields.get(field_name)
        if not field_info or not field_info.annotation:
            return None

        annotation = field_info.annotation
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        nested_model = None
        if origin is dict and args and len(args) > 1:
             value_type = args[1]
             if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                  nested_model = value_type
        elif origin is list and args:
             item_type = args[0]
             if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                  nested_model = item_type
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
             nested_model = annotation # Direct nested model

        return nested_model


    def _dict_to_commented_map(self, data: Union[Dict, List, Any], model_class: Optional[Type[BaseModel]]) -> Any:
        """Recursively converts dicts to CommentedMaps and adds comments from Pydantic V2 schema."""
        if not RUAMEL_YAML_AVAILABLE: return data # Safety check

        if isinstance(data, dict):
            commented_map = CommentedMap()
            for key, value in data.items():
                field_description = self._get_field_description(model_class, key)
                nested_model_class = self._get_nested_model_class(model_class, key)

                # Represent multiline strings nicely
                if isinstance(value, str) and '\n' in value:
                    processed_value = PreservedScalarString(value)
                else:
                    processed_value = self._dict_to_commented_map(value, nested_model_class)

                commented_map[key] = processed_value
                if field_description:
                    cleaned_comment = "\n".join(f"# {line.strip()}" for line in field_description.strip().splitlines())#.replace('\n', '\n# ')
                    commented_map.yaml_set_comment_before_after_key(key, before=cleaned_comment, indent=0)

            return commented_map
        elif isinstance(data, list):
            commented_seq = CommentedSeq()
            # Assume list items share the nested model type derived from the parent field
            nested_model_class = model_class # Pass potential item model type
            for item in data:
                commented_seq.append(self._dict_to_commented_map(item, nested_model_class))
            return commented_seq
        else:
             # Scalar types
             return data
