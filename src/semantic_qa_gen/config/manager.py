# filename: semantic_qa_gen/config/manager.py

"""Configuration management for SemanticQAGen."""

import os
import re
import json
import yaml  # Standard PyYAML for loading
import logging
from typing import Optional, Dict, Any, Union, List, Type
# Use Pydantic V2 imports
from pydantic import BaseModel, ValidationError, SecretStr
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
    PreservedScalarString = str  # Fallback type
    CommentedMap = dict  # Fallback type
    CommentedSeq = list  # Fallback type
    YAML = None  # Fallback type

from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.utils.error import ConfigurationError


class ConfigManager:
    """
    Manages the configuration for SemanticQAGen using Pydantic V2.

    Handles loading, validation, merging, defaults, and environment variables.
    Uses ruamel.yaml (optional) for comment preservation when saving.
    """
    ENV_PREFIX = "SEMANTICQAGEN_"
    SENSITIVE_KEY_PATTERNS = re.compile(
        r"(password|secret|token|api_key|apikey|private_key|access_key|authorization)",
        re.IGNORECASE
    )

    def __init__(self, config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        """Initializes the ConfigManager."""
        self._config: Optional[SemanticQAGenConfig] = None
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path

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
            initial_config_dict = deepcopy(config_dict)  # Use a copy
            load_source = "dictionary"
        else:
            # Get defaults using Pydantic V2's model_construct or default factory on instantiation
            initial_config_dict = SemanticQAGenConfig().model_dump(exclude_unset=False)  # Get all defaults
            load_source = "defaults"
            self.logger.info("No configuration source provided, using default settings.")

        if initial_config_dict is None:
            raise ConfigurationError("Failed to determine any configuration source.")

        # Process environment variable interpolation (${VAR})
        interpolated_config = self._process_env_vars_interpolation(initial_config_dict)
        # Apply environment variable overrides (SEMANTICQAGEN_SECTION_KEY=value)
        final_config_dict = self._apply_env_overrides(interpolated_config)

        # Convert empty-string placeholders ("") back to None so optional fields
        # like api_base, api_key, etc. don't trip Pydantic validation. This is
        # the inverse of _fill_none_placeholders, which we use when writing files.
        final_config_dict = self._strip_placeholders(final_config_dict)

        # Validate the final configuration dictionary using Pydantic V2
        try:
            self._config = SemanticQAGenConfig(**final_config_dict)
            self.logger.info(f"Configuration successfully loaded and validated from {load_source}.")
            config_dict_for_logging = self._sanitize_for_logging(self._config.model_dump(mode='python'))
            self.logger.debug(
                f"Final configuration (sanitized): "
                f"{json.dumps(config_dict_for_logging, indent=2, default=str)}"
            )
        except ValidationError as e:
            error_msg = self._format_validation_error(e)
            raise ConfigurationError(f"Invalid configuration from {load_source}:\n{error_msg}")
        except Exception as e:  # Catch other potential model creation errors
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
                    raise ConfigurationError(
                        f"Config file {config_path} root is not a YAML dictionary (type: {type(config_dict)}).")
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
                    return match.group(0)  # Keep original ${VAR} string

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
                if raw_env_value.lower() == 'true':
                    typed_value = True
                elif raw_env_value.lower() == 'false':
                    typed_value = False
                elif '.' in raw_env_value:
                    typed_value = float(raw_env_value)
                else:
                    typed_value = int(raw_env_value)
            except ValueError:
                typed_value = raw_env_value  # Keep as string

            current_level = overridden_config
            try:
                # Navigate or create path in the config dict
                for i, part in enumerate(config_path_parts[:-1]):
                    if part not in current_level or not isinstance(current_level.get(part), dict):
                        # If path intermediate doesn't exist or is not dict, create it
                        current_level[part] = {}
                    current_level = current_level[part]
                final_key = config_path_parts[-1]
                current_level[final_key] = typed_value  # Set the value

                # SECURITY FIX: Mask sensitive values in log output
                is_sensitive_override = any(self.SENSITIVE_KEY_PATTERNS.search(part) for part in config_path_parts)
                log_value = "***REDACTED***" if is_sensitive_override else typed_value

                self.logger.debug(
                    f"Applying environment override: {'_'.join(config_path_parts)} = {log_value} (type: {type(typed_value).__name__})")
                overrides_applied += 1
            except Exception as e:
                self.logger.error(f"Error applying env override {env_key}=[REDACTED]: {e}")

        if overrides_applied > 0:
            self.logger.info(f"Applied {overrides_applied} environment variable override(s).")
        return overridden_config

    def _format_validation_error(self, error: ValidationError) -> str:
        """Formats Pydantic V2 validation errors."""
        error_lines = []
        for err in error.errors():
            loc = " -> ".join(map(str, err['loc']))
            msg = err['msg']

            # SECURITY FIX: Mask input if the location corresponds to a sensitive field
            is_sensitive_error = any(self.SENSITIVE_KEY_PATTERNS.search(str(part)) for part in err.get('loc', []))
            if is_sensitive_error:
                inp_str = " (Input: ***REDACTED***)"
            else:
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
        """Saves the *current* config state to YAML file.

        Keeps None-valued keys in the output (as empty strings) so users can see
        every option that's available to fill in. Injects empty `local`/`remote`
        LLM-service stubs into the OUTPUT (not the model) when missing, so users
        always have those sections visible as templates.
        """
        if self._config is None:
            raise ConfigurationError("Cannot save: Configuration not loaded.")

        # exclude_none=False so optional fields (api_key, api_base, custom_headers,
        # organization, api_version, max_tokens, ...) all appear in the file.
        config_dict = self.config.model_dump(mode='python', exclude_none=False)

        # Ensure llm_services.local / llm_services.remote stubs are present in the
        # dict (does NOT touch self._config -> no re-validation).
        self._ensure_optional_sections_in_dict(config_dict)

        # Convert any remaining None -> "" so the YAML reads naturally.
        config_dict = self._fill_none_placeholders(config_dict)

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
        """Creates a default configuration YAML file.

        All schema fields appear in the output (including ones that default to
        None), rendered as empty strings so users have a complete template to
        fill in.
        """
        try:
            default_config_instance = SemanticQAGenConfig()

            # exclude_none=False so we keep every field; placeholders make it readable.
            default_config_dict = default_config_instance.model_dump(
                mode='python', exclude_unset=False, exclude_none=False
            )

            # Inject empty local/remote stubs into the dict.
            self._ensure_optional_sections_in_dict(default_config_dict)

            # Convert None -> "" for readability.
            default_config_dict = self._fill_none_placeholders(default_config_dict)

            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            if include_comments and RUAMEL_YAML_AVAILABLE and YAML is not None:
                yaml_dumper = YAML()
                yaml_dumper.indent(mapping=2, sequence=4, offset=2)
                yaml_dumper.preserve_quotes = True
                commented_defaults = self._dict_to_commented_map(default_config_dict, SemanticQAGenConfig)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write("# SemanticQAGen Default Configuration\n")
                    file.write(
                        "# Modify values as needed. Env vars can override settings "
                        "(e.g., SEMANTICQAGEN_LOCAL_ENABLED=true).\n"
                    )
                    file.write(
                        "# Empty strings (\"\") mark optional fields you can fill in.\n\n"
                    )
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

    def _get_nested_model_class(self, parent_model_class: Type[BaseModel], field_name: str) -> Optional[
        Type[BaseModel]]:
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
            nested_model = annotation  # Direct nested model

        return nested_model

    def _dict_to_commented_map(self, data: Union[Dict, List, Any],
                               model_class: Optional[Type[BaseModel]]) -> Any:
        """Recursively converts dicts to CommentedMaps and attaches schema descriptions
        as YAML comments. Section descriptions are placed above their key; scalar
        descriptions are placed end-of-line."""
        if not RUAMEL_YAML_AVAILABLE:
            return data  # Safety check

        # Handle SecretStr objects explicitly to prevent crashes and mask in files
        if isinstance(data, SecretStr):
            return "**********"  # Mask secrets when writing to file
        elif isinstance(data, dict):
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
                    desc = field_description.strip()
                    if isinstance(value, dict):
                        # Section header — multi-line comment ABOVE the key.
                        cleaned_comment = "\n".join(
                            f"# {line.strip()}" for line in desc.splitlines()
                        )
                        commented_map.yaml_set_comment_before_after_key(
                            key, before=cleaned_comment, indent=0
                        )
                    else:
                        # Scalar / list leaf — collapse to one line and place end-of-line.
                        single_line = " ".join(desc.split())
                        try:
                            commented_map.yaml_add_eol_comment(single_line, key)
                        except Exception:
                            # Fall back to "before" placement if eol fails for any reason.
                            commented_map.yaml_set_comment_before_after_key(
                                key, before=f"# {single_line}", indent=0
                            )

            return commented_map
        elif isinstance(data, list):
            commented_seq = CommentedSeq()
            nested_model_class = model_class
            for item in data:
                commented_seq.append(self._dict_to_commented_map(item, nested_model_class))
            return commented_seq
        else:
            return data

    def _fill_none_placeholders(self, data: Any) -> Any:
        """Recursively replace None with empty string for readable YAML output.

        Empty strings render as `key: ""` which is friendlier than `key: null`
        or omitting the key entirely.
        """
        if isinstance(data, dict):
            return {k: self._fill_none_placeholders(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._fill_none_placeholders(v) for v in data]
        if data is None:
            return ""
        return data

    def _ensure_optional_sections_in_dict(self, config_dict: Dict[str, Any]) -> None:
        """Inject placeholder stubs for llm_services.local / llm_services.remote
        if missing from the dict.

        Operates ONLY on the output dict — never on the validated Pydantic model.
        This avoids re-triggering LLMServiceConfig.validate_llm_config, which
        forbids the state "both services defined but both disabled".
        """
        llm = config_dict.setdefault("llm_services", {})
        # If llm came back as None (it was Optional and unset), make it a dict.
        if llm is None:
            config_dict["llm_services"] = llm = {}

        if llm.get("remote") is None:
            llm["remote"] = self._get_optional_section_defaults("remote")
        if llm.get("local") is None:
            llm["local"] = self._get_optional_section_defaults("local")

    def _get_optional_section_defaults(self, which: str) -> Dict[str, Any]:
        """Return a plain-dict template for an LLM service section.

        Built by instantiating the schema model with `enabled=False` and dumping
        it, so we automatically pick up any new fields added to the schema later.
        Falls back to a small hardcoded skeleton if instantiation fails for any
        reason (e.g., schema added required fields).
        """
        from semantic_qa_gen.config.schema import LocalServiceConfig, RemoteServiceConfig

        model_cls = RemoteServiceConfig if which == "remote" else LocalServiceConfig
        try:
            instance = model_cls(enabled=False)
            return instance.model_dump(mode='python', exclude_none=False)
        except Exception as e:
            self.logger.debug(
                f"Could not instantiate {model_cls.__name__} for template; "
                f"using minimal skeleton. ({e})"
            )
            if which == "remote":
                return {
                    "enabled": False,
                    "provider": "openrouter",
                    "model": "gpt-4o",
                    "api_key": None,
                    "api_base": None,
                    "api_version": None,
                    "organization": None,
                    "timeout": 120,
                    "max_retries": 3,
                    "initial_delay": 1.0,
                    "max_delay": 60.0,
                    "custom_headers": None,
                    "preferred_tasks": ["analysis", "generation"],
                }
            else:  # local
                return {
                    "enabled": False,
                    "model": "mistral:7b",
                    "url": "http://localhost:11434",
                    "api_key": None,
                    "timeout": 120,
                    "max_retries": 3,
                    "initial_delay": 1.0,
                    "max_delay": 60.0,
                    "custom_headers": None,
                    "preferred_tasks": ["validation"],
                }

    def _strip_placeholders(self, data: Any) -> Any:
        """Inverse of _fill_none_placeholders: convert empty strings back to None.

        Run on freshly-loaded YAML (or env-interpolated) dicts before Pydantic
        validation, so that fields written as `api_base: ""` don't crash HttpUrl
        validation. Legitimate empty strings in config are vanishingly rare; for
        those few fields, Pydantic will re-apply the default after we strip.
        """
        if isinstance(data, dict):
            return {k: self._strip_placeholders(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._strip_placeholders(v) for v in data]
        if isinstance(data, str) and data == "":
            return None
        return data

    def _sanitize_for_logging(self, data: Any) -> Any:
        """
        Recursively sanitizes a dictionary for logging.
        1. Masks values for keys that look like secrets.
        2. Unwraps Pydantic SecretStr objects.
        """
        if isinstance(data, dict):
            sanitized_dict = {}
            for k, v in data.items():
                # Check if the key indicates a secret (covers custom_headers['Authorization'])
                if isinstance(k, str) and self.SENSITIVE_KEY_PATTERNS.search(k):
                    sanitized_dict[k] = "***REDACTED***"
                else:
                    sanitized_dict[k] = self._sanitize_for_logging(v)
            return sanitized_dict
        elif isinstance(data, list):
            return [self._sanitize_for_logging(item) for item in data]
        elif isinstance(data, SecretStr):
            # Explicitly mask SecretStr objects when converting dict to JSON log
            return "***REDACTED***"
        else:
            # Return primitive types as-is
            return data