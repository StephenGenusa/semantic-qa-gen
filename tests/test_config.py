import os
import pytest
from unittest.mock import patch, mock_open
import yaml

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.utils.error import ConfigurationError


def test_config_manager_init_default():
    """Test initializing ConfigManager with default values"""
    config = ConfigManager()
    assert isinstance(config.config, SemanticQAGenConfig)
    assert config.config.version == "1.0"


def test_config_manager_from_dict():
    """Test initializing ConfigManager from dictionary"""
    config_dict = {"version": "1.1", "output": {"format": "json", "json_indent": 4}}
    config = ConfigManager(config_dict=config_dict)
    assert config.config.version == "1.1"
    assert config.config.output.json_indent == 4


def test_config_manager_from_file():
    """Test loading configuration from a YAML file"""
    yaml_content = """
    version: "1.1"
    output:
      format: json
      json_indent: 4
    """
    with patch("builtins.open", mock_open(read_data=yaml_content)) as m:
        with patch("os.path.exists", return_value=True):
            config = ConfigManager(config_path="config.yaml")
            assert config.config.version == "1.1"
            assert config.config.output.json_indent == 4
            m.assert_called_once_with("config.yaml", 'r')


def test_config_manager_file_not_found():
    """Test behavior when config file doesn't exist"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(ConfigurationError) as excinfo:
            ConfigManager(config_path="nonexistent.yaml")
        assert "Configuration file not found" in str(excinfo.value)


def test_config_manager_invalid_yaml():
    """Test behavior with invalid YAML"""
    with patch("builtins.open", mock_open(read_data=": invalid: yaml: content")) as m:
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ConfigurationError) as excinfo:
                ConfigManager(config_path="invalid.yaml")
            assert "Failed to parse YAML" in str(excinfo.value)


def test_environment_variable_interpolation():
    """Test environment variable interpolation in configuration"""
    config_dict = {
        "llm_services": {
            "remote": {
                "api_key": "${TEST_API_KEY}"
            }
        }
    }
    
    with patch.dict(os.environ, {"TEST_API_KEY": "test-key-value"}):
        config = ConfigManager(config_dict=config_dict)
        assert config.get_section("llm_services").remote.api_key == "test-key-value"


def test_get_section():
    """Test retrieving specific sections from configuration"""
    config = ConfigManager()
    output_section = config.get_section("output")
    assert output_section.format == "json"
    
    with pytest.raises(ConfigurationError):
        config.get_section("nonexistent_section")


def test_save_config():
    """Test saving configuration to a file"""
    config = ConfigManager()
    
    with patch("builtins.open", mock_open()) as m:
        config.save_config("config_out.yaml")
        m.assert_called_once_with("config_out.yaml", 'w')
        handle = m()
        assert handle.write.called


def test_validation_constraints():
    """Test configuration validation constraints"""
    # Test invalid chunking strategy
    with pytest.raises(ValueError):
        ConfigManager(config_dict={"chunking": {"strategy": "invalid_strategy"}})

    # Test invalid chunk size relationships
    with pytest.raises(ValueError):
        ConfigManager(config_dict={"chunking": {
            "min_chunk_size": 2000, 
            "target_chunk_size": 1500
        }})
    
    # Test LLM services validation
    with pytest.raises(ValueError):
        ConfigManager(config_dict={"llm_services": {
            "local": {"enabled": False},
            "remote": {"enabled": False}
        }})


def test_config_schema_defaults():
    """Test that schema defaults are applied correctly"""
    config = ConfigManager()
    assert config.config.chunking.strategy == "semantic"
    assert config.config.processing.concurrency == 3
    assert config.config.question_generation.max_questions_per_chunk == 10
