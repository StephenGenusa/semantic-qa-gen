import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from semantic_qa_gen.semantic_qa_gen import SemanticQAGen
from semantic_qa_gen.utils.error import SemanticQAGenError


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing"""
    pipeline = MagicMock()
    pipeline.process_document = AsyncMock()
    return pipeline


def test_semanticqagen_init():
    """Test SemanticQAGen initialization"""
    # Test with default config
    qa_gen = SemanticQAGen()
    assert qa_gen.pipeline is not None
    
    # Test with verbose flag
    with patch("logging.basicConfig") as mock_log_config:
        qa_gen = SemanticQAGen(verbose=True)
        mock_log_config.assert_called_once()
        assert "DEBUG" in str(mock_log_config.call_args)


def test_init_with_config():
    """Test initialization with config dictionary"""
    config = {"output": {"format": "csv"}}
    qa_gen = SemanticQAGen(config_dict=config)
    assert qa_gen.config.output.format == "csv"


def test_init_with_invalid_config():
    """Test initialization with invalid configuration"""
    with pytest.raises(SemanticQAGenError):
        SemanticQAGen(config_dict={"invalid": {"nonexistent": True}})


def test_process_document_file_not_found():
    """Test behavior when document file doesn't exist"""
    qa_gen = SemanticQAGen()
    
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            qa_gen.process_document("nonexistent.txt")


def test_process_document(mock_pipeline):
    """Test document processing"""
    # Set up mock result
    mock_result = {
        "document": {"id": "doc-1", "title": "Test Document"},
        "questions": [{"id": "q1", "text": "Question 1", "answer": "Answer 1"}],
        "statistics": {"total_questions": 1}
    }
    mock_pipeline.process_document.return_value = mock_result
    
    # Create SemanticQAGen with mock pipeline
    with patch("semantic_qa_gen.semantic_qa_gen.SemanticPipeline", return_value=mock_pipeline):
        with patch("os.path.exists", return_value=True):
            qa_gen = SemanticQAGen()
            result = qa_gen.process_document("test.txt")
            
            # Assertions
            assert result == mock_result
            mock_pipeline.process_document.assert_called_once_with("test.txt")


def test_save_questions():
    """Test saving questions to file"""
    qa_gen = SemanticQAGen()
    
    # Mock output formatter
    mock_formatter = MagicMock()
    mock_formatter.format_and_save.return_value = "/path/to/output.json"
    qa_gen.pipeline.output_formatter = mock_formatter
    
    # Test data
    result = {
        "questions": [{"id": "q1", "text": "Question?", "answer": "Answer"}],
        "document": {"title": "Test Document"},
        "statistics": {"total_questions": 1}
    }
    
    # Call save_questions
    output_path = qa_gen.save_questions(result, "output.json")
    
    # Assertions
    assert output_path == "/path/to/output.json"
    mock_formatter.format_and_save.assert_called_once()
    args = mock_formatter.format_and_save.call_args[1]
    assert args["output_path"] == "output.json"
    assert args["questions"] == result["questions"]


def test_create_default_config_file():
    """Test creating default config file"""
    qa_gen = SemanticQAGen()
    
    with patch.object(qa_gen.config_manager, "save_config") as mock_save:
        qa_gen.create_default_config_file("config.yaml")
        mock_save.assert_called_once_with("config.yaml")
