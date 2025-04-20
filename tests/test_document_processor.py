import os
import pytest
from unittest.mock import patch, MagicMock, mock_open  # Fixed import

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.processor import DocumentProcessor
from semantic_qa_gen.document.models import Document, Section, SectionType
from semantic_qa_gen.utils.error import DocumentError


@pytest.fixture
def document_processor():
    """Create a document processor for testing"""
    config = ConfigManager()
    return DocumentProcessor(config)


@pytest.fixture
def mock_text_file():
    """Mock text file content"""
    return "# Sample Document\n\nThis is a paragraph with some content.\n\nAnother paragraph here."


def test_load_document_text_file(document_processor, mock_text_file):
    """Test loading a text document"""
    with patch("builtins.open", mock_open(read_data=mock_text_file)):
        with patch("os.path.exists", return_value=True):
            with patch.object(document_processor, "_get_loader_for_file") as mock_get_loader:
                # Create a mock loader that returns a predefined document
                mock_document = Document(content=mock_text_file, doc_type="text")
                mock_loader = MagicMock()
                mock_loader.load.return_value = mock_document
                mock_get_loader.return_value = mock_loader
                
                # Load the document
                document = document_processor.load_document("test.txt")
                
                # Assert the document was loaded and preprocessed
                assert document.content == mock_text_file
                mock_loader.load.assert_called_once()


def test_file_not_found(document_processor):
    """Test behavior when file is not found"""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(DocumentError) as excinfo:
            document_processor.load_document("nonexistent.txt")
        assert "Document file not found" in str(excinfo.value)


def test_no_loader_available(document_processor):
    """Test behavior when no suitable loader is available"""
    with patch("os.path.exists", return_value=True):
        with patch.object(document_processor, "_get_loader_for_file", return_value=None):
            with pytest.raises(DocumentError) as excinfo:
                document_processor.load_document("test.unsupported")
            assert "No loader available" in str(excinfo.value)


def test_normalize_whitespace(document_processor):
    """Test whitespace normalization"""
    text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
    normalized = document_processor._normalize_whitespace(text)
    assert normalized == "Line 1\n\nLine 2\n\nLine 3\n"


def test_extract_sections_text_document(document_processor):
    """Test extracting sections from a text document"""
    content = "# Main Title\n\nFirst paragraph.\n\n## Section 1\n\nContent here.\n\n## Section 2\n\nMore content."
    document = Document(content=content, doc_type="text")
    
    sections = document_processor.extract_sections(document)
    
    assert len(sections) > 0
    # Check if headings are correctly identified
    assert any(s.section_type == SectionType.HEADING and s.content == "# Main Title" for s in sections)


def test_preprocess_document(document_processor):
    """Test document preprocessing"""
    content = "Line 1\n\n\n\nLine 2\nLine 3"
    document = Document(content=content, doc_type="text")
    
    processed_doc = document_processor.preprocess_document(document)
    
    # Check that whitespace is normalized
    assert processed_doc.content != content
    assert "\n\n\n" not in processed_doc.content
