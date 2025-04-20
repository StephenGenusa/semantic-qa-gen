import pytest
from unittest.mock import MagicMock

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.chunking.engine import ChunkingEngine
from semantic_qa_gen.document.models import Document, Section, SectionType, DocumentType
from semantic_qa_gen.utils.error import ChunkingError
from semantic_qa_gen.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from semantic_qa_gen.chunking.strategies.semantic import SemanticChunkingStrategy


@pytest.fixture
def config_manager():
    """Create a config manager for chunking tests"""
    return ConfigManager()


@pytest.fixture
def chunking_engine(config_manager):
    """Create a chunking engine for testing"""
    return ChunkingEngine(config_manager)


@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    content = """# Main Heading

This is an introduction paragraph with some information that will be useful for testing.

## Section 1

Content for section 1 that provides details about something important.

## Section 2

More content that will be chunked appropriately based on the strategy.
This part has more text to make the chunks more realistic."""

    return Document(
        content=content,
        doc_type=DocumentType.TEXT,
        path="sample.txt"
    )


@pytest.fixture
def sample_sections():
    """Create sample sections for testing"""
    return [
        Section(content="Main Heading", section_type=SectionType.HEADING, level=1),
        Section(content="This is an introduction paragraph with some information that will be useful for testing.",
               section_type=SectionType.PARAGRAPH, level=0),
        Section(content="Section 1", section_type=SectionType.HEADING, level=2),
        Section(content="Content for section 1 that provides details about something important.",
               section_type=SectionType.PARAGRAPH, level=0),
        Section(content="Section 2", section_type=SectionType.HEADING, level=2),
        Section(content="More content that will be chunked appropriately based on the strategy. "
                "This part has more text to make the chunks more realistic.",
               section_type=SectionType.PARAGRAPH, level=0),
    ]


def test_chunking_engine_initialization(chunking_engine):
    """Test chunking engine initialization"""
    assert chunking_engine.active_strategy is not None
    assert isinstance(chunking_engine.strategies["fixed_size"], FixedSizeChunkingStrategy)
    assert isinstance(chunking_engine.strategies["semantic"], SemanticChunkingStrategy)


def test_set_strategy(chunking_engine):
    """Test setting chunking strategy"""
    # Initially should be 'semantic' based on default config
    assert chunking_engine.config.strategy == "semantic"
    
    # Change to fixed_size
    chunking_engine.set_strategy("fixed_size")
    assert isinstance(chunking_engine.active_strategy, FixedSizeChunkingStrategy)
    
    # Test strategy that doesn't exist
    with pytest.raises(ChunkingError):
        chunking_engine.set_strategy("nonexistent_strategy")


def test_chunk_document(chunking_engine, sample_document, sample_sections):
    """Test chunking a document"""
    document_processor = MagicMock()
    document_processor.extract_sections.return_value = sample_sections
    
    # Mock the active strategy to return predefined chunks
    chunking_engine.active_strategy = MagicMock()
    chunking_engine.active_strategy.chunk_document.return_value = [
        MagicMock(content="Chunk 1", sequence=0),
        MagicMock(content="Chunk 2", sequence=1),
    ]
    
    # Process chunking
    chunks = chunking_engine.chunk_document(sample_document, document_processor)
    
    # Assertions
    document_processor.extract_sections.assert_called_once_with(sample_document)
    chunking_engine.active_strategy.chunk_document.assert_called_once()
    assert len(chunks) == 2


def test_optimize_chunks(chunking_engine):
    """Test chunk optimization"""
    # Create mock chunks, some small enough to be merged
    chunks = [
        MagicMock(content="A" * 400, id="1", sequence=0, preceding_headings=[]),
        MagicMock(content="B" * 300, id="2", sequence=1, preceding_headings=[]),
        MagicMock(content="C" * 1500, id="3", sequence=2, preceding_headings=[]),
    ]
    
    # Set small chunk threshold
    chunking_engine.config.min_chunk_size = 500
    
    # Optimize chunks
    optimized = chunking_engine.optimize_chunks(chunks)
    
    # First two should be merged
    assert len(optimized) == 2
    assert "A" * 400 in optimized[0].content
    assert "B" * 300 in optimized[0].content


def test_fixed_size_chunking():
    """Test the fixed size chunking strategy"""
    config = {"target_chunk_size": 500, "overlap_size": 50}
    strategy = FixedSizeChunkingStrategy(config)
    
    document = Document(
        content="A" * 1500,
        doc_type=DocumentType.TEXT
    )
    
    sections = [
        Section(content="A" * 1500, section_type=SectionType.PARAGRAPH, level=0)
    ]
    
    chunks = strategy.chunk_document(document, sections)
    
    # Should result in multiple chunks
    assert len(chunks) > 1
    # Check for overlap
    if len(chunks) >= 2:
        assert chunks[0].content[-50:] in chunks[1].content


def test_semantic_chunking(sample_document, sample_sections):
    """Test the semantic chunking strategy"""
    config = {"target_chunk_size": 500, "preserve_headings": True}
    strategy = SemanticChunkingStrategy(config)
    
    chunks = strategy.chunk_document(sample_document, sample_sections)
    
    # Should produce at least one chunk
    assert len(chunks) >= 1
    # Headings should be preserved
    assert any("Main Heading" in chunk.content for chunk in chunks)
