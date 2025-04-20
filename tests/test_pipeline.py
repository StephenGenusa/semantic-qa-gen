import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.pipeline.semantic import SemanticPipeline
from semantic_qa_gen.document.models import Document, Chunk, AnalysisResult
from semantic_qa_gen.utils.error import SemanticQAGenError


@pytest.fixture
def config_manager():
    """Create a config manager for testing"""
    return ConfigManager()


@pytest.fixture
def semantic_pipeline(config_manager):
    """Create a semantic pipeline for testing"""
    # Create pipeline with mocked components
    pipeline = SemanticPipeline(config_manager)
    
    # Mock components
    pipeline.document_processor = MagicMock()
    pipeline.chunking_engine = MagicMock()
    pipeline.semantic_analyzer = MagicMock()
    pipeline.question_processor = MagicMock()
    pipeline.task_router = MagicMock()
    pipeline.progress_reporter = MagicMock()
    
    return pipeline


@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return Document(
        content="This is sample content for testing the pipeline.",
        doc_type="text",
        path="sample.txt"
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    return [
        Chunk(
            content="Chunk 1 content",
            id="chunk-1",
            document_id="doc-1",
            sequence=0,
            context={"title": "Test Document"}
        ),
        Chunk(
            content="Chunk 2 content",
            id="chunk-2",
            document_id="doc-1",
            sequence=1,
            context={"title": "Test Document"}
        )
    ]


@pytest.mark.asyncio
async def test_process_document(semantic_pipeline, sample_document, sample_chunks):
    """Test the complete document processing pipeline"""
    # Set up mock document processor
    semantic_pipeline.document_processor.load_document.return_value = sample_document
    
    # Set up mock chunking engine
    semantic_pipeline.chunking_engine.chunk_document.return_value = sample_chunks
    
    # Set up mock semantic analyzer
    analyses = {
        "chunk-1": AnalysisResult(
            chunk_id="chunk-1",
            information_density=0.8,
            topic_coherence=0.7,
            complexity=0.6,
            estimated_question_yield={"factual": 3, "inferential": 2, "conceptual": 1},
            key_concepts=["concept1", "concept2"]
        ),
        "chunk-2": AnalysisResult(
            chunk_id="chunk-2",
            information_density=0.7,
            topic_coherence=0.6,
            complexity=0.5,
            estimated_question_yield={"factual": 2, "inferential": 1, "conceptual": 1},
            key_concepts=["concept3", "concept4"]
        )
    }
    semantic_pipeline.semantic_analyzer.analyze_chunks.return_value = analyses
    
    # Set up mock question processor
    mock_questions = {
        "chunk-1": [MagicMock(id="q1"), MagicMock(id="q2")],
        "chunk-2": [MagicMock(id="q3"), MagicMock(id="q4")]
    }
    mock_stats = {
        "total_generated_questions": 4,
        "total_valid_questions": 4,
        "categories": {"factual": 2, "inferential": 1, "conceptual": 1}
    }
    semantic_pipeline.question_processor.process_chunks.return_value = (mock_questions, mock_stats)
    
    # Process document
    result = await semantic_pipeline.process_document("sample.txt")
    
    # Assertions
    semantic_pipeline.document_processor.load_document.assert_called_once_with("sample.txt")
    semantic_pipeline.chunking_engine.chunk_document.assert_called_once()
    semantic_pipeline.semantic_analyzer.analyze_chunks.assert_called_once_with(sample_chunks, semantic_pipeline.progress_reporter)
    semantic_pipeline.question_processor.process_chunks.assert_called_once()
    
    # Check result structure
    assert "document" in result
    assert "questions" in result
    assert "statistics" in result
    assert len(result["questions"]) == 4  # Total questions across chunks


@pytest.mark.asyncio
async def test_process_document_with_errors(semantic_pipeline):
    """Test error handling in process_document"""
    # Make document processor raise an exception
    semantic_pipeline.document_processor.load_document.side_effect = Exception("Document loading error")
    
    # Should raise SemanticQAGenError
    with pytest.raises(SemanticQAGenError):
        await semantic_pipeline.process_document("sample.txt")


@pytest.mark.asyncio
async def test_checkpoint_functionality(semantic_pipeline, sample_document, sample_chunks):
    """Test checkpoint functionality"""
    # Set up checkpoint manager mock
    semantic_pipeline.checkpoint_manager = MagicMock()
    semantic_pipeline.checkpoint_manager.load_checkpoint.return_value = {
        "current_chunk_idx": 1,
        "completed_chunks": ["chunk-1"]
    }
    
    # Set up other mocks
    semantic_pipeline.document_processor.load_document.return_value = sample_document
    semantic_pipeline.chunking_engine.chunk_document.return_value = sample_chunks
    semantic_pipeline.semantic_analyzer.analyze_chunks.return_value = {}
    semantic_pipeline.question_processor.process_chunks.return_value = ({}, {})
    
    # Process document
    await semantic_pipeline.process_document("sample.txt")
    
    # Check that checkpoint was loaded
    semantic_pipeline.checkpoint_manager.load_checkpoint.assert_called_once_with(sample_document)
