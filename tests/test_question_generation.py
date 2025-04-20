import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.utils.error import ValidationError


@pytest.fixture
def config_manager():
    """Create a config manager for testing"""
    return ConfigManager()


@pytest.fixture
def mock_task_router():
    """Create a mock task router"""
    router = MagicMock()
    router.generate_questions = AsyncMock()
    return router


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager"""
    return MagicMock()


@pytest.fixture
def question_generator(config_manager, mock_task_router, mock_prompt_manager):
    """Create a question generator for testing"""
    return QuestionGenerator(config_manager, mock_task_router, mock_prompt_manager)


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing"""
    return Chunk(
        content="This is sample content for testing question generation.",
        id="chunk-123",
        document_id="doc-456",
        sequence=1,
        context={"title": "Test Document"}
    )


@pytest.fixture
def sample_analysis():
    """Create a sample analysis for testing"""
    return AnalysisResult(
        chunk_id="chunk-123",
        information_density=0.8,
        topic_coherence=0.7,
        complexity=0.6,
        estimated_question_yield={
            "factual": 3,
            "inferential": 2,
            "conceptual": 1
        },
        key_concepts=["concept1", "concept2"]
    )


@pytest.mark.asyncio
async def test_generate_questions(question_generator, sample_chunk, sample_analysis):
    """Test generating questions"""
    # Set up mock return value
    mock_questions = [
        Question(
            id="q1",
            text="What is the first concept?",
            answer="It's concept1",
            chunk_id="chunk-123",
            category="factual",
            metadata={}
        ),
        Question(
            id="q2",
            text="How do the concepts relate?",
            answer="They connect through...",
            chunk_id="chunk-123",
            category="inferential",
            metadata={}
        )
    ]
    
    question_generator.task_router.generate_questions.return_value = mock_questions
    
    # Call the function
    result = await question_generator.generate_questions(sample_chunk, sample_analysis)
    
    # Assertions
    assert result == mock_questions
    question_generator.task_router.generate_questions.assert_called_once()
    # Check that category counts were calculated
    args, kwargs = question_generator.task_router.generate_questions.call_args
    assert kwargs["chunk"] == sample_chunk
    assert kwargs["analysis"] == sample_analysis
    assert "categories" in kwargs


@pytest.mark.asyncio
async def test_generate_questions_failure(question_generator, sample_chunk, sample_analysis):
    """Test behavior when question generation fails"""
    # Make the task router raise an exception
    question_generator.task_router.generate_questions.side_effect = Exception("API error")
    
    # Call should raise ValidationError
    with pytest.raises(ValidationError):
        await question_generator.generate_questions(sample_chunk, sample_analysis)


def test_calculate_question_counts(question_generator, sample_analysis):
    """Test calculating question counts for different categories"""
    counts = question_generator._calculate_question_counts(sample_analysis)
    
    # Verify categories are present
    assert "factual" in counts
    assert "inferential" in counts
    assert "conceptual" in counts
    
    # Test adaptive generation
    question_generator.config.adaptive_generation = True
    adapted_counts = question_generator._calculate_question_counts(sample_analysis)
    
    # Information density should influence count
    assert adapted_counts["factual"] >= counts["factual"]
    
    # Test maximum questions limit
    question_generator.config.max_questions_per_chunk = 3
    limited_counts = question_generator._calculate_question_counts(sample_analysis)
    assert sum(limited_counts.values()) <= 3
