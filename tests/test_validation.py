import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, Question
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.question.validation.base import ValidationResult
from semantic_qa_gen.question.validation.diversity import DiversityValidator
from semantic_qa_gen.question.validation.factual import FactualAccuracyValidator


@pytest.fixture
def config_manager():
    """Create a config manager for testing"""
    return ConfigManager()


@pytest.fixture
def mock_task_router():
    """Create a mock task router"""
    router = MagicMock()
    # Set up common methods with proper AsyncMock
    router.validate_question = AsyncMock()
    router.get_service = MagicMock()

    # Configure get_service to return a properly configured mock service
    mock_service = MagicMock()
    mock_service.validate_question = AsyncMock()
    router.get_service.return_value = mock_service

    return router


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager"""
    return MagicMock()


@pytest.fixture
def validation_engine(config_manager, mock_task_router, mock_prompt_manager):
    """Create a validation engine for testing"""
    engine = ValidationEngine(config_manager, mock_task_router, mock_prompt_manager)
    # Clear validators and add test ones
    engine.validators = {}
    return engine


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing"""
    return Chunk(
        content="This is sample content for testing validation.",
        id="chunk-123",
        document_id="doc-456",
        sequence=1,
        context={"title": "Test Document"}
    )


@pytest.fixture
def sample_questions():
    """Create sample questions for testing"""
    return [
        Question(
            id="q1",
            text="What is concept A?",
            answer="Concept A is...",
            chunk_id="chunk-123",
            category="factual",
            metadata={}
        ),
        Question(
            id="q2",
            text="How do concepts A and B relate?",
            answer="They relate through...",
            chunk_id="chunk-123",
            category="inferential",
            metadata={}
        ),
        Question(
            id="q3", 
            text="What is concept A?",  # Duplicate of q1
            answer="A different answer...",
            chunk_id="chunk-123",
            category="factual",
            metadata={}
        )
    ]


@pytest.mark.asyncio
async def test_diversity_validator(validation_engine, sample_chunk, sample_questions):
    """Test the diversity validator"""
    # Add diversity validator to engine
    validator = DiversityValidator({"min_similarity_threshold": 0.8, "enabled": True})
    validation_engine.validators["diversity"] = validator
    
    # First question should pass
    result1 = await validator.validate(sample_questions[0], sample_chunk)
    assert result1.is_valid is True
    
    # Second question (different) should pass
    result2 = await validator.validate(sample_questions[1], sample_chunk)
    assert result2.is_valid is True
    
    # Third question (similar to first) should fail
    result3 = await validator.validate(sample_questions[2], sample_chunk)
    assert result3.is_valid is False
    assert "too similar" in result3.reasons[0].lower()


@pytest.mark.asyncio
async def test_factual_accuracy_validator(validation_engine, mock_task_router, sample_chunk, sample_questions):
    """Test the factual accuracy validator"""
    # Set up mock service response
    mock_service = MagicMock()
    mock_service.validate_question = AsyncMock(return_value={
        "factual_accuracy": 0.9,
        "answer_completeness": 0.8,
        "question_clarity": 0.7,
        "reasons": ["Good factual accuracy"]
    })
    mock_task_router.get_service.return_value = mock_service

    # Add factual validator to engine
    validator = FactualAccuracyValidator(mock_task_router,
                                         validation_engine.prompt_manager,
                                         {"threshold": 0.8, "enabled": True})
    validation_engine.validators["factual"] = validator

    # Validate question
    result = await validator.validate(sample_questions[0], sample_chunk)

    # Assertions
    assert result.is_valid is True
    assert "factual_accuracy" in result.scores
    assert result.scores["factual_accuracy"] == 0.9
    mock_service.validate_question.assert_called_once_with(sample_questions[0], sample_chunk)

    # Test with low score that should fail
    mock_service.validate_question = AsyncMock(return_value={
        "factual_accuracy": 0.5,  # Below threshold
        "answer_completeness": 0.8,
        "question_clarity": 0.7,
        "reasons": ["Poor factual accuracy"]
    })

    result = await validator.validate(sample_questions[0], sample_chunk)
    assert result.is_valid is False

@pytest.mark.asyncio
async def test_validate_questions(validation_engine, sample_chunk, sample_questions):
    """Test validating multiple questions"""
    # Set up mock validator
    mock_validator = MagicMock()
    mock_validator.is_enabled.return_value = True
    mock_validator.validate = AsyncMock()
    mock_validator.validate.side_effect = [
        ValidationResult(
            question_id="q1",
            is_valid=True,
            scores={"test": 0.9},
            reasons=["Valid question"]
        ),
        ValidationResult(
            question_id="q2",
            is_valid=False,
            scores={"test": 0.5},
            reasons=["Invalid question"]
        ),
        ValidationResult(
            question_id="q3",
            is_valid=True,
            scores={"test": 0.8},
            reasons=["Valid question"]
        )
    ]
    
    # Add test validator
    validation_engine.validators["test"] = mock_validator
    
    # Validate all questions
    results = await validation_engine.validate_questions(sample_questions, sample_chunk)
    
    # Assertions
    assert len(results) == 3
    assert results["q1"]["is_valid"] is True
    assert results["q2"]["is_valid"] is False
    assert results["q3"]["is_valid"] is True
    
    # Test get_valid_questions
    valid_questions = validation_engine.get_valid_questions(sample_questions, results)
    assert len(valid_questions) == 2
    assert valid_questions[0].id == "q1"
    assert valid_questions[1].id == "q3"


@pytest.mark.asyncio
async def test_validation_error_handling(validation_engine, sample_chunk, sample_questions):
    """Test error handling in validation"""
    # Set up validator that raises an exception
    mock_validator = MagicMock()
    mock_validator.is_enabled.return_value = True
    mock_validator.validate = AsyncMock(side_effect=Exception("Validation error"))
    
    # Add test validator
    validation_engine.validators["error_test"] = mock_validator
    
    # Validate questions - should handle the error
    results = await validation_engine.validate_questions([sample_questions[0]], sample_chunk)
    
    # Should return a failed validation result
    assert results[sample_questions[0].id]["is_valid"] is False
    assert "error" in results[sample_questions[0].id]["reasons"][0].lower()

