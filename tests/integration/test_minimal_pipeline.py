import pytest
import os
import tempfile
from unittest.mock import patch, AsyncMock

from semantic_qa_gen.semantic_qa_gen import SemanticQAGen
from semantic_qa_gen.document.models import Document, DocumentType


def create_test_file(content):
    """Create a temporary test file with the given content"""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, content.encode('utf-8'))
    os.close(fd)
    return path


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing"""
    content = """# Sample Document

This is a sample document for testing the SemanticQAGen pipeline.
It contains multiple paragraphs with some information.

## Section 1

This section contains factual information that can be used for question generation.
The earth is the third planet from the sun and the only astronomical object known to harbor life.

## Section 2

This section explores more conceptual ideas.
Knowledge representation is the study of how knowledge can be represented symbolically.
"""
    path = create_test_file(content)
    yield path
    os.unlink(path)  # Clean up after test


@pytest.mark.asyncio
@patch("semantic_qa_gen.llm.adapters.remote.OpenAIAdapter.call_model")
@patch("semantic_qa_gen.llm.adapters.local.LocalLLMAdapter.call_model")
async def test_minimal_pipeline(mock_local_call, mock_remote_call, sample_text_file):
    """
    Test a minimal version of the pipeline with mocked LLM responses
    """

    # Set up mock responses as an async functions
    async def remote_call_side_effect(*args, **kwargs):
        # Check which call this is and return appropriate mock response
        if hasattr(remote_call_side_effect, 'call_count'):
            remote_call_side_effect.call_count += 1
        else:
            remote_call_side_effect.call_count = 1

        # Return different responses based on the call count
        if remote_call_side_effect.call_count == 1:
            # Analysis response
            return {
                "information_density": 0.7,
                "topic_coherence": 0.8,
                "complexity": 0.6,
                "estimated_question_yield": {
                    "factual": 3,
                    "inferential": 2,
                    "conceptual": 1
                },
                "key_concepts": ["Earth", "Knowledge", "Life"],
                "notes": "Sample document suitable for question generation"
            }
        elif remote_call_side_effect.call_count == 2:
            # Question generation response
            return [
                {
                    "question": "What planet is Earth in our solar system?",
                    "answer": "Earth is the third planet from the sun.",
                    "category": "factual"
                },
                {
                    "question": "What is knowledge representation?",
                    "answer": "Knowledge representation is the study of how knowledge can be represented symbolically.",
                    "category": "factual"
                }
            ]
        else:
            # Validation responses for questions
            return {
                "is_valid": True,
                "factual_accuracy": 0.9,
                "answer_completeness": 0.8,
                "question_clarity": 0.9,
                "reasons": ["The answer is factually correct based on the source text."]
            }

    # Set up the mocks with our side effect function
    mock_remote_call.side_effect = remote_call_side_effect
    mock_local_call.side_effect = AsyncMock()  # Fallback if local is used

    # Create a minimal configuration using only remote service
    config = {
        "llm_services": {
            "local": {"enabled": False},
            "remote": {
                "enabled": True,
                "api_key": "dummy-key",
                "default_for": ["analysis", "generation", "validation", "chunking"]
            }
        },
        "processing": {"concurrency": 1}
    }

    # Initialize SemanticQAGen with our test configuration
    qa_gen = SemanticQAGen(config_dict=config)

    # Process the document
    result = qa_gen.process_document(sample_text_file)

    # Check that we got the expected results
    assert "questions" in result
    assert len(result["questions"]) > 0

    # Check that our mock was called the expected number of times
    assert mock_remote_call.call_count >= 3  # At least analysis, questions, validation

    # Create a temporary output file and save the results
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        output_path = qa_gen.save_questions(result, tmp.name)
        assert os.path.exists(output_path)
