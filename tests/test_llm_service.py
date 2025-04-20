import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from semantic_qa_gen.llm.adapters.local import LocalLLMAdapter
from semantic_qa_gen.llm.adapters.remote import OpenAIAdapter
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.utils.error import LLMServiceError


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API calls"""
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    client.__aenter__.return_value = client
    client.post.return_value = response
    return client


@pytest.fixture
def local_llm_config():
    """Local LLM configuration for testing"""
    return {
        "url": "http://localhost:11434/api",
        "model": "mistral:7b",
        "timeout": 30
    }


@pytest.fixture
def remote_openai_config():
    """OpenAI configuration for testing"""
    return {
        "api_key": "test-api-key",
        "model": "gpt-4",
        "timeout": 30
    }


@pytest.fixture
def sample_chunk():
    """Sample chunk for testing"""
    return Chunk(
        content="This is sample content for testing LLM services.",
        id="chunk-123",
        document_id="doc-456",
        sequence=1,
        context={"title": "Test Document"}
    )


@pytest.mark.asyncio
async def test_local_llm_call_model(local_llm_config, mock_httpx_client):
    """Test calling a local LLM"""
    # Set up mock response
    mock_response = mock_httpx_client.post.return_value
    mock_response.json.return_value = {"response": "Test response"}
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = LocalLLMAdapter(local_llm_config)
        result = await adapter.call_model("Test prompt", system_prompt="System prompt")
        
        # Assertions
        assert result == "Test response"
        mock_httpx_client.post.assert_called_once()
        assert "test prompt" in str(mock_httpx_client.post.call_args).lower()


@pytest.mark.asyncio
async def test_local_llm_json_response(local_llm_config, mock_httpx_client):
    """Test parsing JSON responses from local LLM"""
    # Set up mock response with JSON
    mock_response = mock_httpx_client.post.return_value
    mock_response.json.return_value = {"response": '{"key": "value"}'}
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = LocalLLMAdapter(local_llm_config)
        result = await adapter.call_model("Test prompt", json_response=True)
        
        # Assertions
        assert isinstance(result, dict)
        assert result.get("key") == "value"


@pytest.mark.asyncio
async def test_local_llm_error_handling(local_llm_config, mock_httpx_client):
    """Test error handling in local LLM adapter"""
    # Set up mock to raise exception
    mock_httpx_client.post.side_effect = httpx.HTTPError("Connection error")
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = LocalLLMAdapter(local_llm_config)
        
        with pytest.raises(LLMServiceError):
            await adapter.call_model("Test prompt")


@pytest.mark.asyncio
async def test_openai_adapter_call_model(remote_openai_config, mock_httpx_client):
    """Test calling OpenAI API"""
    # Set up mock response
    mock_response = mock_httpx_client.post.return_value
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "API response"}}],
        "usage": {"total_tokens": 100}
    }
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = OpenAIAdapter(remote_openai_config)
        result = await adapter.call_model("Test prompt", system_prompt="System prompt")
        
        # Assertions
        assert result == "API response"
        mock_httpx_client.post.assert_called_once()
        # Check that API key was included in headers
        headers = mock_httpx_client.post.call_args[1]["headers"]
        assert "Authorization" in headers
        assert "Bearer test-api-key" in headers["Authorization"]


@pytest.mark.asyncio
async def test_analyze_chunk(remote_openai_config, sample_chunk, mock_httpx_client):
    """Test analyzing a chunk with an LLM service"""
    # Set up mock response for JSON data
    mock_response = mock_httpx_client.post.return_value
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "information_density": 0.8,
                    "topic_coherence": 0.7,
                    "complexity": 0.6,
                    "estimated_question_yield": {
                        "factual": 3,
                        "inferential": 2,
                        "conceptual": 1
                    },
                    "key_concepts": ["concept1", "concept2"],
                    "notes": "Test notes"
                })
            }
        }],
        "usage": {"total_tokens": 100}
    }
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = OpenAIAdapter(remote_openai_config)
        
        # Patch tiktoken to avoid dependency issues in testing
        with patch.object(adapter, "encoding"):
            result = await adapter.analyze_chunk(sample_chunk)
        
        # Assertions
        assert isinstance(result, AnalysisResult)
        assert result.chunk_id == sample_chunk.id
        assert result.information_density == 0.8
        assert result.topic_coherence == 0.7
        assert len(result.key_concepts) == 2


@pytest.mark.asyncio
async def test_generate_questions(remote_openai_config, sample_chunk, mock_httpx_client):
    """Test generating questions with an LLM service"""
    # Set up mock analysis
    analysis = AnalysisResult(
        chunk_id=sample_chunk.id,
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
    
    # Set up mock response with question data
    mock_response = mock_httpx_client.post.return_value
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps([
                    {
                        "question": "What is concept1?",
                        "answer": "Concept1 is...",
                        "category": "factual"
                    },
                    {
                        "question": "How do concept1 and concept2 relate?",
                        "answer": "They relate through...",
                        "category": "inferential"
                    }
                ])
            }
        }],
        "usage": {"total_tokens": 200}
    }
    
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        adapter = OpenAIAdapter(remote_openai_config)
        
        # Patch tiktoken and uuid
        with patch.object(adapter, "encoding"):
            with patch("uuid.uuid4", return_value="mock-uuid"):
                questions = await adapter.generate_questions(
                    chunk=sample_chunk,
                    analysis=analysis,
                    categories={"factual": 1, "inferential": 1}
                )
        
        # Assertions
        assert len(questions) == 2
        assert questions[0].text == "What is concept1?"
        assert questions[0].category == "factual"
        assert questions[1].text == "How do concept1 and concept2 relate?"
