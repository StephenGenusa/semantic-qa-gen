import pytest
import os
import asyncio
import time
from unittest.mock import MagicMock, patch

from semantic_qa_gen.utils.error import (
    SemanticQAGenError, with_error_handling,
    async_with_error_handling, RetryStrategy
)
from semantic_qa_gen.utils.progress import ProgressReporter, ProcessingStage
from semantic_qa_gen.utils.checkpoint import CheckpointManager


def test_retry_strategy():
    """Test the retry strategy implementation"""
    strategy = RetryStrategy(max_retries=3, base_delay=0.01, backoff_factor=2)
    
    # Test should_retry
    assert strategy.should_retry(Exception("Test"), 0) is True  # First attempt
    assert strategy.should_retry(Exception("Test"), 1) is True  # Second attempt
    assert strategy.should_retry(Exception("Test"), 2) is True  # Third attempt
    assert strategy.should_retry(Exception("Test"), 3) is False  # Fourth attempt (exceeds max)
    
    # Test get_delay
    assert strategy.get_delay(0) == 0.01  # First delay
    assert strategy.get_delay(1) == 0.02  # Second delay
    assert strategy.get_delay(2) == 0.04  # Third delay


def test_with_error_handling_decorator():
    """Test the error handling decorator"""
    mock_func = MagicMock(side_effect=[ValueError("First error"), "success"])
    
    # Apply decorator with retry
    decorated = with_error_handling(error_types=ValueError, max_retries=1)(mock_func)
    
    # Call should succeed on second attempt
    result = decorated()
    assert result == "success"
    assert mock_func.call_count == 2


def test_with_error_handling_max_retries():
    """Test behavior when max retries are exhausted"""
    mock_func = MagicMock(side_effect=ValueError("Persistent error"))
    
    # Apply decorator with retry
    decorated = with_error_handling(error_types=ValueError, max_retries=2)(mock_func)
    
    # Should fail after max retries
    with pytest.raises(ValueError):
        decorated()
    
    assert mock_func.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_async_with_error_handling():
    """Test async error handling decorator"""
    mock_func = MagicMock(side_effect=[
        asyncio.Future(),
        asyncio.Future()
    ])
    mock_func.side_effect[0].set_exception(ValueError("First error"))
    mock_func.side_effect[1].set_result("success")
    
    # Apply decorator
    decorated = async_with_error_handling(error_types=ValueError, max_retries=1)(mock_func)
    
    # Call should succeed on second attempt
    result = await decorated()
    assert result == "success"
    assert mock_func.call_count == 2


def test_progress_reporter_initialization():
    """Test progress reporter initialization"""
    reporter = ProgressReporter(show_progress_bar=True)
    assert reporter.current_stage == ProcessingStage.LOADING
    
    # Test with rich available/unavailable
    with patch("semantic_qa_gen.utils.progress.RICH_AVAILABLE", True):
        reporter = ProgressReporter(show_progress_bar=True)
        assert reporter.rich_enabled is True
        
    with patch("semantic_qa_gen.utils.progress.RICH_AVAILABLE", False):
        reporter = ProgressReporter(show_progress_bar=True)
        assert reporter.rich_enabled is False


def test_progress_reporter_update_stage():
    """Test updating processing stage"""
    reporter = ProgressReporter(show_progress_bar=False)
    
    with patch.object(reporter, "logger") as mock_logger:
        reporter.update_stage(ProcessingStage.CHUNKING)
        assert reporter.current_stage == ProcessingStage.CHUNKING
        mock_logger.info.assert_called_once()


def test_progress_reporter_update_progress():
    """Test updating progress"""
    reporter = ProgressReporter(show_progress_bar=False)
    
    with patch.object(reporter, "logger") as mock_logger:
        # Update with extra info
        reporter.update_progress(5, 10, {"key": "value"})
        mock_logger.debug.assert_called_once()
        
        # Simple update without extras
        reporter.update_progress(10, 10)


def test_checkpoint_manager(tmp_path):
    """Test checkpoint manager functionality"""
    from semantic_qa_gen.config.schema import SemanticQAGenConfig
    
    # Create config with temp directory
    config = SemanticQAGenConfig()
    config.processing.checkpoint_dir = str(tmp_path)
    
    # Create manager
    manager = CheckpointManager(config)
    
    # Mock document and chunks
    document = MagicMock(id="doc-123", content="Test content")
    chunks = [MagicMock(id=f"chunk-{i}") for i in range(3)]
    stats = {"processed_questions": 10}
    
    # Test saving checkpoint
    checkpoint_path = manager.save_checkpoint(document, chunks, 2, stats)
    assert checkpoint_path.startswith(str(tmp_path))
    assert os.path.exists(checkpoint_path)
    
    # Test loading checkpoint
    loaded_checkpoint = manager.load_checkpoint(document)
    assert loaded_checkpoint is not None
    assert loaded_checkpoint["document_id"] == "doc-123"
    assert loaded_checkpoint["current_chunk_idx"] == 2
