# filename: semantic_qa_gen/llm/adapters/base.py

"""Base abstract class for LLM adapters."""

from abc import ABC, abstractmethod
import logging
import re
import yaml # Keep for fallback parsing just in case
import json
from typing import Dict, List, Optional, Any, Union

# Use Pydantic V2 imports
from pydantic import BaseModel # For potential internal types

# Local project imports
from semantic_qa_gen.config.schema import ModelConfig
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError
from semantic_qa_gen.utils.error import async_with_error_handling # Use the refined decorator


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM service adapters.

    Defines the common interface for interacting with different LLM backends.
    Implementations handle provider-specific details like API calls,
    authentication, and potentially response parsing nuances.

    Core Task Methods (analyze_chunk, generate_questions, validate_question)
    are *removed* from the adapter base. They are now handled by dedicated
    components (`SemanticAnalyzer`, `QuestionGenerator`, `ValidationEngine`)
    which *use* this adapter's `generate_completion` method.
    """

    def __init__(self, service_config: Dict[str, Any], prompt_manager: PromptManager):
        """
        Initialize the base adapter.

        Args:
            service_config: Configuration dictionary specific to this service,
                            validated against BaseLLMServiceDetails schema or its children.
            prompt_manager: An instance of the PromptManager.
        """
        self.service_config = service_config # This is the config dict for 'local' or 'remote' section
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Extract common config values or use defaults
        self.timeout = service_config.get('timeout', 120)
        self.max_retries = service_config.get('max_retries', 3)
        self.initial_delay = service_config.get('initial_delay', 1.0)
        self.max_delay = service_config.get('max_delay', 60.0)
        # Note: Specific configuration aspects like api_key, api_base, provider are
        # handled within the concrete adapter implementation (e.g., OpenAIAdapter).

    @abstractmethod
    # Apply retry decorator to the core communication method
    @async_with_error_handling(
        error_types=(LLMServiceError,), # Retry only on potentially transient LLMServiceErrors
        # Note: The decorator now needs to extract config from the instance
        # This requires a slight modification or passing config during call.
        # Let's assume the decorator can access self.max_retries etc.
        # If not, we'd need to pass them: max_retries=self.max_retries ... No, decorators don't work like that easily.
        # --> Modify the decorator slightly OR handle retries manually here.
        # Simpler: Modify the decorator to fetch from `self` if available.
        # Okay, let's simplify and handle retries within the adapter implementation if needed,
        # or enhance the decorator later. For now, remove it from the abstract method signature.
    )
    async def generate_completion(self, prompt: str, model_config: ModelConfig) -> str:
        """
        Core method to get a text completion from the LLM backend.

        This method is responsible for the actual communication with the LLM
        (e.g., making the API call). It should handle authentication, endpoint
        construction, payload formatting, and basic error handling related to
        the communication itself, potentially raising LLMServiceError on failure.
        Retry logic for transient network errors or rate limits should ideally
        be implemented within this method or called helper methods.

        Args:
            prompt: The fully formatted prompt string to send to the LLM.
            model_config: Configuration for the specific model invocation
                          (name, temperature, max_tokens, etc.).

        Returns:
            The raw text completion string from the LLM.

        Raises:
            LLMServiceError: If the API call fails after exhausting retries or
                             due to non-retryable errors (e.g., auth failure).
            ConfigurationError: If the adapter configuration is insufficient
                                 to make the call (e.g., missing API key/URL).
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources used by the adapter.

        This should be called to gracefully close connections, such as
        stopping HTTP client sessions.
        """
        pass

    # Helper method potentially used by concrete implementation's retry logic
    def _get_retry_delay(self, attempt: int) -> float:
        """Calculates exponential backoff delay with jitter."""
        import random
        delay = self.initial_delay * (2 ** attempt) # Exponential backoff
        delay = min(delay, self.max_delay)
        # Add jitter: delay +/- (delay * 0.1)
        jitter_amount = delay * 0.1
        delay += random.uniform(-jitter_amount, jitter_amount)
        return max(0.1, delay) # Ensure minimum delay

    # Optional: Token counting helper can remain if useful for subclasses
    @abstractmethod
    def count_tokens(self, text: str, model_config: ModelConfig) -> int:
        """
        Estimate the number of tokens for a given text string based on the specified model.

        Args:
            text: The text string to count tokens for.
            model_config: The model configuration (name is important for tokenizer selection).

        Returns:
            The estimated number of tokens.
        """
        pass

