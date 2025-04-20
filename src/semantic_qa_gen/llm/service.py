"""LLM service interface and base classes for SemanticQAGen."""

from abc import ABC, abstractmethod
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union

from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.utils.error import LLMServiceError, with_error_handling


class LLMServiceInterface(ABC):
    """
    Abstract interface for language model services.
    
    This interface defines the methods that all LLM service implementations
    must provide, abstracting away the specific details of different LLM
    providers (OpenAI, local models, etc.).
    """
    
    @abstractmethod
    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a document chunk to determine its information density and question potential.
        
        Args:
            chunk: The document chunk to analyze.
            
        Returns:
            AnalysisResult containing information about the chunk.
            
        Raises:
            LLMServiceError: If analysis fails.
        """
        pass
    
    @abstractmethod
    async def generate_questions(self, 
                               chunk: Chunk, 
                               analysis: AnalysisResult, 
                               count: Optional[int] = None, 
                               categories: Optional[Dict[str, int]] = None) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis.
        
        Args:
            chunk: The document chunk to generate questions for.
            analysis: Analysis results for the chunk.
            count: Optional total number of questions to generate.
            categories: Optional dictionary mapping category names to question counts.
            
        Returns:
            List of generated Question objects.
            
        Raises:
            LLMServiceError: If question generation fails.
        """
        pass
    
    @abstractmethod
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a generated question against the source chunk.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            Validation results dictionary.
            
        Raises:
            LLMServiceError: If validation fails.
        """
        pass
    
    @abstractmethod
    async def call_model(self, 
                       prompt: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Call the language model with a prompt and receive a response.
        
        Args:
            prompt: The main prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response length in tokens.
            json_response: Whether to parse the response as JSON.
            
        Returns:
            Model response as string or parsed JSON.
            
        Raises:
            LLMServiceError: If the model call fails.
        """
        pass


class BaseLLMService(LLMServiceInterface):
    """
    Base implementation for LLM services with common functionality.
    
    This class provides a foundation for specific LLM service implementations,
    with shared functionality like rate limiting, retry logic, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM service.
        
        Args:
            config: Configuration dictionary for the service.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limit_tokens = config.get('rate_limit_tokens', 90000)  # tokens per minute
        self.rate_limit_requests = config.get('rate_limit_requests', 100)  # requests per minute
        self.timeout = config.get('timeout', 60)
        
        # Rate limiting state
        self._request_timestamps = []
        self._token_usage = []
    
    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a document chunk using the LLM.
        
        Args:
            chunk: The document chunk to analyze.
            
        Returns:
            AnalysisResult containing information about the chunk.
            
        Raises:
            LLMServiceError: If analysis fails.
        """
        # This will be implemented in derived classes or with prompt templates
        raise NotImplementedError("Method not implemented in base class")
    
    async def generate_questions(self, 
                               chunk: Chunk, 
                               analysis: AnalysisResult, 
                               count: Optional[int] = None, 
                               categories: Optional[Dict[str, int]] = None) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis.
        
        Args:
            chunk: The document chunk to generate questions for.
            analysis: Analysis results for the chunk.
            count: Optional total number of questions to generate.
            categories: Optional dictionary mapping category names to question counts.
            
        Returns:
            List of generated Question objects.
            
        Raises:
            LLMServiceError: If question generation fails.
        """
        # This will be implemented in derived classes or with prompt templates
        raise NotImplementedError("Method not implemented in base class")
    
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a generated question against the source chunk.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            Validation results dictionary.
            
        Raises:
            LLMServiceError: If validation fails.
        """
        # This will be implemented in derived classes or with prompt templates
        raise NotImplementedError("Method not implemented in base class")
    
    async def call_model(self, 
                       prompt: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Call the language model with a prompt and receive a response.
        
        Args:
            prompt: The main prompt to send.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response length in tokens.
            json_response: Whether to parse the response as JSON.
            
        Returns:
            Model response as string or parsed JSON.
            
        Raises:
            LLMServiceError: If the model call fails.
        """
        # This will be implemented in derived classes
        raise NotImplementedError("Method not implemented in base class")
    
    async def _enforce_rate_limits(self, estimated_tokens: int = 1000) -> None:
        """
        Enforce rate limits by waiting if necessary.
        
        Args:
            estimated_tokens: Estimated token usage for this request.
        """
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Update request and token tracking
        self._request_timestamps = [t for t in self._request_timestamps if t > one_minute_ago]
        self._token_usage = [t for t, _ in self._token_usage if t > one_minute_ago]
        
        # Check rate limits
        if len(self._request_timestamps) >= self.rate_limit_requests:
            # Request rate limit hit, wait until oldest request is outside window
            wait_time = self._request_timestamps[0] - one_minute_ago
            self.logger.warning(f"Rate limit (requests) reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
            # Recursive call to recheck after waiting
            await self._enforce_rate_limits(estimated_tokens)
            return
        
        current_token_usage = sum(tokens for _, tokens in self._token_usage)
        if current_token_usage + estimated_tokens > self.rate_limit_tokens:
            # Token rate limit hit, wait until enough tokens are outside window
            tokens_to_free = current_token_usage + estimated_tokens - self.rate_limit_tokens
            for t, tokens in self._token_usage:
                if tokens_to_free <= 0:
                    break
                wait_time = t - one_minute_ago
                tokens_to_free -= tokens
            
            self.logger.warning(f"Rate limit (tokens) reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
            # Recursive call to recheck after waiting
            await self._enforce_rate_limits(estimated_tokens)
            return
        
        # If we get here, rate limits are satisfied
        self._request_timestamps.append(current_time)
        self._token_usage.append((current_time, estimated_tokens))
