
"""Error handling for SemanticQAGen."""

import logging
import time
import functools
import traceback
from typing import Optional, Dict, Any, Callable, Type, Union, List, Tuple


class SemanticQAGenError(Exception):
    """Base exception class for SemanticQAGen."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with message and optional details.
        
        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message)
        self.details = details or {}
        self.timestamp = time.time()


class ConfigurationError(SemanticQAGenError):
    """Exception raised for configuration errors."""
    pass


class DocumentError(SemanticQAGenError):
    """Exception raised for document processing errors."""
    pass


class ChunkingError(SemanticQAGenError):
    """Exception raised for chunking errors."""
    pass


class LLMServiceError(SemanticQAGenError):
    """Exception raised for LLM service errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, retryable: bool = True):
        """
        Initialize with message and retry flag.
        
        Args:
            message: Error message.
            details: Additional error details.
            retryable: Whether the operation can be retried.
        """
        super().__init__(message, details)
        self.retryable = retryable


class ValidationError(SemanticQAGenError):
    """Exception raised for validation errors."""
    pass


class OutputError(SemanticQAGenError):
    """Exception raised for output formatting errors."""
    pass


class RetryStrategy:
    """Strategy for retrying operations."""
    
    def __init__(self, max_retries: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 30.0,
                 backoff_factor: float = 2.0,
                 jitter: float = 0.2):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            backoff_factor: Factor by which to increase delay with each attempt.
            jitter: Random factor to add to delay to prevent synchronized retries.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error: The exception that was raised.
            attempt: The current attempt number (0-indexed).
            
        Returns:
            True if the operation should be retried, False otherwise.
        """
        # Check if we've exceeded max retries
        if attempt >= self.max_retries:
            return False
            
        # For LLM errors, check if they're marked as retryable
        if isinstance(error, LLMServiceError) and not error.retryable:
            return False
            
        return True
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry with jitter.
        
        Args:
            attempt: The current attempt number (0-indexed).
            
        Returns:
            Delay in seconds before next retry attempt.
        """
        import random
        
        # Calculate base delay with exponential backoff
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent synchronized retries
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        # Ensure delay is positive
        return max(0.1, delay)


class ErrorHandler:
    """Handles errors in SemanticQAGen."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance for error logging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.default_retry_strategy = RetryStrategy()
        self.error_listeners: List[Callable[[Exception, Dict[str, Any]], None]] = []
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error.
        
        Args:
            error: The exception that was raised.
            context: Additional context about where the error occurred.
        """
        error_type = type(error).__name__
        error_msg = str(error)
        context_str = str(context) if context else "No context provided"
        
        # Basic logging
        self.logger.error(f"{error_type}: {error_msg} - Context: {context_str}")
        
        # Specialized handling based on error type
        if isinstance(error, LLMServiceError):
            self.logger.warning("LLM service error encountered. Check API keys and quotas.")
        elif isinstance(error, ConfigurationError):
            self.logger.error("Configuration error. Please check your configuration.")
        elif isinstance(error, DocumentError):
            self.logger.error("Document processing error. Check file format and permissions.")
        
        # Detailed debug logging with traceback
        self.logger.debug(f"Error details: {traceback.format_exc()}")
        
        # Notify all error listeners
        for listener in self.error_listeners:
            try:
                listener(error, context or {})
            except Exception as e:
                self.logger.error(f"Error in error listener: {e}")
    
    def retry(self, 
              func: Callable, 
              retry_strategy: Optional[RetryStrategy] = None,
              error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
              context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Retry a function with backoff.
        
        Args:
            func: Function to retry.
            retry_strategy: Strategy for retrying.
            error_types: Exception type(s) that should trigger a retry.
            context: Additional context about where the error occurred.
            
        Returns:
            Result of the function.
            
        Raises:
            The last exception raised by the function.
        """
        retry_strategy = retry_strategy or self.default_retry_strategy
        context = context or {}
        
        attempt = 0
        last_error = None
        
        while True:
            try:
                return func()
            except error_types as e:
                last_error = e
                attempt += 1
                
                # Store the attempt in context
                context['attempt'] = attempt
                
                # Check if we should retry
                if not retry_strategy.should_retry(e, attempt):
                    self.logger.warning(f"Max retries ({retry_strategy.max_retries}) exceeded")
                    self.handle_error(e, {**context, "final_attempt": True})
                    raise
                
                # Calculate delay for this attempt
                delay = retry_strategy.get_delay(attempt)
                
                self.logger.warning(
                    f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                self.handle_error(e, {**context, "retrying": True, "delay": delay})
                time.sleep(delay)


def with_error_handling(error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
                       max_retries: int = 3,
                       base_delay: float = 1.0,
                       jitter: float = 0.1):
    """
    Decorator to apply error handling and retry logic to functions.
    
    Args:
        error_types: Exception type(s) that should trigger a retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries.
        jitter: Random factor to add to delay.
        
    Returns:
        Decorated function with error handling.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            retry_strategy = RetryStrategy(
                max_retries=max_retries, 
                base_delay=base_delay,
                jitter=jitter
            )
            
            # Prepare context with function info and args summary
            context = {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            }
            
            return handler.retry(
                lambda: func(*args, **kwargs),
                retry_strategy=retry_strategy,
                error_types=error_types,
                context=context
            )
        return wrapper
    return decorator


def async_with_error_handling(error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
                            max_retries: int = 3,
                            base_delay: float = 1.0,
                            jitter: float = 0.1):
    """
    Decorator to apply error handling and retry logic to async functions.
    
    Args:
        error_types: Exception type(s) that should trigger a retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries.
        jitter: Random factor to add to delay.
        
    Returns:
        Decorated async function with error handling.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            attempt = 0
            last_error = None
            
            # Prepare context with function info
            context = {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            }
            
            while attempt <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    attempt += 1
                    
                    # Check if we should retry
                    if attempt > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        # Log detailed error on final attempt
                        error_type = type(e).__name__
                        logger.error(f"Final error: {error_type}: {str(e)}")
                        raise
                    
                    # Calculate delay with jitter
                    import random
                    delay = base_delay * (2 ** (attempt - 1))
                    if jitter > 0:
                        delay += random.uniform(0, jitter * delay)
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for {func.__name__} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # For async functions, use asyncio.sleep
                    import asyncio
                    await asyncio.sleep(delay)
            
            # This should never be reached due to the raise inside the loop
            raise last_error
            
        return wrapper
    return decorator
