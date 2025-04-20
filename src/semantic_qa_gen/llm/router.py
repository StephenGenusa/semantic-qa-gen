"""Enhanced task routing with improved error handling."""

import logging
import time
from typing import Dict, Any, Optional, List, Type, Union

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.service import LLMServiceInterface
from semantic_qa_gen.llm.adapters.remote import OpenAIAdapter
from semantic_qa_gen.llm.adapters.local import LocalLLMAdapter
from semantic_qa_gen.utils.error import LLMServiceError


class TaskRouter:
    """
    Routes LLM tasks with enhanced error handling and fallback mechanisms.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the task router."""
        self.config_manager = config_manager
        self.config = config_manager.get_section("llm_services")
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.services: Dict[str, LLMServiceInterface] = {}
        self._retry_counts: Dict[str, int] = {}
        self._last_failure_time: Dict[str, float] = {}
        self._initialize_services()
        
        # Task routing configuration
        self.routing = {
            "chunking": self._get_service_for_task("chunking"),
            "analysis": self._get_service_for_task("analysis"),
            "generation": self._get_service_for_task("generation"),
            "validation": self._get_service_for_task("validation")
        }
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 3  # Failures before switching to fallback
        self.circuit_reset_time = 300.0     # 5 minutes before trying again
    
    def _initialize_services(self) -> None:
        """Initialize all configured LLM services."""
        # Initialize remote service if enabled
        if self.config.remote.enabled:
            try:
                self.services["remote"] = OpenAIAdapter(self.config.remote.dict())
                self.logger.info(f"Initialized remote LLM service: {self.config.remote.provider}")
            except Exception as e:
                self.logger.error(f"Failed to initialize remote LLM service: {str(e)}")
        
        # Initialize local service if enabled
        if self.config.local.enabled:
            try:
                self.services["local"] = LocalLLMAdapter(self.config.local.dict())
                self.logger.info(f"Initialized local LLM service: {self.config.local.model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize local LLM service: {str(e)}")
    
    def _get_service_for_task(self, task_type: str) -> str:
        """Get the appropriate service name for a task type."""
        # Check remote service first
        if self.config.remote.enabled and task_type in self.config.remote.default_for:
            return "remote"
        
        # Check local service next
        if self.config.local.enabled and task_type in self.config.local.default_for:
            return "local"
        
        # Default to remote if available, otherwise local
        if "remote" in self.services:
            return "remote"
        elif "local" in self.services:
            return "local"
        
        # No services available
        raise LLMServiceError("No LLM services available")
    
    def get_service(self, task_type: str) -> LLMServiceInterface:
        """
        Get the appropriate LLM service with circuit breaker protection.
        
        Args:
            task_type: Type of task.
            
        Returns:
            LLM service instance.
            
        Raises:
            LLMServiceError: If no service is available for the task.
        """
        primary_service_name = self.routing.get(task_type)
        fallback_service_name = "local" if primary_service_name == "remote" else "remote"
        
        # Check if primary service is in circuit-broken state
        if primary_service_name in self._retry_counts:
            failures = self._retry_counts[primary_service_name]
            last_failure = self._last_failure_time.get(primary_service_name, 0)
            
            # If circuit breaker tripped and not enough time passed
            if failures >= self.circuit_breaker_threshold:
                time_since_failure = time.time() - last_failure
                if time_since_failure < self.circuit_reset_time:
                    self.logger.warning(
                        f"Circuit breaker active for {primary_service_name}, "
                        f"using {fallback_service_name} instead. "
                        f"Will retry in {self.circuit_reset_time - time_since_failure:.1f}s"
                    )
                    # Use fallback service if available
                    if fallback_service_name in self.services:
                        return self.services[fallback_service_name]
                else:
                    # Reset circuit breaker after timeout period
                    self._retry_counts[primary_service_name] = 0
        
        # Get the primary service
        service = self.services.get(primary_service_name)
        if not service:
            # Try fallback
            service = self.services.get(fallback_service_name)
            
            if service:
                self.logger.warning(f"Using fallback {fallback_service_name} service for {task_type}")
            else:
                raise LLMServiceError(f"No LLM service available for task type: {task_type}")
                
        return service

    def mark_service_failure(self, service_name: str) -> None:
        """
        Mark a service failure for circuit breaker tracking.

        Args:
            service_name: Name of the failed service.
        """
        self._retry_counts[service_name] = self._retry_counts.get(service_name, 0) + 1
        self._last_failure_time[service_name] = time.time()
        self.logger.warning(
            f"Service {service_name} failed {self._retry_counts[service_name]} times. "
            f"Circuit breaker threshold: {self.circuit_breaker_threshold}"
        )

    def reset_service_failure(self, service_name: str) -> None:
        """
        Reset failure counter for a service after successful operation.
        
        Args:
            service_name: Name of the service.
        """
        if service_name in self._retry_counts and self._retry_counts[service_name] > 0:
            self._retry_counts[service_name] = 0
            self.logger.info(f"Reset failure counter for service {service_name}")
    
    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a document chunk with optimized fallback.
        """
        task_type = "analysis"
        service_name = self.routing.get(task_type)
        
        try:
            service = self.get_service(task_type)
            result = await service.analyze_chunk(chunk)
            # Mark success
            self.reset_service_failure(service_name)
            return result
        except Exception as e:
            self.logger.error(f"Service {service_name} failed during {task_type}: {str(e)}")
            self.mark_service_failure(service_name)
            
            # Try fallback service
            fallback_name = "local" if service_name == "remote" else "remote"
            if fallback_name in self.services:
                self.logger.info(f"Trying fallback service {fallback_name} for {task_type}")
                try:
                    fallback_service = self.services[fallback_name]
                    return await fallback_service.analyze_chunk(chunk)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback service {fallback_name} also failed: {str(fallback_e)}")
                    # Both services failed, re-raise the original error
                    raise e
            else:
                # No fallback available
                raise e
    
    async def generate_questions(self, chunk: Chunk, analysis: AnalysisResult, 
                               count: Optional[int] = None,
                               categories: Optional[Dict[str, int]] = None) -> List[Question]:
        """
        Generate questions with automatic fallback.
        """
        task_type = "generation"
        service_name = self.routing.get(task_type)
        
        try:
            service = self.get_service(task_type)
            result = await service.generate_questions(chunk, analysis, count, categories)
            # Mark success
            self.reset_service_failure(service_name)
            return result
        except Exception as e:
            self.logger.error(f"Service {service_name} failed during {task_type}: {str(e)}")
            self.mark_service_failure(service_name)
            
            # Try fallback service
            fallback_name = "local" if service_name == "remote" else "remote"
            if fallback_name in self.services:
                self.logger.info(f"Trying fallback service {fallback_name} for {task_type}")
                try:
                    fallback_service = self.services[fallback_name]
                    return await fallback_service.generate_questions(chunk, analysis, count, categories)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback service {fallback_name} also failed: {str(fallback_e)}")
                    raise e
            else:
                raise e
    
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a question with automatic fallback.
        """
        task_type = "validation"
        service_name = self.routing.get(task_type)
        
        try:
            service = self.get_service(task_type)
            result = await service.validate_question(question, chunk)
            # Mark success
            self.reset_service_failure(service_name)
            return result
        except Exception as e:
            self.logger.error(f"Service {service_name} failed during {task_type}: {str(e)}")
            self.mark_service_failure(service_name)
            
            # Try fallback service
            fallback_name = "local" if service_name == "remote" else "remote"
            if fallback_name in self.services:
                self.logger.info(f"Trying fallback service {fallback_name} for {task_type}")
                try:
                    fallback_service = self.services[fallback_name]
                    return await fallback_service.validate_question(question, chunk)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback service {fallback_name} also failed: {str(fallback_e)}")
                    raise e
            else:
                raise e
