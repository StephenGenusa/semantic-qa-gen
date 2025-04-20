"""Factual accuracy validator for question validation."""

import asyncio
from typing import Dict, Any, Optional, List

from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.llm.router import TaskRouter
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError


class FactualAccuracyValidator(BaseValidator):
    """
    Validator that checks if answers are factually accurate based on the source text.
    
    This validator uses an LLM to assess whether the answer contains information
    that is consistent with the source text.
    """
    
    def __init__(self, 
                task_router: TaskRouter, 
                prompt_manager: PromptManager,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the factual accuracy validator.
        
        Args:
            task_router: Task router for LLM calls.
            prompt_manager: Prompt manager for validation prompts.
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.threshold = self.config.get('threshold', 0.7)
    
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """
        Validate that the answer is factually accurate based on the source text.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            ValidationResult indicating factual accuracy.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Get the validation service
            service = self.task_router.get_service("validation")
            
            # Use the system's validate_question method
            result = await service.validate_question(question, chunk)
            
            # Extract the factual accuracy score
            factual_accuracy = result.get("factual_accuracy", 0.0)
            is_valid = factual_accuracy >= self.threshold
            
            # Extract reasons
            reasons = result.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
                
            # Create validation result
            validation_result = ValidationResult(
                question_id=question.id,
                is_valid=is_valid,
                scores={"factual_accuracy": factual_accuracy},
                reasons=[f"Factual accuracy: {factual_accuracy:.2f}" + 
                         (f" - {reasons[0]}" if reasons else "")],
                suggested_improvements=result.get("suggested_improvements")
            )
            
            self.logger.debug(
                f"Factual validation for q:{question.id}: "
                f"accuracy={factual_accuracy:.2f}, valid={is_valid}"
            )
            
            return validation_result
            
        except Exception as e:
            raise ValidationError(f"Factual accuracy validation failed: {str(e)}")


class AnswerCompletenessValidator(BaseValidator):
    """
    Validator that checks if answers completely address the questions.
    
    This validator assesses whether the answer provides a thorough
    response to the question being asked.
    """
    
    def __init__(self, 
                task_router: TaskRouter, 
                prompt_manager: PromptManager,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the answer completeness validator.
        
        Args:
            task_router: Task router for LLM calls.
            prompt_manager: Prompt manager for validation prompts.
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.threshold = self.config.get('threshold', 0.7)
    
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """
        Validate that the answer completely addresses the question.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            ValidationResult indicating answer completeness.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Get the validation service
            service = self.task_router.get_service("validation")
            
            # Use the system's validate_question method
            result = await service.validate_question(question, chunk)
            
            # Extract the completeness score
            completeness = result.get("answer_completeness", 0.0)
            is_valid = completeness >= self.threshold
            
            # Extract reasons
            reasons = result.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
                
            # Create validation result
            validation_result = ValidationResult(
                question_id=question.id,
                is_valid=is_valid,
                scores={"answer_completeness": completeness},
                reasons=[f"Answer completeness: {completeness:.2f}" + 
                         (f" - {reasons[1] if len(reasons) > 1 else ''}" if reasons else "")],
                suggested_improvements=result.get("suggested_improvements")
            )
            
            self.logger.debug(
                f"Completeness validation for q:{question.id}: "
                f"completeness={completeness:.2f}, valid={is_valid}"
            )
            
            return validation_result
            
        except Exception as e:
            raise ValidationError(f"Answer completeness validation failed: {str(e)}")


class QuestionClarityValidator(BaseValidator):
    """
    Validator that checks if questions are clear and unambiguous.
    
    This validator assesses whether the question is well-formulated
    and has a clear intent.
    """
    
    def __init__(self, 
                task_router: TaskRouter, 
                prompt_manager: PromptManager,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the question clarity validator.
        
        Args:
            task_router: Task router for LLM calls.
            prompt_manager: Prompt manager for validation prompts.
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.threshold = self.config.get('threshold', 0.7)
    
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """
        Validate that the question is clear and unambiguous.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            ValidationResult indicating question clarity.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Get the validation service
            service = self.task_router.get_service("validation")
            
            # Use the system's validate_question method
            result = await service.validate_question(question, chunk)
            
            # Extract the clarity score
            clarity = result.get("question_clarity", 0.0)
            is_valid = clarity >= self.threshold
            
            # Extract reasons
            reasons = result.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
                
            # Create validation result
            validation_result = ValidationResult(
                question_id=question.id,
                is_valid=is_valid,
                scores={"question_clarity": clarity},
                reasons=[f"Question clarity: {clarity:.2f}" + 
                         (f" - {reasons[2] if len(reasons) > 2 else ''}" if reasons else "")],
                suggested_improvements=result.get("suggested_improvements")
            )
            
            self.logger.debug(
                f"Clarity validation for q:{question.id}: "
                f"clarity={clarity:.2f}, valid={is_valid}"
            )
            
            return validation_result
            
        except Exception as e:
            raise ValidationError(f"Question clarity validation failed: {str(e)}")
