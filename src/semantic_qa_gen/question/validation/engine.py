"""Validation engine to coordinate multiple validators."""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Set

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.question.validation.factual import (
    FactualAccuracyValidator, 
    AnswerCompletenessValidator,
    QuestionClarityValidator
)
from semantic_qa_gen.question.validation.diversity import DiversityValidator
from semantic_qa_gen.llm.router import TaskRouter
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError


class ValidationEngine:
    """
    Engine for coordinating multiple validators.
    
    This class manages the validation process, applying multiple validators
    to question-answer pairs and aggregating the results.
    """
    
    def __init__(self, config_manager: ConfigManager, 
                task_router: TaskRouter, 
                prompt_manager: PromptManager):
        """
        Initialize the validation engine.
        
        Args:
            config_manager: Configuration manager.
            task_router: Task router for LLM calls.
            prompt_manager: Prompt manager for validation prompts.
        """
        self.config_manager = config_manager
        self.config = config_manager.get_section("validation")
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.validators: Dict[str, BaseValidator] = {}
        self._initialize_validators()
    
    def _initialize_validators(self) -> None:
        """Initialize standard validators based on configuration."""
        # Factual accuracy validator
        if self.config.factual_accuracy.enabled:
            self.validators["factual_accuracy"] = FactualAccuracyValidator(
                self.task_router,
                self.prompt_manager,
                self.config.factual_accuracy.dict()
            )
            
        # Answer completeness validator
        if self.config.answer_completeness.enabled:
            self.validators["answer_completeness"] = AnswerCompletenessValidator(
                self.task_router,
                self.prompt_manager,
                self.config.answer_completeness.dict()
            )
            
        # Question clarity validator
        if self.config.question_clarity.enabled:
            self.validators["question_clarity"] = QuestionClarityValidator(
                self.task_router,
                self.prompt_manager,
                self.config.question_clarity.dict()
            )
            
        # Diversity validator
        if self.config.diversity.enabled:
            self.validators["diversity"] = DiversityValidator(
                self.config.diversity.dict()
            )
    
    def register_validator(self, name: str, validator: BaseValidator) -> None:
        """
        Register a custom validator.
        
        Args:
            name: Name for the validator.
            validator: Validator instance.
        """
        self.validators[name] = validator
        self.logger.info(f"Registered validator: {name}")
    
    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """
        Validate a question-answer pair using all enabled validators.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            Dictionary with validation results.
            
        Raises:
            ValidationError: If validation fails.
        """
        results = {}
        all_valid = True
        combined_scores = {}
        all_reasons = []
        improvements = []
        
        try:
            # Run all enabled validators
            for name, validator in self.validators.items():
                if not validator.is_enabled():
                    continue
                    
                result = await validator.validate(question, chunk)
                results[name] = result
                
                # Update combined results
                all_valid = all_valid and result.is_valid
                combined_scores.update(result.scores)
                all_reasons.extend(result.reasons)
                
                if result.suggested_improvements:
                    improvements.append(result.suggested_improvements)
                    
                # If a validator fails, log the issue
                if not result.is_valid:
                    self.logger.info(
                        f"Validation '{name}' failed for question {question.id}: "
                        f"{'; '.join(result.reasons)}"
                    )
            
            # Calculate average score (if any)
            avg_score = sum(combined_scores.values()) / len(combined_scores) if combined_scores else 0.0
            
            return {
                "question_id": question.id,
                "is_valid": all_valid,
                "validation_results": results,
                "combined_score": avg_score,
                "reasons": all_reasons,
                "suggested_improvements": "\n".join(improvements) if improvements else None
            }
            
        except Exception as e:
            raise ValidationError(f"Failed to validate question {question.id}: {str(e)}")
    
    async def validate_questions(self, questions: List[Question], 
                              chunk: Chunk) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple questions against a chunk.
        
        Args:
            questions: List of questions to validate.
            chunk: The source chunk.
            
        Returns:
            Dictionary mapping question IDs to validation results.
            
        Raises:
            ValidationError: If validation fails.
        """
        results = {}
        
        # Reset the diversity validator for this chunk
        diversity_validator = self.validators.get("diversity")
        if diversity_validator and diversity_validator.is_enabled():
            diversity_validator.reset_for_chunk(chunk.id)
        
        # Create tasks for all questions
        tasks = []
        for question in questions:
            task = asyncio.create_task(self.validate_question(question, chunk))
            tasks.append((question.id, task))
        
        # Process results as they complete
        for question_id, task in tasks:
            try:
                result = await task
                results[question_id] = result
            except Exception as e:
                self.logger.error(f"Failed to validate question {question_id}: {str(e)}")
                # Add a failed result
                results[question_id] = {
                    "question_id": question_id,
                    "is_valid": False,
                    "combined_score": 0.0,
                    "reasons": [f"Validation error: {str(e)}"],
                    "suggested_improvements": None
                }
                
        return results
    
    def get_valid_questions(self, questions: List[Question], 
                          validation_results: Dict[str, Dict[str, Any]]) -> List[Question]:
        """
        Filter a list of questions to include only valid ones.
        
        Args:
            questions: List of questions.
            validation_results: Validation results for the questions.
            
        Returns:
            List of valid questions.
        """
        return [q for q in questions if validation_results.get(q.id, {}).get("is_valid", False)]
