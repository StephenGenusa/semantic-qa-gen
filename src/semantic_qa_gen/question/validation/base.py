"""Base validator interface for question validation."""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List

from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.utils.error import ValidationError


class ValidationResult:
    """
    Results of validating a question-answer pair.
    
    This class stores the validation outcome and metrics for a single question.
    """
    
    def __init__(self, 
                question_id: str, 
                is_valid: bool,
                scores: Dict[str, float],
                reasons: Optional[List[str]] = None,
                suggested_improvements: Optional[str] = None):
        """
        Initialize validation result.
        
        Args:
            question_id: ID of the validated question.
            is_valid: Whether the question-answer pair is valid.
            scores: Dictionary of validation scores by category.
            reasons: Optional list of reasons for validation result.
            suggested_improvements: Optional suggestions for improvement.
        """
        self.question_id = question_id
        self.is_valid = is_valid
        self.scores = scores
        self.reasons = reasons or []
        self.suggested_improvements = suggested_improvements
        
    def __bool__(self):
        """Convert validation result to boolean."""
        return self.is_valid
        
    def __str__(self):
        """String representation of validation result."""
        status = "Valid" if self.is_valid else "Invalid"
        reasons = f": {', '.join(self.reasons)}" if self.reasons else ""
        return f"{status} (Q:{self.question_id}){reasons}"


class BaseValidator(ABC):
    """
    Abstract base class for question validators.
    
    Validators check question-answer pairs against specific criteria
    and return validation results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.6)
        self.enabled = self.config.get('enabled', True)
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """
        Validate a question-answer pair against the source chunk.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            ValidationResult object.
            
        Raises:
            ValidationError: If validation fails.
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if this validator is enabled.
        
        Returns:
            True if the validator is enabled, False otherwise.
        """
        return self.enabled
