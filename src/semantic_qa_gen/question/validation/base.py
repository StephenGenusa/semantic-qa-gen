# filename: semantic_qa_gen/question/validation/base.py

"""Base validator interface and ValidationResult model."""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List

# Use Pydantic V2 imports
from pydantic import BaseModel, Field

from semantic_qa_gen.document.models import Question, Chunk # Keep these dependencies
from semantic_qa_gen.utils.error import ValidationError # Keep custom error

# --- Pydantic V2 ValidationResult Model ---
class ValidationResult(BaseModel):
    """Stores the outcome of validating a single Question by a specific validator."""
    question_id: str = Field(..., description="ID of the question being validated.")
    validator_name: Optional[str] = Field(None, description="Name of the validator producing this result.")
    is_valid: bool = Field(..., description="Validity outcome from this specific validator.")
    # Score(s) produced by this specific validator
    scores: Dict[str, float] = Field(default_factory=dict, description="Dictionary of scores from this validator (e.g., {'factual_accuracy': 0.9}).")
    # Reason(s) provided by this specific validator
    reasons: List[str] = Field(default_factory=list, description="List of reasons supporting the validity decision (especially for failure).")
    # Optional suggestions from this specific validator
    suggested_improvements: Optional[str] = Field(None, description="Optional textual suggestions for improving the question/answer from this validator.")

    model_config = { # Replaces Config class
        "validate_assignment": True
    }

    def __bool__(self) -> bool:
        """Allows treating the result directly as a boolean (True if valid)."""
        return self.is_valid

    def __str__(self) -> str:
        """Provides a concise string representation."""
        status = "Valid" if self.is_valid else "Invalid"
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items()) if self.scores else "N/A"
        # Include validator name for clarity
        name_prefix = f"({self.validator_name}) " if self.validator_name else ""
        reason_str = f": {'; '.join(self.reasons)}" if self.reasons else ""
        return f"Q:{self.question_id} {name_prefix}-> {status} (Scores: [{score_str}]{reason_str})"


# --- Base Validator Abstract Class ---
class BaseValidator(ABC):
    """
    Abstract base class for question validators using Pydantic V2.

    Validators check question-answer pairs against specific criteria.
    LLM-based validators may receive pre-computed data obtained from a
    single LLM call managed by the ValidationEngine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.

        Args:
            config: Optional configuration dictionary (from the specific validator's
                    section in the main config, e.g., validation.factual_accuracy).
        """
        # Use Pydantic V2 BaseValidatorConfig or specific child for validation?
        # Simpler: pass dict and let validator use it.
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.6) # Default threshold
        self.enabled = self.config.get('enabled', True)
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

        if not self.enabled:
            self.logger.debug(f"Validator '{self.name}' is disabled via configuration.")

    @abstractmethod
    async def validate(self,
                       question: Question,
                       chunk: Chunk,
                       llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a question-answer pair against specific criteria.

        Args:
            question: The Question object to validate.
            chunk: The source Chunk context.
            llm_validation_data: Optional dictionary containing pre-fetched results
                                 from a shared LLM validation call (used only by
                                 validators marked as requiring LLM in ValidationEngine).

        Returns:
            A ValidationResult object detailing the outcome of this specific validator.

        Raises:
            ValidationError: If the validation logic itself encounters an unrecoverable error.
                             Should NOT raise for simply failing validation (return is_valid=False).
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this validator is enabled via configuration."""
        return self.enabled

