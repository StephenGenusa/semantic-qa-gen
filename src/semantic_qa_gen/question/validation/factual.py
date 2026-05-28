# filename: semantic_qa_gen/question/validation/factual.py

# TODO: DRY VIOLATIONS ACROSS ALL THREE VALIDATIONS

"""Factual accuracy, completeness, and clarity validators for question validation."""

import asyncio
from typing import Dict, Any, Optional, List, Union

# Import Question/Chunk models
from semantic_qa_gen.document.models import Question, Chunk
# Import new base validator and result model
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
# Import Pydantic config models for type safety
from semantic_qa_gen.config.schema import (
    FactualValidatorConfig,
    CompletenessValidatorConfig,
    ClarityValidatorConfig
)


class FactualAccuracyValidator(BaseValidator):
    """
    Validator checking answer factual accuracy based on source text via LLM results.

    Relies on pre-computed validation data provided by the ValidationEngine.
    Expects the per-dimension nested shape produced by the question_validation
    prompt:
        {"factual_accuracy": {"score": float, "reason": str}, ...}
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], FactualValidatorConfig]] = None):
        """Initializes the validator with type-safe configuration."""
        super().__init__(config)
        # Type-safe threshold extraction to maintain specific default (0.7) for dict inputs
        if isinstance(self.config, FactualValidatorConfig):
            self.threshold = self.config.threshold
        elif isinstance(self.config, dict):
            self.threshold = self.config.get('threshold', 0.7)

    async def validate(self, question: Question, chunk: Chunk, llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate factual accuracy using pre-computed LLM results."""
        validator_name = self.name # Get name for result

        # Handle missing LLM data case first
        if llm_validation_data is None:
            self.logger.warning(f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid.")
            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=False,
                scores={"factual_accuracy": 0.0},
                reasons=["Missing shared LLM validation data"]
            )

        try:
            # Read the per-dimension nested object. `... or {}` handles both
            # missing-key and explicit-null cases.
            dim_data = llm_validation_data.get("factual_accuracy") or {}
            if not isinstance(dim_data, dict):
                raise ValueError(
                    f"Expected 'factual_accuracy' to be a dict with score/reason; "
                    f"got {type(dim_data).__name__}"
                )

            raw_score = dim_data.get("score", 0.0)
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                score = 0.0

            is_valid = score >= self.threshold

            # Per-dimension reason comes directly from the LLM under this dimension's key.
            # No more keyword scanning across a flat 'reasons' list.
            llm_reason = dim_data.get("reason")
            reason_text = f"Accuracy score: {score:.2f} (Threshold: {self.threshold})"

            if not is_valid:
                reason_text += f" - Reason: {llm_reason or 'Score below threshold'}"
            elif llm_reason: # Add LLM reason even if valid, if provided
                reason_text += f" - LLM Note: {llm_reason}"


            validation_result = ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=is_valid,
                scores={"factual_accuracy": score},
                reasons=[reason_text],
                # Pass through overall suggestions if they exist
                suggested_improvements=llm_validation_data.get("suggested_improvements")
            )

            self.logger.debug(
                f"{validator_name} check for Q:{question.id}: "
                f"score={score:.2f}, valid={is_valid}"
            )
            return validation_result

        except Exception as e:
            self.logger.exception(f"{validator_name}: Error during internal logic for Q:{question.id}: {e}", exc_info=True)
            # Return an invalid result indicating internal validator error
            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=False,
                scores={},
                reasons=[f"Internal Validator Error: {str(e)}"]
            )


class AnswerCompletenessValidator(BaseValidator):
    """
    Validator checking answer completeness based on source text via LLM results.
    Relies on pre-computed validation data provided by the ValidationEngine.
    Expects the per-dimension nested shape produced by the question_validation
    prompt:
        {"answer_completeness": {"score": float, "reason": str}, ...}
    """
    def __init__(self, config: Optional[Union[Dict[str, Any], CompletenessValidatorConfig]] = None):
        """Initializes the validator with type-safe configuration."""
        super().__init__(config)
        # Type-safe threshold extraction
        if isinstance(self.config, CompletenessValidatorConfig):
            self.threshold = self.config.threshold
        elif isinstance(self.config, dict):
            self.threshold = self.config.get('threshold', 0.7)

    async def validate(self, question: Question, chunk: Chunk, llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate answer completeness using pre-computed LLM results."""
        validator_name = self.name
        if llm_validation_data is None:
            self.logger.warning(f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid.")
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={"answer_completeness": 0.0}, reasons=["Missing shared LLM validation data"])

        try:
            dim_data = llm_validation_data.get("answer_completeness") or {}
            if not isinstance(dim_data, dict):
                raise ValueError(
                    f"Expected 'answer_completeness' to be a dict with score/reason; "
                    f"got {type(dim_data).__name__}"
                )

            raw_score = dim_data.get("score", 0.0)
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                score = 0.0

            is_valid = score >= self.threshold

            llm_reason = dim_data.get("reason")
            reason_text = f"Completeness score: {score:.2f} (Threshold: {self.threshold})"

            if not is_valid: reason_text += f" - Reason: {llm_reason or 'Score below threshold'}"
            elif llm_reason: reason_text += f" - LLM Note: {llm_reason}"

            validation_result = ValidationResult(
                question_id=question.id, validator_name=validator_name, is_valid=is_valid,
                scores={"answer_completeness": score}, reasons=[reason_text],
                suggested_improvements=llm_validation_data.get("suggested_improvements")
            )
            self.logger.debug(f"{validator_name} check for Q:{question.id}: score={score:.2f}, valid={is_valid}")
            return validation_result
        except Exception as e:
            self.logger.exception(f"{validator_name}: Error during internal logic for Q:{question.id}: {e}", exc_info=True)
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={}, reasons=[f"Internal Validator Error: {str(e)}"])


class QuestionClarityValidator(BaseValidator):
    """
    Validator checking question clarity and lack of ambiguity via LLM results.
    Relies on pre-computed validation data provided by the ValidationEngine.
    Expects the per-dimension nested shape produced by the question_validation
    prompt:
        {"question_clarity": {"score": float, "reason": str}, ...}
    """
    def __init__(self, config: Optional[Union[Dict[str, Any], ClarityValidatorConfig]] = None):
        """Initializes the validator with type-safe configuration."""
        super().__init__(config)
        # Type-safe threshold extraction
        if isinstance(self.config, ClarityValidatorConfig):
            self.threshold = self.config.threshold
        elif isinstance(self.config, dict):
            self.threshold = self.config.get('threshold', 0.7)

    async def validate(self, question: Question, chunk: Chunk, llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate question clarity using pre-computed LLM results."""
        validator_name = self.name
        if llm_validation_data is None:
            self.logger.warning(f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid.")
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={"question_clarity": 0.0}, reasons=["Missing shared LLM validation data"])

        try:
            dim_data = llm_validation_data.get("question_clarity") or {}
            if not isinstance(dim_data, dict):
                raise ValueError(
                    f"Expected 'question_clarity' to be a dict with score/reason; "
                    f"got {type(dim_data).__name__}"
                )

            raw_score = dim_data.get("score", 0.0)
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                score = 0.0

            is_valid = score >= self.threshold

            llm_reason = dim_data.get("reason")
            reason_text = f"Clarity score: {score:.2f} (Threshold: {self.threshold})"

            if not is_valid: reason_text += f" - Reason: {llm_reason or 'Score below threshold'}"
            elif llm_reason: reason_text += f" - LLM Note: {llm_reason}"

            validation_result = ValidationResult(
                question_id=question.id, validator_name=validator_name, is_valid=is_valid,
                scores={"question_clarity": score}, reasons=[reason_text],
                suggested_improvements=llm_validation_data.get("suggested_improvements")
            )
            self.logger.debug(f"{validator_name} check for Q:{question.id}: score={score:.2f}, valid={is_valid}")
            return validation_result
        except Exception as e:
            self.logger.exception(f"{validator_name}: Error during internal logic for Q:{question.id}: {e}", exc_info=True)
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={}, reasons=[f"Internal Validator Error: {str(e)}"])