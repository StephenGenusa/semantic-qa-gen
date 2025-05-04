# filename: semantic_qa_gen/question/validation/factual.py

"""Factual accuracy, completeness, and clarity validators for question validation."""

import asyncio
from typing import Dict, Any, Optional, List

# Import Question/Chunk models
from semantic_qa_gen.document.models import Question, Chunk
# Import new base validator and result model
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.utils.error import ValidationError


class FactualAccuracyValidator(BaseValidator):
    """
    Validator checking answer factual accuracy based on source text via LLM results.

    Relies on pre-computed validation data provided by the ValidationEngine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the validator."""
        super().__init__(config)
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
            score = 0.0
            raw_score = llm_validation_data.get("factual_accuracy", 0.0)
            try:
                 score = float(raw_score)
            except (ValueError, TypeError):
                  self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                  score = 0.0

            is_valid = score >= self.threshold

            # Extract relevant reason if available, otherwise generic reason
            all_reasons = llm_validation_data.get("reasons", [])
            reason_text = f"Accuracy score: {score:.2f} (Threshold: {self.threshold})"
            # Look for keywords in reasons provided by LLM validation call
            keywords = ["accuracy", "factual", "correct", "incorrect", "source"]
            relevant_llm_reason = next((r for r in all_reasons if any(k in r.lower() for k in keywords)), None)

            if not is_valid:
                 reason_text += f" - Reason: {relevant_llm_reason or 'Score below threshold'}"
            elif relevant_llm_reason: # Add LLM reason even if valid, if found
                 reason_text += f" - LLM Note: {relevant_llm_reason}"


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
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.7)

    async def validate(self, question: Question, chunk: Chunk, llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate answer completeness using pre-computed LLM results."""
        validator_name = self.name
        if llm_validation_data is None:
            self.logger.warning(f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid.")
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={"answer_completeness": 0.0}, reasons=["Missing shared LLM validation data"])

        try:
            score = 0.0
            raw_score = llm_validation_data.get("answer_completeness", 0.0)
            try: score = float(raw_score)
            except (ValueError, TypeError):
                  self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                  score = 0.0

            is_valid = score >= self.threshold

            all_reasons = llm_validation_data.get("reasons", [])
            reason_text = f"Completeness score: {score:.2f} (Threshold: {self.threshold})"
            keywords = ["complete", "address", "cover", "missing", "sufficient"]
            relevant_llm_reason = next((r for r in all_reasons if any(k in r.lower() for k in keywords)), None)

            if not is_valid: reason_text += f" - Reason: {relevant_llm_reason or 'Score below threshold'}"
            elif relevant_llm_reason: reason_text += f" - LLM Note: {relevant_llm_reason}"

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
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.7)

    async def validate(self, question: Question, chunk: Chunk, llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate question clarity using pre-computed LLM results."""
        validator_name = self.name
        if llm_validation_data is None:
            self.logger.warning(f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid.")
            return ValidationResult(question_id=question.id, validator_name=validator_name, is_valid=False, scores={"question_clarity": 0.0}, reasons=["Missing shared LLM validation data"])

        try:
            score = 0.0
            raw_score = llm_validation_data.get("question_clarity", 0.0)
            try: score = float(raw_score)
            except (ValueError, TypeError):
                  self.logger.warning(f"{validator_name}: Invalid score type ('{raw_score}') for Q:{question.id}. Defaulting to 0.0.")
                  score = 0.0

            is_valid = score >= self.threshold

            all_reasons = llm_validation_data.get("reasons", [])
            reason_text = f"Clarity score: {score:.2f} (Threshold: {self.threshold})"
            keywords = ["clear", "ambiguous", "confusing", "understand", "vague"]
            relevant_llm_reason = next((r for r in all_reasons if any(k in r.lower() for k in keywords)), None)

            if not is_valid: reason_text += f" - Reason: {relevant_llm_reason or 'Score below threshold'}"
            elif relevant_llm_reason: reason_text += f" - LLM Note: {relevant_llm_reason}"

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

