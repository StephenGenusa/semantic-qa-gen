# src/semantic_qa_gen/question/validation/factual.py

"""Scored-dimension validators for question validation.

A single shared base (`_ScoredDimensionValidator`) holds the logic that was
previously copy-pasted across three validators (resolving the long-standing
"DRY VIOLATIONS ACROSS ALL THREE VALIDATIONS" TODO). Each concrete validator is
now a thin subclass that declares which key it reads from the pre-computed LLM
payload.

Phase 0 note on the payload:
  The ValidationEngine now makes TWO LLM calls per question and merges them into
  one dict:
    - faithfulness call (WITH source) -> "factual_accuracy", "answer_completeness"
    - standalone call   (WITHOUT source) -> "standalone"
  Each validator below reads exactly one of those keys, so it does not matter to
  the validator which call produced it.
"""

from typing import Any, Dict, Optional, Union

from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.config.schema import BaseValidatorConfig


class _ScoredDimensionValidator(BaseValidator):
    """Base for validators that read a single {'score', 'reason'} dimension
    from pre-computed LLM validation data and threshold it.

    Subclasses set two class attributes:
        dimension_key: the key to read from llm_validation_data.
        score_label:   the key to publish in the ValidationResult.scores dict.
    """

    dimension_key: str = ""
    score_label: str = ""

    def __init__(self, config: Optional[Union[Dict[str, Any], BaseValidatorConfig]] = None):
        super().__init__(config)
        # The engine passes validator configs in as dicts (via model_dump()),
        # but support the typed model too. Keep the 0.7 default these validators
        # have always used (BaseValidator's own default is 0.6).
        if isinstance(self.config, BaseValidatorConfig):
            self.threshold = self.config.threshold
        elif isinstance(self.config, dict):
            self.threshold = self.config.get("threshold", 0.7)

    async def validate(
        self,
        question: Question,
        chunk: Chunk,
        llm_validation_data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        validator_name = self.name

        if llm_validation_data is None:
            self.logger.warning(
                f"{validator_name}: No LLM validation data for Q:{question.id}. Marking invalid."
            )
            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=False,
                scores={self.score_label: 0.0},
                reasons=["Missing shared LLM validation data"],
            )

        try:
            # `... or {}` handles both missing-key and explicit-null cases.
            dim_data = llm_validation_data.get(self.dimension_key) or {}
            if not isinstance(dim_data, dict):
                raise ValueError(
                    f"Expected '{self.dimension_key}' to be a dict with score/reason; "
                    f"got {type(dim_data).__name__}"
                )

            raw_score = dim_data.get("score", 0.0)
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"{validator_name}: Invalid score type ('{raw_score}') for "
                    f"Q:{question.id}. Defaulting to 0.0."
                )
                score = 0.0

            is_valid = score >= self.threshold
            llm_reason = dim_data.get("reason")
            reason_text = f"{self.score_label} score: {score:.2f} (Threshold: {self.threshold})"
            if not is_valid:
                reason_text += f" - Reason: {llm_reason or 'Score below threshold'}"
            elif llm_reason:
                reason_text += f" - LLM Note: {llm_reason}"

            result = ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=is_valid,
                scores={self.score_label: score},
                reasons=[reason_text],
                suggested_improvements=llm_validation_data.get("suggested_improvements"),
            )
            self.logger.debug(
                f"{validator_name} check for Q:{question.id}: score={score:.2f}, valid={is_valid}"
            )
            return result

        except Exception as e:
            self.logger.exception(
                f"{validator_name}: Error during internal logic for Q:{question.id}: {e}",
                exc_info=True,
            )
            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=False,
                scores={},
                reasons=[f"Internal Validator Error: {str(e)}"],
            )


class FactualAccuracyValidator(_ScoredDimensionValidator):
    """Answer is supported by the source text. Reads the faithfulness call."""
    dimension_key = "factual_accuracy"
    score_label = "factual_accuracy"


class AnswerCompletenessValidator(_ScoredDimensionValidator):
    """Answer fully addresses the question. Reads the faithfulness call."""
    dimension_key = "answer_completeness"
    score_label = "answer_completeness"


class StandaloneValidator(_ScoredDimensionValidator):
    """Question/answer are understandable and answerable without the source.

    Reads the "standalone" key produced by the source-free standalone_validation
    call. This replaces the old QuestionClarityValidator, which scored
    self-containment while the judge could see the source — a judgment that call
    structurally could not make.
    """
    dimension_key = "standalone"
    score_label = "standalone"


# Backwards-compatibility alias. The behavior now reflects the source-free
# standalone check rather than the old source-contaminated clarity check.
QuestionClarityValidator = StandaloneValidator