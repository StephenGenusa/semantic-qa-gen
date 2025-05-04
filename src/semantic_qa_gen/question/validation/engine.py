# filename: semantic_qa_gen/question/validation/engine.py

"""Validation engine coordinating multiple validators using Pydantic V2 Models."""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Type, Tuple

# Project Imports
from semantic_qa_gen.config.manager import ConfigManager
# Use Pydantic V2 base model and schema
from semantic_qa_gen.config.schema import ValidationConfig
from semantic_qa_gen.document.models import Question, Chunk
# Use new BaseValidator and ValidationResult
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
# Import specific validators
from semantic_qa_gen.question.validation.factual import (
    FactualAccuracyValidator,
    AnswerCompletenessValidator,
    QuestionClarityValidator
)
from semantic_qa_gen.question.validation.diversity import DiversityValidator
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError, ConfigurationError


class ValidationEngine:
    """
    Coordinates multiple validators efficiently for question validation.

    Manages the validation workflow, including:
    - Initializing validators based on configuration.
    - Making a single shared LLM call for relevant validators.
    - Distributing LLM results and invoking individual validators.
    - Resetting stateful validators (like DiversityValidator) per chunk.
    - Aggregating results from all enabled validators.
    """

    def __init__(self, config_manager: ConfigManager,
                 task_router: TaskRouter,
                 prompt_manager: PromptManager):
        """Initializes the ValidationEngine."""
        self.config_manager = config_manager
        # Get the validated ValidationConfig section
        self.config: ValidationConfig = config_manager.get_section("validation")
        self.task_router = task_router
        self.prompt_manager = prompt_manager # Needed by adapters called via task_router
        self.logger = logging.getLogger(__name__)

        self.validators: Dict[str, BaseValidator] = {}
        # Track which validators use the shared LLM call result
        self.llm_dependent_validators: List[str] = []
        self._initialize_validators()

    def _initialize_validators(self) -> None:
        """Initialize standard validators based on config."""
        validator_map: Dict[str, Tuple[Type[BaseValidator], Any]] = { # Maps name to (Class, ConfigModel)
            "factual_accuracy": (FactualAccuracyValidator, self.config.factual_accuracy),
            "answer_completeness": (AnswerCompletenessValidator, self.config.answer_completeness),
            "question_clarity": (QuestionClarityValidator, self.config.question_clarity),
            "diversity": (DiversityValidator, self.config.diversity),
        }
        llm_dependent_names = ["factual_accuracy", "answer_completeness", "question_clarity"]

        for name, (validator_cls, validator_config) in validator_map.items():
             if validator_config.enabled:
                try:
                    # Pass config dict to validator constructor
                    instance = validator_cls(validator_config.model_dump())
                    self.validators[name] = instance
                    if name in llm_dependent_names:
                         self.llm_dependent_validators.append(name)
                    self.logger.debug(f"Initialized validator: {name}")
                except Exception as e:
                     # Catch individual validator init errors
                     self.logger.error(f"Failed to initialize validator '{name}': {e}", exc_info=True)
             else:
                 self.logger.info(f"Validator '{name}' disabled by configuration.")

        self.logger.info(f"Initialized {len(self.validators)} validators. "
                        f"LLM-dependent: {len(self.llm_dependent_validators)}")


    def register_validator(self, name: str, validator: BaseValidator, requires_llm: bool = False) -> None:
        """Register a custom validator instance."""
        name_lower = name.lower()
        if name_lower in self.validators:
            self.logger.warning(f"Overwriting previously registered validator: {name_lower}")

        if not isinstance(validator, BaseValidator):
             raise TypeError("Registered validator must be instance of BaseValidator.")

        self.validators[name_lower] = validator
        if requires_llm and name_lower not in self.llm_dependent_validators:
            self.llm_dependent_validators.append(name_lower)
        self.logger.info(f"Registered custom validator: {name_lower} (Requires LLM: {requires_llm})")


    async def _get_shared_llm_validation_data(self, question: Question, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """Makes the single LLM call needed by factual/completeness/clarity validators."""
        try:
             validation_task_service: LLMTaskService = self.task_router.get_task_handler("validation")
             self.logger.debug(f"Making shared LLM validation call for Q:{question.id} using {type(validation_task_service.adapter).__name__}")
             # Adapter handles prompt formatting, LLM call, and parsing the JSON response
             llm_results = await validation_task_service.validate_question(question, chunk)
             self.logger.debug(f"Received LLM validation data for Q:{question.id}") #: {llm_results}") # Avoid logging potentially large data by default
             return llm_results
        except (LLMServiceError, ConfigurationError) as e:
             self.logger.error(f"Shared LLM validation call failed for Q:{question.id}: {e}")
             return None # Indicate failure
        except Exception as e:
             self.logger.exception(f"Unexpected error during shared LLM validation call for Q:{question.id}", exc_info=True)
             return None # Indicate failure


    async def validate_single_question(self,
                                     question: Question,
                                     chunk: Chunk,
                                     shared_llm_data: Optional[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """
        Run all enabled validators for a single question, using shared LLM data if provided.

        Args:
            question: The question to validate.
            chunk: The source chunk context.
            shared_llm_data: Pre-fetched data from the shared LLM validation call.

        Returns:
            Dictionary mapping validator names to their ValidationResult.
        """
        individual_results: Dict[str, ValidationResult] = {}

        validation_tasks = []
        enabled_validator_names = []

        # Create tasks for each enabled validator
        for name, validator in self.validators.items():
            if not validator.is_enabled():
                continue

            enabled_validator_names.append(name)
            is_llm_dependent = name in self.llm_dependent_validators

            # Determine if LLMdependent validator can run
            if is_llm_dependent and shared_llm_data is None:
                 # LLM call failed earlier, create a default invalid result immediately
                 self.logger.warning(f"Skipping LLM-dependent validator '{name}' for Q:{question.id} due to prior LLM call failure.")
                 individual_results[name] = ValidationResult(
                     question_id=question.id, validator_name=name, is_valid=False,
                     scores={}, reasons=["Shared LLM validation call failed"]
                 )
                 continue # Don't create an async task for this one

            # Pass the shared data only to LLM-dependent validators
            llm_data_for_validator = shared_llm_data if is_llm_dependent else None

            # Create the validation task
            validation_tasks.append(
                asyncio.create_task(
                    validator.validate(question, chunk, llm_data_for_validator),
                    name=f"validate_{question.id}_{name}"
                )
            )

        # Run validation tasks concurrently
        if validation_tasks:
            results_or_exceptions = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            processed_count = 0
            for i, res_or_err in enumerate(results_or_exceptions):
                # Need to map back to validator name - requires careful indexing or better task tracking
                validator_name = enabled_validator_names[i] # Assuming order is preserved

                if isinstance(res_or_err, ValidationResult):
                    individual_results[validator_name] = res_or_err
                elif isinstance(res_or_err, Exception):
                    self.logger.error(f"Validator '{validator_name}' failed for Q:{question.id}: {res_or_err}", exc_info=isinstance(res_or_err, ValidationError))
                    # Create a basic invalid result indicating the validator error
                    individual_results[validator_name] = ValidationResult(
                        question_id=question.id, validator_name=validator_name, is_valid=False,
                        scores={}, reasons=[f"Internal Validator Error: {str(res_or_err)}"]
                    )
                else: # Should not happen if validators return ValidationReult
                     self.logger.error(f"Validator '{validator_name}' for Q:{question.id} returned unexpected type: {type(res_or_err)}")
                     individual_results[validator_name] = ValidationResult(
                         question_id=question.id, validator_name=validator_name, is_valid=False,
                         scores={}, reasons=["Validator returned unexpected result type"]
                     )

        return individual_results


    def _aggregate_validation_results(self,
                                      question_id: str,
                                      individual_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Aggregates results from individual validators into a final verdict."""
        overall_valid = True
        combined_scores = {}
        all_reasons = []
        all_suggestions = []

        if not individual_results: # Handle case where no validators ran
             return {
                 "question_id": question_id, "is_valid": False, "combined_score": 0.0,
                 "reasons": ["No validators were enabled or ran successfully."],
                 "suggested_improvements": None, "validation_results": {}
             }

        for name, result in individual_results.items():
            overall_valid &= result.is_valid
            # Prefix scores with validator name to avoid clashes? Or assume unique keys? Assume unique for now.
            combined_scores.update(result.scores)
            # Prefix reasons with validator name for clarity
            all_reasons.extend([f"({name}) {r}" for r in result.reasons])
            if result.suggested_improvements:
                all_suggestions.append(f"({name}) {result.suggested_improvements}")

        # Calculate average score (maybe less meaningful now with diverse score types?)
        # Let's just provide all scores.
        # avg_score = sum(combined_scores.values()) / len(combined_scores) if combined_scores else 0.0

        return {
            "question_id": question_id,
            "is_valid": overall_valid,
            # "combined_score": avg_score, # Removed avg score
            "scores": combined_scores, # Return dict of all scores
            "reasons": all_reasons,
            "suggested_improvements": "\n".join(all_suggestions) if all_suggestions else None,
            "validation_results": individual_results, # Keep per-validator details
        }


    async def validate_questions(self, questions: List[Question],
                               chunk: Chunk) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple questions against a chunk, optimizing LLM calls.

        Args:
            questions: List of questions to validate.
            chunk: The source chunk context.

        Returns:
            Dictionary mapping question IDs to their aggregated validation results.

        Raises:
            ValidationError: If the validation process encounters critical errors.
        """
        final_results: Dict[str, Dict[str, Any]] = {}
        needs_llm_call = any(
            validator.is_enabled()
            for name in self.llm_dependent_validators
            # Check if the validator exists first using get() and then call is_enabled()
            if (validator := self.validators.get(name)) is not None
        )

        # Reset stateful validators for this chunk run
        diversity_validator = self.validators.get("diversity")
        if isinstance(diversity_validator, DiversityValidator) and diversity_validator.is_enabled():
            diversity_validator.reset_for_chunk(chunk.id)
            self.logger.debug(f"Reset DiversityValidator state for chunk {chunk.id}")

        # Perform validations concurrently for all questions
        validation_coros = []
        for question in questions:
             # Need a way to run _get_shared_llm_validation_data ONCE per question if needed,
             # then pass that data to validate_single_question.
             validation_coros.append(self._validate_question_workflow(question, chunk, needs_llm_call))

        gathered_results = await asyncio.gather(*validation_coros, return_exceptions=True)

        # Collate results
        for i, res_or_err in enumerate(gathered_results):
            question_id = questions[i].id # Assumes order preserved
            if isinstance(res_or_err, Exception):
                 self.logger.error(f"Validation workflow failed for Q_ID {question_id}: {res_or_err}")
                 # Create a failed overall result entry
                 final_results[question_id] = {
                     "question_id": question_id, "is_valid": False, "scores":{},
                     "reasons": [f"Validation Workflow Error: {str(res_or_err)}"],
                     "suggested_improvements": None, "validation_results": {}
                 }
            elif isinstance(res_or_err, dict): # Should be the aggregated results dict
                 final_results[question_id] = res_or_err
            else: # Unexpected return type
                  self.logger.error(f"Validation workflow returned unexpected type for Q_ID {question_id}: {type(res_or_err)}")
                  final_results[question_id] = {
                      "question_id": question_id, "is_valid": False, "scores":{},
                      "reasons": [f"Validation Workflow Error: Unexpected return type {type(res_or_err)}"],
                      "suggested_improvements": None, "validation_results": {}
                  }


        return final_results


    async def _validate_question_workflow(self, question: Question, chunk: Chunk, needs_llm_call: bool) -> Dict[str, Any]:
        """Helper coroutine to manage LLM call and validation for one question."""
        shared_llm_data = None
        if needs_llm_call:
             shared_llm_data = await self._get_shared_llm_validation_data(question, chunk)
             # If LLM call failed, shared_llm_data will be None, and validators handle it

        # Run all enabled validators (LLM and non-LLM) using the potentially fetched data
        individual_results = await self.validate_single_question(question, chunk, shared_llm_data)

        # Aggregate results into the final structure
        aggregated_result = self._aggregate_validation_results(question.id, individual_results)
        return aggregated_result


    def get_valid_questions(self, questions: List[Question],
                          aggregated_results: Dict[str, Dict[str, Any]]) -> List[Question]:
        """Filter questions based on the aggregated validation results."""
        valid_questions = []
        for q in questions:
            result_for_q = aggregated_results.get(q.id)
            # Check the overall 'is_valid' flag in the aggregated result dict
            if result_for_q and result_for_q.get("is_valid", False):
                valid_questions.append(q)
        return valid_questions

