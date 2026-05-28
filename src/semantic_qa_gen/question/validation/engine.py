# filename: semantic_qa_gen/question/validation/engine.py

import logging
import asyncio
import json
import re
import datetime
from typing import Dict, Any, Optional, List, Type, Tuple

# Project Imports
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.config.schema import ValidationConfig
from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.question.validation.factual import (
    FactualAccuracyValidator, AnswerCompletenessValidator, QuestionClarityValidator
)
from semantic_qa_gen.question.validation.diversity import DiversityValidator
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService # Keep import
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError, ConfigurationError


# Dimension keys expected in the LLM validation response. Centralized so the
# parser and any future consumer agree on the contract.
_VALIDATION_DIMENSIONS = ("factual_accuracy", "answer_completeness", "question_clarity")


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
        self.config: ValidationConfig = config_manager.get_section("validation")
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        self.validators: Dict[str, BaseValidator] = {}
        self.llm_dependent_validators: List[str] = []
        self._initialize_validators()

    def _initialize_validators(self) -> None:
        """Initialize standard validators based on config."""
        validator_map: Dict[str, Tuple[Type[BaseValidator], Any]] = {
            "factual_accuracy": (FactualAccuracyValidator, self.config.factual_accuracy),
            "answer_completeness": (AnswerCompletenessValidator, self.config.answer_completeness),
            "question_clarity": (QuestionClarityValidator, self.config.question_clarity),
            "diversity": (DiversityValidator, self.config.diversity),
        }
        llm_dependent_names = ["factual_accuracy", "answer_completeness", "question_clarity"]
        for name, (validator_cls, validator_config) in validator_map.items():
             if validator_config.enabled:
                 try:
                     instance = validator_cls(validator_config.model_dump())
                     self.validators[name] = instance
                     if name in llm_dependent_names: self.llm_dependent_validators.append(name)
                     self.logger.debug(f"Initialized validator: {name}")
                 except Exception as e: self.logger.error(f"Failed init validator '{name}': {e}", exc_info=True)
             else: self.logger.info(f"Validator '{name}' disabled.")
        self.logger.info(f"Initialized {len(self.validators)} validators. LLM-dependent: {len(self.llm_dependent_validators)}")

    def register_validator(self, name: str, validator: BaseValidator, requires_llm: bool = False) -> None:
        """Register a custom validator instance."""
        name_lower = name.lower()
        if name_lower in self.validators: self.logger.warning(f"Overwriting validator: {name_lower}")
        if not isinstance(validator, BaseValidator): raise TypeError("Validator must be BaseValidator instance.")
        self.validators[name_lower] = validator
        if requires_llm and name_lower not in self.llm_dependent_validators: self.llm_dependent_validators.append(name_lower)
        self.logger.info(f"Registered custom validator: {name_lower} (Requires LLM: {requires_llm})")

    async def _get_shared_llm_validation_data(self, question: Question, chunk: Chunk) -> Optional[Dict[str, Any]]:
        """Makes the single LLM call needed by factual/completeness/clarity validators."""
        task_name = "validation"
        prompt_key = "question_validation"
        try:
             # 1. Get the LLMTaskService
             llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)

             # Use the renamed field task_model_config
             self.logger.debug(f"Making shared LLM validation call for Q:{question.id} using "
                              f"adapter {type(llm_service.adapter).__name__} and model {llm_service.task_model_config.name}")

             # 2. Format the prompt
             prompt_vars = {"chunk_content": chunk.content, "question_text": question.text, "answer_text": question.answer}
             formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
             expects_json = llm_service.prompt_manager.is_json_output(prompt_key)

             # 3. Call adapter's generic completion method
             response_text = await llm_service.adapter.generate_completion(
                 prompt=formatted_prompt,
                 model_config=llm_service.task_model_config
             )

             # 4. Parse the response
             validation_data = self._parse_validation_response(response_text, expects_json, question.id)

             self.logger.debug(f"Received and parsed LLM validation data for Q:{question.id}")
             return validation_data

        except (LLMServiceError, ConfigurationError) as e:
             self.logger.error(f"Shared LLM validation call failed for Q:{question.id}: {e}")
             return None
        except ValidationError as e:
             self.logger.error(f"Failed to parse LLM validation response for Q:{question.id}: {e}")
             return None
        except Exception as e:
             self.logger.exception(f"Unexpected error during shared LLM validation call for Q:{question.id}", exc_info=True)
             return None

    def _parse_validation_response(self, response_text: str, expected_json: bool, question_id: str) -> Dict[str, Any]:
        """
        Parses the LLM validation response, expecting the per-dimension nested
        shape produced by the question_validation prompt:

            {
              "factual_accuracy":     {"score": float, "reason": str},
              "answer_completeness":  {"score": float, "reason": str},
              "question_clarity":     {"score": float, "reason": str},
              "suggested_improvements": str | null
            }

        Args:
            response_text: Raw response string.
            expected_json: Whether prompt requested JSON.
            question_id: ID for logging.

        Returns:
            Parsed and normalized dictionary with the nested-dimension shape.
            Each dimension is a dict with float 'score' and optional str 'reason'.

        Raises:
            ValidationError: If parsing fails, the result is not a dict, or
                the required dimension structure is missing.
        """
        response_text = response_text.strip()
        if not response_text: raise ValidationError(f"LLM returned empty validation response for Q:{question_id}.")
        if not expected_json: raise ValidationError(f"Validation prompt did not specify JSON output for Q:{question_id}.")
        parsed_data: Optional[Dict] = None; source = "unknown"
        try: # Direct JSON
             if response_text.startswith('{') and response_text.endswith('}'): parsed_data = json.loads(response_text); source = "direct json object"
        except json.JSONDecodeError: pass
        if parsed_data is None: # Code block JSON
            code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
            if code_block_match:
                 try: parsed_data = json.loads(code_block_match.group(1).strip()); source = "extracted block"
                 except json.JSONDecodeError: pass
                 if not isinstance(parsed_data, dict): parsed_data = None
        if parsed_data is None: # YAML
            try:
                 import yaml; parsed_yaml = yaml.safe_load(response_text)
                 if isinstance(parsed_yaml, dict): parsed_data = parsed_yaml; source = source + "+yaml" if source != "unknown" else "yaml"
            except (ImportError, yaml.YAMLError): pass
        if parsed_data is None or not isinstance(parsed_data, dict): raise ValidationError(f"Could not parse validation dictionary for Q:{question_id}.", {"preview": response_text[:200]})

        # Shape check: each dimension must be present and must be a dict containing 'score'.
        # No backward-compatibility path: the old flat shape ('factual_accuracy': 0.85)
        # is rejected here rather than silently coerced, so a mismatch surfaces immediately.
        missing = [d for d in _VALIDATION_DIMENSIONS if d not in parsed_data]
        if missing:
            raise ValidationError(
                f"Validation response for Q:{question_id} missing required dimension(s): {missing}",
                {"preview": response_text[:200]}
            )
        non_dict = [d for d in _VALIDATION_DIMENSIONS if not isinstance(parsed_data.get(d), dict)]
        if non_dict:
            raise ValidationError(
                f"Validation response for Q:{question_id} has non-dict dimension(s) {non_dict}; "
                f"expected per-dimension {{'score': float, 'reason': str}} structure.",
                {"preview": response_text[:200]}
            )

        # Normalize each dimension: coerce score to float, reason to str (or None).
        for dim in _VALIDATION_DIMENSIONS:
            dim_data = parsed_data[dim]
            try:
                dim_data['score'] = float(dim_data.get('score'))
            except (ValueError, TypeError, SystemError):
                dim_data['score'] = 0.0
            raw_reason = dim_data.get('reason')
            dim_data['reason'] = str(raw_reason).strip() if raw_reason else None

        # Normalize suggested_improvements (still emitted at top level by the prompt).
        suggestions = parsed_data.get('suggested_improvements')
        parsed_data['suggested_improvements'] = str(suggestions).strip() if suggestions else None
        return parsed_data

    async def validate_single_question(self,
                                       question: Question,
                                       chunk: Chunk,
                                       shared_llm_data: Optional[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """
        Run all enabled NON-DIVERSITY validators for a single question in parallel.

        NOTE: The diversity validator is intentionally excluded here because it has
        per-chunk state (cache of accepted question texts) and must be run sequentially
        across questions to function correctly. Diversity is handled in a separate
        sequential pass inside `validate_questions`.
        """
        individual_results: Dict[str, ValidationResult] = {}
        validation_tasks = []
        enabled_validator_names = []

        for name, validator in self.validators.items():
            if not validator.is_enabled():
                continue
            # Skip diversity here — it runs sequentially in validate_questions
            if name == "diversity":
                continue

            enabled_validator_names.append(name)
            is_llm_dep = name in self.llm_dependent_validators
            if is_llm_dep and shared_llm_data is None:
                individual_results[name] = ValidationResult(
                    question_id=question.id, validator_name=name, is_valid=False,
                    scores={}, reasons=["Shared LLM validation call failed"]
                )
                continue
            llm_data = shared_llm_data if is_llm_dep else None
            validation_tasks.append(
                asyncio.create_task(
                    validator.validate(question, chunk, llm_data),
                    name=f"validate_{question.id}_{name}"
                )
            )

        if validation_tasks:
            name_map = {i: name for i, name in enumerate(enabled_validator_names) if name not in individual_results}
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            for i, res_or_err in enumerate(results):
                v_name = name_map.get(i)
                if not v_name:
                    self.logger.error(f"Mapping error index {i} for Q:{question.id}")
                    continue
                if isinstance(res_or_err, ValidationResult):
                    individual_results[v_name] = res_or_err
                elif isinstance(res_or_err, Exception):
                    self.logger.error(f"Validator '{v_name}' failed Q:{question.id}: {res_or_err}")
                    individual_results[v_name] = ValidationResult(
                        question_id=question.id, validator_name=v_name, is_valid=False,
                        scores={}, reasons=[f"Internal Validator Error: {str(res_or_err)}"]
                    )
                else:
                    self.logger.error(f"Validator '{v_name}' Q:{question.id} returned type {type(res_or_err)}")
                    individual_results[v_name] = ValidationResult(
                        question_id=question.id, validator_name=v_name, is_valid=False,
                        scores={}, reasons=["Unexpected result type"]
                    )

        return individual_results

    def _aggregate_validation_results(self,
                                      question_id: str,
                                      individual_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Aggregates results from individual validators into a final verdict."""
        overall_valid = True; scores = {}; reasons = []; suggestions = []
        if not individual_results: return {"question_id": question_id, "is_valid": False, "scores": {}, "reasons": ["No validators ran."], "suggested_improvements": None, "validation_results": {}}
        for name, result in individual_results.items():
            overall_valid &= result.is_valid; scores.update(result.scores)
            reasons.extend([f"({name}) {r}" for r in result.reasons])
            if result.suggested_improvements: suggestions.append(f"({name}) {result.suggested_improvements}")
        return {"question_id": question_id, "is_valid": overall_valid, "scores": scores, "reasons": reasons, "suggested_improvements": "\n".join(suggestions) if suggestions else None, "validation_results": individual_results}

    async def validate_questions(self, questions: List[Question],
                                 chunk: Chunk) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple questions against a chunk.

        Two-phase design:
        1. Run all non-diversity validators (factual_accuracy, answer_completeness,
           question_clarity) concurrently across questions via gather.
        2. Run diversity sequentially across questions, in input order, so its
           per-chunk cache of accepted questions is built up correctly. Diversity
           is cheap (no LLM call, just SequenceMatcher) so serialization is fine.
        """
        final_results: Dict[str, Dict[str, Any]] = {}

        # Reset diversity validator state for this chunk before any work
        diversity_val = self.validators.get("diversity")
        diversity_enabled = isinstance(diversity_val, DiversityValidator) and diversity_val.is_enabled()
        if diversity_enabled:
            diversity_val.reset_for_chunk(chunk.id)

        needs_llm = any(
            self.validators.get(name) and self.validators[name].is_enabled()
            for name in self.llm_dependent_validators
        )

        # --- Phase 1: parallel non-diversity validation ---
        # Each workflow does: shared LLM call → run non-diversity validators in parallel
        coros = [self._validate_question_workflow(q, chunk, needs_llm) for q in questions]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Materialize phase-1 results into final_results and also keep the per-question
        # individual_results dict so we can merge diversity into it in phase 2.
        per_question_individual: Dict[str, Dict[str, ValidationResult]] = {}
        for i, res_or_err in enumerate(results):
            q_id = questions[i].id
            if isinstance(res_or_err, Exception):
                self.logger.error(f"Validation workflow failed Q_ID {q_id}: {res_or_err}")
                final_results[q_id] = {
                    "question_id": q_id, "is_valid": False, "scores": {},
                    "reasons": [f"Workflow Error: {str(res_or_err)}"],
                    "suggested_improvements": None, "validation_results": {}
                }
                per_question_individual[q_id] = {}
            elif isinstance(res_or_err, dict):
                final_results[q_id] = res_or_err
                per_question_individual[q_id] = res_or_err.get("validation_results", {}) or {}
            else:
                self.logger.error(f"Unexpected workflow return type Q_ID {q_id}: {type(res_or_err)}")
                final_results[q_id] = {
                    "question_id": q_id, "is_valid": False, "scores": {},
                    "reasons": ["Workflow Error: Unexpected type"],
                    "suggested_improvements": None, "validation_results": {}
                }
                per_question_individual[q_id] = {}

        # --- Phase 2: sequential diversity validation ---
        if diversity_enabled:
            for q in questions:
                q_id = q.id
                try:
                    diversity_result = await diversity_val.validate(q, chunk, llm_validation_data=None)
                except Exception as e:
                    self.logger.error(f"Diversity validator failed Q_ID {q_id}: {e}")
                    diversity_result = ValidationResult(
                        question_id=q_id, validator_name="diversity", is_valid=False,
                        scores={}, reasons=[f"Internal Validator Error: {str(e)}"]
                    )

                # Merge diversity into this question's individual_results and re-aggregate
                individual = per_question_individual.get(q_id, {})
                individual["diversity"] = diversity_result
                per_question_individual[q_id] = individual

                # Re-aggregate now that diversity is included
                aggregated = self._aggregate_validation_results(q_id, individual)

                # Preserve metadata that _validate_question_workflow set on the question;
                # only update the aggregated verdict fields.
                final_results[q_id] = aggregated

                # Re-write the validation metadata on the Question object so it reflects
                # the post-diversity verdict (mirrors what _validate_question_workflow does).
                try:
                    if not q.metadata:
                        q.metadata = {}
                    validation_metadata = {
                        'validation_is_valid': aggregated['is_valid'],
                        'validation_scores': aggregated['scores'],
                        'validation_reasons': aggregated['reasons'],
                    }
                    if aggregated['suggested_improvements']:
                        validation_metadata['validation_suggestions'] = aggregated['suggested_improvements']
                    validator_details = {}
                    for vname, vres in aggregated['validation_results'].items():
                        if isinstance(vres, ValidationResult):
                            validator_details[vname] = {
                                'is_valid': vres.is_valid,
                                'scores': vres.scores,
                                'reasons': vres.reasons,
                            }
                    if validator_details:
                        validation_metadata['validator_details'] = validator_details
                    validation_metadata['validation_timestamp'] = datetime.datetime.now(
                        datetime.timezone.utc).isoformat()
                    q.metadata['validation'] = validation_metadata
                except Exception as meta_err:
                    self.logger.warning(
                        f"Could not update validation metadata after diversity for Q:{q_id}: {meta_err}")

        return final_results

    async def _validate_question_workflow(self, question: Question, chunk: Chunk, needs_llm_call: bool) -> Dict[str, Any]:
        """Helper coroutine to manage LLM call and validation for one question, adding validation results to metadata."""
        shared_llm_data = await self._get_shared_llm_validation_data(question, chunk) if needs_llm_call else None
        individual_results = await self.validate_single_question(question, chunk, shared_llm_data)

        # Get the aggregated validation result
        result = self._aggregate_validation_results(question.id, individual_results)

        # Add validation results to question metadata for fine-tuning
        try:
            if not question.metadata:
                question.metadata = {}

            # Store validation results in question metadata
            validation_metadata = {
                'validation_is_valid': result['is_valid'],
                'validation_scores': result['scores'],
                'validation_reasons': result['reasons']
            }

            if result['suggested_improvements']:
                validation_metadata['validation_suggestions'] = result['suggested_improvements']

            # Store detailed validator results
            validator_details = {}
            for validator_name, validator_result in result['validation_results'].items():
                if isinstance(validator_result, ValidationResult):
                    validator_details[validator_name] = {
                        'is_valid': validator_result.is_valid,
                        'scores': validator_result.scores,
                        'reasons': validator_result.reasons
                    }

            if validator_details:
                validation_metadata['validator_details'] = validator_details

            # Add validation timestamp
            validation_metadata['validation_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            question.metadata['validation'] = validation_metadata

        except Exception as e:
            self.logger.warning(f"Could not add validation metadata to question {question.id}: {e}")

        return result

    def get_valid_questions(self, questions: List[Question],
                          aggregated_results: Dict[str, Dict[str, Any]]) -> List[Question]:
        """Filter questions based on the aggregated validation results."""
        return [q for q in questions if aggregated_results.get(q.id, {}).get("is_valid", False)]
