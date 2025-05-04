import logging
import uuid
import asyncio
import json # Only needed potentially by the adapter, not directly here
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService # Import LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError


class QuestionGenerator:
    """
    Generator for creating questions based on document chunks.

    Delegates the LLM interaction to the appropriate adapter via the
    LLMTaskService provided by the TaskRouter.
    """

    def __init__(self, config_manager: ConfigManager,
                task_router: TaskRouter,
                prompt_manager: Optional[PromptManager] = None):
        """
        Initialize the question generator.

        Args:
            config_manager: Configuration manager instance.
            task_router: Task router for getting LLM services for tasks.
            prompt_manager: Optional prompt manager instance. If provided, passed to adapter if needed; otherwise ignored.
        """
        self.config_manager = config_manager
        try:
            self.config = config_manager.get_section("question_generation")
            if not self.config:
                 raise AttributeError
        except AttributeError:
            raise ValidationError("Configuration section 'question_generation' is missing or invalid.")

        self.task_router = task_router
        # PromptManager is primarily used by the adapter now. Store if needed for other logic?
        # For now, generator doesn't need it directly.
        # self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)

    async def generate_questions(self,
                               chunk: Chunk,
                               analysis: AnalysisResult) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis results.

        Uses the TaskRouter to get the LLM service for generation and calls
        its generate_questions method.

        Args:
            chunk: The document chunk to generate questions for.
            analysis: The pre-computed analysis result for the chunk.

        Returns:
            A list of generated Question objects.

        Raises:
            ValidationError: If generation configuration is invalid or if the
                             underlying LLM call fails.
        """
        try:
            # Determine how many questions to generate based on config & analysis
            category_counts = self._calculate_question_counts(analysis)
            total_questions_to_generate = sum(category_counts.values())

            if total_questions_to_generate <= 0:
                self.logger.info(f"Skipping question generation for chunk {chunk.id}: Calculated question count is 0.")
                return []

            self.logger.info(
                f"Requesting generation of {total_questions_to_generate} questions for chunk {chunk.id} "
                f"with category distribution: {category_counts}"
            )

            # 1. Get the LLM task handler for generation
            try:
                llm_task_service: LLMTaskService = self.task_router.get_task_handler("generation")
            except LLMServiceError as e:
                 raise ValidationError(f"Failed to get LLM service for generation task: {e}") from e

            self.logger.debug(f"Using adapter {type(llm_task_service.adapter).__name__} "
                             f"with model config {llm_task_service.model_config.name} for generation.")

            # 2. Delegate the generation call to the LLMTaskService wrapper
            # This wrapper calls the adapter's generate_questions method, which handles
            # prompt formatting and the generate_completion LLM call.
            generated_questions: List[Question] = await llm_task_service.generate_questions(
                chunk=chunk,
                analysis=analysis, # Pass analysis if the adapter needs it (e.g., for context)
                category_counts=category_counts # Pass the calculated counts
                # model_config is handled inside the LLMTaskService
            )

            self.logger.info(f"Adapter generated {len(generated_questions)} questions for chunk {chunk.id}")

            # --- Optional: Post-generation filtering or adjustments ---
            # e.g., Ensure categories match request, enforce max limit strictly if needed
            # For now, rely on the LLM adhering to the category counts requested in the prompt

            return generated_questions

        except LLMServiceError as e:
            self.logger.error(f"LLM service error during question generation for chunk {chunk.id}: {e}", exc_info=self.config_manager.config.processing.debug_mode)
            # Re-raise wrapped in ValidationError to indicate failure at this stage
            raise ValidationError(f"LLM service failed during question generation: {str(e)}") from e
        except ValidationError as e: # Catch our own explicit errors
            self.logger.error(f"Validation error during question generation setup for chunk {chunk.id}: {e}")
            raise # Re-raise specific validation errors
        except Exception as e:
            self.logger.exception(f"Unexpected error during question generation for chunk {chunk.id}: {e}", exc_info=True)
            raise ValidationError(f"An unexpected error occurred during question generation: {str(e)}") from e

    # _calculate_question_counts method remains the same (no changes needed)
    def _calculate_question_counts(self, analysis: AnalysisResult) -> Dict[str, int]:
        """
        Calculate how many questions to generate for each category based on configuration and analysis.

        Args:
            analysis: AnalysisResult containing estimated yield and density.

        Returns:
            Dictionary mapping category names ('factual', 'inferential', 'conceptual')
            to the number of questions to generate for that category.
        """
        category_config = getattr(self.config, 'categories', {})
        max_questions = getattr(self.config, 'max_questions_per_chunk', 10)
        adaptive_enabled = getattr(self.config, 'adaptive_generation', True)
        density = getattr(analysis, 'information_density', 0.5)  # Default density if missing

        # Extract valid categories from model fields accounting for Pydantic V2 structure
        valid_categories_from_config = set()

        # Check if we have Pydantic V2 model_fields
        model_fields = {}
        if hasattr(self.config, 'model_fields'):
            model_fields = self.config.model_fields
        elif hasattr(self.config, '__fields__'):  # Fallback for Pydantic V1
            model_fields = self.config.__fields__

        # Extract valid categories using Pydantic V2 compatible approach
        categories_field = model_fields.get('categories')
        if categories_field:
            # For Pydantic V2 - the annotation structure is different
            try:
                # Use get_type_hints or check annotation
                annotation = categories_field.annotation
                # If it's a Dict type
                if hasattr(annotation, '__origin__') and annotation.__origin__ == dict:
                    # Get the second type argument (dict value type)
                    value_type = annotation.__args__[1] if len(annotation.__args__) > 1 else None
                    if value_type and hasattr(value_type, 'model_fields'):
                        # It's likely a Pydantic model for category config
                        valid_categories_from_config = set(category_config.keys())
                    else:
                        # It's a simple dict with string keys
                        valid_categories_from_config = set(category_config.keys())
            except Exception:
                self.logger.debug("Could not extract type information from Pydantic model, using keys directly.")
                valid_categories_from_config = set(category_config.keys())
        else:
            # Fallback to category_config keys if no field definition found
            valid_categories_from_config = set(category_config.keys())

        # 1. Start with minimums defined in config
        counts = {cat: cfg.min_questions for cat, cfg in category_config.items() if cat in valid_categories_from_config}

        # Rest of the method remains the same
        # 2. Consider yield estimates from analysis (if higher than min)
        if hasattr(analysis, 'estimated_question_yield') and analysis.estimated_question_yield:
            for cat, estimated_yield in analysis.estimated_question_yield.items():
                if cat in counts:
                    # Use the higher of the configured minimum or the LLM's estimate
                    counts[cat] = max(counts[cat], estimated_yield)
                # Ignore categories from analysis not in config

        # 3. Apply adaptive scaling if enabled
        if adaptive_enabled:
            # Adjust counts based on density. Factor can be tuned.
            density_factor = 1.0 + (density - 0.5)  # e.g., 0.8 density -> factor 1.3; 0.3 density -> factor 0.8
            density_factor = max(0.5, min(2.0, density_factor))  # Clamp factor

            for cat in counts:
                # Get weight safely from config
                cat_cfg = category_config.get(cat)
                weight = getattr(cat_cfg, 'weight', 1.0) if cat_cfg else 1.0
                # Apply density factor and category weight
                adjusted_count = counts[cat] * density_factor * weight
                # Keep at least the minimum, round to nearest int
                min_q = getattr(cat_cfg, 'min_questions', 0) if cat_cfg else 0
                counts[cat] = max(min_q, round(adjusted_count))

        # 4. Enforce total max questions per chunk
        current_total = sum(counts.values())
        if current_total > max_questions:
            if current_total > 0:  # Avoid division by zero
                scale_factor = max_questions / current_total
                scaled_counts = {}
                for cat, count in counts.items():
                    cat_cfg = category_config.get(cat)
                    min_q = getattr(cat_cfg, 'min_questions', 0) if cat_cfg else 0
                    scaled_counts[cat] = max(min_q, int(round(count * scale_factor)))
                counts = scaled_counts

                # Final check: if rounding still exceeds max, trim from lowest weight/count cats
                final_total = sum(counts.values())
                excess = final_total - max_questions
                if excess > 0:
                    cats_to_trim = sorted(counts.keys(), key=lambda k: (
                        getattr(category_config.get(k), 'weight', 1.0), counts[k], k)
                                          )
                    for cat in cats_to_trim:
                        cat_cfg = category_config.get(cat)
                        min_q = getattr(cat_cfg, 'min_questions', 0) if cat_cfg else 0
                        can_trim = counts[cat] - min_q  # How much can we trim from this cat?
                        trim_amount = min(excess, can_trim)
                        if trim_amount > 0:
                            counts[cat] -= trim_amount
                            excess -= trim_amount
                        if excess <= 0:
                            break

        # Ensure final counts are non-negative integers
        counts = {cat: max(0, int(count)) for cat, count in counts.items()}

        self.logger.debug(
            f"Calculated question counts for chunk {analysis.chunk_id}: {counts} (Max: {max_questions}, Adaptive: {adaptive_enabled}, Density: {density:.2f})")
        return counts

