# filename: semantic_qa_gen/question/generator.py

import logging
import uuid
import asyncio
import json
import re  # Import re for parsing
from typing import Dict, Any, Optional, List, Tuple
from pydantic import ValidationError, TypeAdapter # Import pydantic tools

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
# Stick with ValidationError as used in the original file for consistency
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError, ConfigurationError

# Pydantic TypeAdapter for robust list validation
DictListValidator = TypeAdapter(List[Dict[str, Any]])

class QuestionGenerator:
    """
    Generator for creating questions based on document chunks.

    Gets the required LLM adapter and model config via the TaskRouter,
    formats the prompt, calls the adapter's generate_completion,
    parses the response, and creates Question objects.
    """

    def __init__(self, config_manager: ConfigManager,
                task_router: TaskRouter,
                prompt_manager: PromptManager): # PromptManager is now essential
        """
        Initialize the question generator.

        Args:
            config_manager: Configuration manager instance.
            task_router: Task router for getting LLM services for tasks.
            prompt_manager: Prompt manager instance.
        """
        self.config_manager = config_manager
        try:
            self.config = config_manager.get_section("question_generation")
            if not self.config:
                 raise AttributeError
        except AttributeError:
            raise ConfigurationError("Configuration section 'question_generation' is missing or invalid.")

        self.task_router = task_router
        self.prompt_manager = prompt_manager # Store prompt manager
        self.logger = logging.getLogger(__name__)

    async def generate_questions(self,
                               chunk: Chunk,
                               analysis: AnalysisResult) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis results.

        Formats the prompt, calls the appropriate LLM adapter via TaskRouter,
        parses the response, and creates Question objects.

        Args:
            chunk: The document chunk to generate questions for.
            analysis: The pre-computed analysis result for the chunk.

        Returns:
            A list of generated Question objects.

        Raises:
            ValidationError: If generation configuration is invalid or if the
                             underlying LLM call or parsing fails.
        """
        task_name = "generation"
        # Adjust 'generate_questions' if using a different default key in prompts.yaml
        prompt_key = "question_generation"
        response_text: Optional[str] = None # Initialize response_text

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

            # 1. Get the LLMTaskService (adapter, model_config, prompt_manager)
            try:
                llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)
            except LLMServiceError as e:
                 raise ValidationError(f"Failed to get LLM service for generation task: {e}") from e

            self.logger.debug(f"Using adapter {type(llm_service.adapter).__name__} "
                             # Use task_model_config as defined in Router/LLMTaskService
                             f"with model config {llm_service.task_model_config.name} for generation.")

            # 2. Format the prompt using the PromptManager from the service
            prompt_vars = {
                "chunk_content": chunk.content,
                "total_questions": total_questions_to_generate, # Use calculated total
                "factual_count": category_counts.get("factual", 0),
                "inferential_count": category_counts.get("inferential", 0),
                "conceptual_count": category_counts.get("conceptual", 0),
                "key_concepts": ", ".join(analysis.key_concepts) if analysis and analysis.key_concepts else "N/A",
                # Pass full analysis dict if needed by the prompt
                "analysis_details": json.dumps(analysis.model_dump(), indent=2) if analysis else "N/A"
            }
            formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
            expects_json = llm_service.prompt_manager.is_json_output(prompt_key)

            # 3. Call adapter's generic completion method
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted_prompt,
                # Use task_model_config from the service bundle
                model_config=llm_service.task_model_config
            )

            if not response_text:
                 self.logger.warning(f"LLM returned empty response during generation for chunk {chunk.id}.")
                 raise ValidationError("LLM returned empty response during question generation.", details={"chunk_id": chunk.id})

            # 4. Parse the response (expected JSON list for questions)
            #    _parse_question_response now includes debug logging on failure
            parsed_response = self._parse_question_response(response_text, chunk.id, expected_json=expects_json) # Pass chunk_id

            # 5. Convert parsed data to Question objects
            questions = self._create_questions_from_parsed(parsed_response, chunk.id)

            self.logger.info(f"Successfully parsed and created {len(questions)} questions for chunk {chunk.id}")

            return questions

        # Catch specific errors and wrap them appropriately
        except (LLMServiceError, ConfigurationError) as e:
            self.logger.error(f"LLM/Config error during question generation for chunk {chunk.id}: {e}", exc_info=self.config_manager.config.processing.debug_mode)
            # Log raw response if available
            if response_text:
                 self.logger.error(f"[DEBUG] Raw LLM response during LLM/Config error for chunk {chunk.id}:\n>>>\n{response_text}\n<<<")
            raise ValidationError(f"LLM or Config error during question generation: {str(e)}") from e
        except ValidationError as e: # Catch parsing/validation errors raised internally
             # Error should have been logged with details inside _parse_question_response if it originated there
            self.logger.error(f"Validation error during question generation for chunk {chunk.id}: {e}")
            raise # Re-raise specific validation errors
        except Exception as e:
            self.logger.exception(f"Unexpected error during question generation for chunk {chunk.id}: {e}", exc_info=True)
            # Log raw response if available
            if response_text:
                 self.logger.error(f"[DEBUG] Raw LLM response during unexpected exception for chunk {chunk.id}:\n>>>\n{response_text}\n<<<")
            raise ValidationError(f"An unexpected error occurred during question generation: {str(e)}") from e


    def _parse_question_response(self, response_text: str, chunk_id: str, expected_json: bool) -> List[Dict[str, Any]]:
        """
        Parses the LLM response expected to contain a list of questions, adding debug logging on failure.

        Args:
            response_text: The raw string response from the LLM.
            chunk_id: The ID of the chunk (for logging).
            expected_json: Whether the prompt specifically requested JSON.

        Returns:
            A list of dictionaries, each representing a question item.

        Raises:
            ValidationError: If parsing fails or response is not a list.
        """
        response_text = response_text.strip()
        if not response_text:
            raise ValidationError("LLM returned an empty response for question generation.", details={"chunk_id": chunk_id})

        # Even if not required by prompt meta, we often *expect* JSON list for questions based on instructions
        # So, we'll proceed with parsing attempts, but 'expected_json' could inform which attempt is primary.
        if not expected_json:
            self.logger.debug("Prompt metadata did not strictly require JSON, but attempting JSON list parse for questions.")

        parsed_list: Optional[List[Dict[str, Any]]] = None

        # Attempt 1: Direct JSON parse (expecting a list)
        try:
            # Add basic check - might help avoid large non-JSON string parsing
            if response_text.startswith('[') and response_text.endswith(']'):
                # Use TypeAdapter for better structure validation
                parsed_list = DictListValidator.validate_json(response_text)
                self.logger.debug(f"Successfully parsed direct JSON list for chunk {chunk_id}.")
                # Clean items: ensure they are dictionaries before returning
                cleaned_list = [item for item in parsed_list if isinstance(item, dict)]
                if len(cleaned_list) != len(parsed_list):
                    self.logger.warning(f"Filtered out {len(parsed_list) - len(cleaned_list)} non-dictionary items from direct JSON list for chunk {chunk_id}.")
                return cleaned_list # Return early on success
            else:
                 self.logger.debug("Response doesn't look like a direct JSON list. Trying code block extraction.")

        except (json.JSONDecodeError, ValidationError) as e:
            # Only log a warning here, we'll try extracting from code block next
            self.logger.warning(f"Direct JSON list parsing failed for chunk {chunk_id}: {e}. Trying code block extraction.")
            # --- Add Debug Logging Here on Initial Failure ---
            self.logger.error(
                f"[DEBUG] Raw LLM response that failed initial JSON parsing for chunk {chunk_id}:\n"
                f">>>\n{response_text}\n<<<"
            )
            # --- End Debug Logging ---
            pass # Continue to next attempt

        # Attempt 2: Extract JSON from Markdown code blocks ```json ... ```
        # This regex handles optional 'json' label and potential leading/trailing whitespace
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            code_block_text = match.group(1).strip()
            try:
                # Try parsing the extracted content using TypeAdapter
                parsed_list = DictListValidator.validate_json(code_block_text)
                self.logger.debug(f"Successfully parsed JSON list from code block for chunk {chunk_id}.")
                # Clean items: ensure they are dictionaries before returning
                cleaned_list = [item for item in parsed_list if isinstance(item, dict)]
                if len(cleaned_list) != len(parsed_list):
                    self.logger.warning(f"Filtered out {len(parsed_list) - len(cleaned_list)} non-dictionary items from extracted JSON list for chunk {chunk_id}.")
                return cleaned_list # Return early on success
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"Failed to parse JSON list even after extracting from code block for chunk {chunk_id}: {e}")
                # Log the extracted block for debugging
                self.logger.error(f"[DEBUG] Extracted code block that failed parsing for chunk {chunk_id}:\n>>>\n{code_block_text}\n<<<")
                # Raise the final error here, as both attempts failed
                raise ValidationError(
                    "Could not parse LLM question generation response as JSON list (from code block).",
                    details={"chunk_id": chunk_id, "error": str(e)}
                ) from e

        # Attempt 3: If no JSON found, maybe it's just Q/A pairs directly (less ideal)
        # This is a fallback and depends heavily on the LLM consistently failing JSON
        # We might try simple Q/A regex pair extraction if other methods fail.
        # For now, we will declare failure if JSON list isn't found/parsed.
        # --- (Optional: Add regex for Q:/A: pairs here as a last resort if needed) ---

        # Final check and raise if parsing failed
        if parsed_list is None: # This check redundant if Attempt 2 raises on failure, but kept for clarity
            self.logger.error(f"Failed to find or parse LLM question response as JSON list after all attempts for chunk {chunk_id}.")
            # Log the full raw response again as a last resort for debugging
            self.logger.error(
                 f"[DEBUG] Final raw LLM response that could not be parsed as JSON list for chunk {chunk_id}:\n"
                 f">>>\n{response_text}\n<<<"
             )
            raise ValidationError(
                "Could not parse LLM question generation response as a JSON list (no valid list found).",
                details={"chunk_id": chunk_id, "response_preview": response_text[:500]}
            )

        # Fallback return - should not be reached if Attempt 1 or 2 succeeds or Attempt 2 raises.
        return []


    def _create_questions_from_parsed(self, parsed_list: List[Dict[str, Any]], chunk_id: str) -> List[Question]:
        """Converts parsed dictionary items into Question objects."""
        questions: List[Question] = []
        valid_categories = {"factual", "inferential", "conceptual", "implementation", "architectural", "troubleshooting"} # Expand valid categories

        for i, item in enumerate(parsed_list):
            try:
                # Accommodate prompts that might use 'Q:'/'A:' directly in the dict key
                question_text = item.get("question", item.get("Q"))
                answer_text = item.get("answer", item.get("A"))
                category_raw = item.get("category", "unknown") # Get category if provided
                category = str(category_raw).lower().strip()

                if not question_text or not answer_text or not isinstance(question_text, str) or not isinstance(answer_text, str):
                    self.logger.warning(f"Skipping generated item {i} for chunk {chunk_id}: Invalid or missing 'question'/'answer' text.")
                    continue

                question_text = question_text.strip()
                answer_text = answer_text.strip()
                if not question_text or not answer_text:
                     self.logger.warning(f"Skipping generated item {i} for chunk {chunk_id}: Empty 'question' or 'answer' after stripping.")
                     continue

                # Normalize category
                if category not in valid_categories:
                    self.logger.warning(f"Generated item {i} for chunk {chunk_id} has unknown category '{category_raw}'. Setting to 'unknown'.")
                    category = "unknown" # Standardize unknown category

                questions.append(Question(
                    # id=str(uuid.uuid4()), # Let Pydantic handle default ID generation if defined in model
                    text=question_text,
                    answer=answer_text,
                    chunk_id=chunk_id,
                    category=category,
                    # Add metadata if needed, ensuring it's a dict
                    metadata=item.get('metadata', {}) if isinstance(item.get('metadata', {}), dict) else {'generation_order': i}
                ))
            except Exception as item_err:
                self.logger.error(f"Error creating Question object from item {i} for chunk {chunk_id}: {item_err}. Item: {item}", exc_info=False)

        return questions


    # _calculate_question_counts remains the same as provided previously
    def _calculate_question_counts(self, analysis: AnalysisResult) -> Dict[str, int]:
        """
        Calculate how many questions to generate for each category based on configuration and analysis.

        Args:
            analysis: AnalysisResult containing estimated yield and density.

        Returns:
            Dictionary mapping category names ('factual', 'inferential', 'conceptual')
            to the number of questions to generate for that category.
        """
        # Default empty dict if config section or categories are missing
        category_config_section = getattr(self.config, 'categories', {})
        if not isinstance(category_config_section, dict):
            self.logger.warning("Question generation 'categories' config is not a dictionary. Using defaults.")
            category_config_section = {}

        max_questions = getattr(self.config, 'max_questions_per_chunk', 10)
        adaptive_enabled = getattr(self.config, 'adaptive_generation', True)
        # Safe access to analysis nested attributes
        density = getattr(analysis, 'information_density', 0.5) if analysis else 0.5
        yield_estimates = getattr(analysis, 'estimated_question_yield', {}) if analysis and isinstance(getattr(analysis, 'estimated_question_yield', None), dict) else {}

        # Try to determine valid categories from the config structure
        valid_categories_from_config = set(category_config_section.keys())
        if not valid_categories_from_config:
             # Fallback if config is empty, use standard categories
             valid_categories_from_config = {"factual", "inferential", "conceptual"}
             self.logger.debug("No categories defined in config, using default set: factual, inferential, conceptual")

        # 1. Start with minimums defined in config for valid categories
        counts = {}
        for cat in valid_categories_from_config:
            cat_cfg = category_config_section.get(cat, {}) # Get specific category config (or empty dict)
            if not isinstance(cat_cfg, dict): # Handle case where category value isn't a dict
                 min_q = 0
            else:
                 min_q = cat_cfg.get('min_questions', 0)
            counts[cat] = int(max(0, min_q)) # Ensure non-negative int


        # 2. Consider yield estimates from analysis (if higher than min)
        for cat, estimated_yield in yield_estimates.items():
            if cat in counts:
                try:
                    counts[cat] = max(counts[cat], int(estimated_yield))
                except (ValueError, TypeError):
                     self.logger.warning(f"Invalid yield estimate '{estimated_yield}' for category '{cat}'. Ignoring.")


        # 3. Apply adaptive scaling if enabled
        if adaptive_enabled:
            density_factor = max(0.5, min(2.0, 1.0 + (density - 0.5))) # Clamp density factor
            for cat in counts:
                cat_cfg = category_config_section.get(cat, {})
                if not isinstance(cat_cfg, dict):
                     weight = 1.0
                     min_q = 0
                else:
                     weight = cat_cfg.get('weight', 1.0)
                     min_q = cat_cfg.get('min_questions', 0)

                # Ensure numeric types before calculation
                try:
                    current_count = float(counts[cat])
                    weight = float(weight)
                    adjusted_count = current_count * density_factor * weight
                    counts[cat] = max(int(min_q), int(round(adjusted_count))) # Max with min_q, ensure int
                except (ValueError, TypeError):
                     self.logger.warning(f"Invalid numeric value encountered during adaptive scaling for category '{cat}'. Skipping adaptive scaling for this category.")
                     # Keep the count from step 2 or ensure it meets min_q
                     counts[cat] = max(int(min_q), int(counts.get(cat, 0)))


        # 4. Enforce total max questions per chunk
        current_total = sum(counts.values())
        if current_total > max_questions:
            if current_total > 0: # Avoid division by zero
                scale_factor = max_questions / current_total
                scaled_counts = {}
                # Apply scaling, respecting minimums
                for cat, count in counts.items():
                    cat_cfg = category_config_section.get(cat, {})
                    min_q = int(cat_cfg.get('min_questions', 0)) if isinstance(cat_cfg, dict) else 0
                    scaled_counts[cat] = max(min_q, int(round(count * scale_factor)))
                counts = scaled_counts

                final_total = sum(counts.values())
                excess = final_total - max_questions

                # If rounding up pushed sum over max, trim excess starting from lowest weight/count
                if excess > 0:
                    cats_to_trim = sorted(
                        counts.keys(),
                        key=lambda k: (
                            float(category_config_section.get(k, {}).get('weight', 1.0)) if isinstance(category_config_section.get(k), dict) else 1.0, # Sort by weight first (ascending)
                            counts[k], # Then by current count (ascending)
                            k # Then alphabetically (for tie-breaking)
                        )
                    )
                    for cat in cats_to_trim:
                        cat_cfg = category_config_section.get(cat, {})
                        min_q = int(cat_cfg.get('min_questions', 0)) if isinstance(cat_cfg, dict) else 0
                        can_trim = counts[cat] - min_q
                        trim_amount = min(excess, can_trim)
                        if trim_amount > 0:
                            counts[cat] -= trim_amount
                            excess -= trim_amount
                        if excess <= 0:
                            break

        # Final cleanup: ensure all counts are non-negative integers
        counts = {cat: max(0, int(count)) for cat, count in counts.items()}

        # Log final calculated counts (use analysis.chunk_id if available)
        chunk_id_log = getattr(analysis, 'chunk_id', 'unknown')
        self.logger.debug(
            f"Calculated question counts for chunk {chunk_id_log}: {counts} (Max: {max_questions}, Adaptive: {adaptive_enabled}, Density: {density:.2f})"
        )
        return counts

