# filename: semantic_qa_gen/question/generator.py

import logging
import uuid
import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError, ConfigurationError


class QuestionGenerator:
    """
    Generator for creating questions based on document chunks.

    Gets the required LLM adapter and model config via the TaskRouter,
    formats the prompt, calls the adapter's generate_completion,
    parses the response, and creates Question objects.
    """

    def __init__(self, config_manager: ConfigManager,
                task_router: TaskRouter,
                prompt_manager: PromptManager):
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
        self.prompt_manager = prompt_manager
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
        prompt_key = "question_generation"
        try:
            category_counts = self._calculate_question_counts(analysis)
            total_questions_to_generate = sum(category_counts.values())
            if total_questions_to_generate <= 0:
                self.logger.info(f"Skipping question generation for chunk {chunk.id}: Calculated question count is 0.")
                return []

            self.logger.info(f"Requesting generation of {total_questions_to_generate} questions for chunk {chunk.id}...")

            # 1. Get the LLMTaskService
            try:
                llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)
            except LLMServiceError as e:
                 raise ValidationError(f"Failed to get LLM service for generation task: {e}") from e

            # Use the renamed field task_model_config
            self.logger.debug(f"Using adapter {type(llm_service.adapter).__name__} "
                             f"with model config {llm_service.task_model_config.name} for generation.")

            # 2. Format the prompt
            prompt_vars = {
                "chunk_content": chunk.content,
                "total_questions": total_questions_to_generate,
                "factual_count": category_counts.get("factual", 0),
                "inferential_count": category_counts.get("inferential", 0),
                "conceptual_count": category_counts.get("conceptual", 0),
                "key_concepts": ", ".join(analysis.key_concepts) if analysis and analysis.key_concepts else "N/A",
            }
            formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
            expects_json = llm_service.prompt_manager.is_json_output(prompt_key)

            # 3. Call adapter's generic completion method
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted_prompt,
                model_config=llm_service.task_model_config
            )

            # 4. Parse the response
            parsed_response = self._parse_question_response(response_text, expected_json=expects_json)

            # 5. Convert parsed data to Question objects
            questions = self._create_questions_from_parsed(parsed_response, chunk.id)

            self.logger.info(f"Successfully parsed and created {len(questions)} questions for chunk {chunk.id}")
            return questions

        except (LLMServiceError, ConfigurationError) as e:
            self.logger.error(f"Error during question generation for chunk {chunk.id}: {e}", exc_info=self.config_manager.config.processing.debug_mode)
            raise ValidationError(f"LLM or Config error during question generation: {str(e)}") from e
        except ValidationError as e:
            self.logger.error(f"Validation error during question generation for chunk {chunk.id}: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error during question generation for chunk {chunk.id}: {e}", exc_info=True)
            raise ValidationError(f"An unexpected error occurred during question generation: {str(e)}") from e

    def _parse_question_response(self, response_text: str, expected_json: bool) -> List[Dict[str, Any]]:
        """
        Parses the LLM response expected to contain a list of questions.

        Args:
            response_text: The raw string response from the LLM.
            expected_json: Whether the prompt specifically requested JSON.

        Returns:
            A list of dictionaries, each representing a question item.

        Raises:
            ValidationError: If parsing fails or response is not a list.
        """
        response_text = response_text.strip()
        if not response_text: raise ValidationError("LLM returned empty response for question generation.")
        if not expected_json: raise ValidationError("Response parsing error: Expected JSON output but flag was false.")
        parsed_data: Optional[List] = None; source = "unknown"
        try: # Direct parse
             if response_text.startswith('[') and response_text.endswith(']'): parsed_data = json.loads(response_text); source = "direct json array"
             if not isinstance(parsed_data, list): parsed_data = None
        except json.JSONDecodeError: pass
        if parsed_data is None: # Code block parse
            code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
            if code_block_match:
                 try: parsed_data = json.loads(code_block_match.group(1).strip()); source = "extracted json block"
                 except json.JSONDecodeError: pass
                 if not isinstance(parsed_data, list): parsed_data = None
        if parsed_data is None: # YAML parse
             try:
                  import yaml; parsed_yaml = yaml.safe_load(response_text)
                  if isinstance(parsed_yaml, list): parsed_data = parsed_yaml; source = "direct yaml parse"
             except (ImportError, yaml.YAMLError): pass
        if parsed_data is None or not isinstance(parsed_data, list): raise ValidationError("Could not parse LLM question generation response as a JSON list.", details={"preview": response_text[:200]})
        cleaned_list = [item for item in parsed_data if isinstance(item, dict)]
        if len(cleaned_list) != len(parsed_data): self.logger.warning(f"Filtered {len(parsed_data) - len(cleaned_list)} non-dict items.")
        self.logger.debug(f"Parsed question response via {source} into {len(cleaned_list)} items.")
        return cleaned_list

    def _create_questions_from_parsed(self, parsed_list: List[Dict[str, Any]], chunk_id: str) -> List[Question]:
        """Converts parsed dictionary items into Question objects."""
        questions: List[Question] = []; valid_categories = {"factual", "inferential", "conceptual"}
        for i, item in enumerate(parsed_list):
            try:
                q_text = item.get("question"); a_text = item.get("answer"); cat_raw = item.get("category", "unknown"); cat = str(cat_raw).lower()
                if not q_text or not a_text: self.logger.warning(f"Skipping item {i}: Missing Q or A."); continue
                if cat not in valid_categories: self.logger.warning(f"Item {i}: Unknown category '{cat_raw}'. Setting 'unknown'."); cat = "unknown"
                questions.append(Question(id=str(uuid.uuid4()), text=str(q_text), answer=str(a_text), chunk_id=chunk_id, category=cat, metadata={'generation_order': i}))
            except Exception as item_err: self.logger.error(f"Error creating Question object {i}: {item_err}. Item: {item}")
        return questions

    def _calculate_question_counts(self, analysis: AnalysisResult) -> Dict[str, int]:
        """
        Calculate how many questions to generate for each category based on configuration and analysis.

        Args:
            analysis: AnalysisResult containing estimated yield and density.

        Returns:
            Dictionary mapping category names ('factual', 'inferential', 'conceptual')
            to the number of questions to generate for that category.
        """
        category_config = getattr(self.config, 'categories', {}); max_questions = getattr(self.config, 'max_questions_per_chunk', 10)
        adaptive_enabled = getattr(self.config, 'adaptive_generation', True); density = getattr(analysis, 'information_density', 0.5)
        valid_categories_from_config = set(category_config.keys()) # Simplified fallback
        counts = {cat: cfg.min_questions for cat, cfg in category_config.items() if cat in valid_categories_from_config}
        if hasattr(analysis, 'estimated_question_yield') and analysis.estimated_question_yield:
            for cat, est_yield in analysis.estimated_question_yield.items():
                 if cat in counts: counts[cat] = max(counts[cat], est_yield)
        if adaptive_enabled:
            density_factor = max(0.5, min(2.0, 1.0 + (density - 0.5)))
            for cat in counts:
                cat_cfg = category_config.get(cat); weight = getattr(cat_cfg, 'weight', 1.0) if cat_cfg else 1.0
                adj_count = counts[cat] * density_factor * weight; min_q = getattr(cat_cfg, 'min_questions', 0) if cat_cfg else 0
                counts[cat] = max(min_q, round(adj_count))
        current_total = sum(counts.values())
        if current_total > max_questions and current_total > 0:
            scale = max_questions / current_total; scaled_counts = {}
            for cat, count in counts.items():
                 min_q = getattr(category_config.get(cat), 'min_questions', 0)
                 scaled_counts[cat] = max(min_q, int(round(count * scale)))
            counts = scaled_counts
            excess = sum(counts.values()) - max_questions
            if excess > 0:
                 cats_to_trim = sorted(counts.keys(), key=lambda k: (getattr(category_config.get(k), 'weight', 1.0), counts[k], k))
                 for cat in cats_to_trim:
                       min_q = getattr(category_config.get(cat), 'min_questions', 0)
                       trim = min(excess, counts[cat] - min_q)
                       if trim > 0: counts[cat] -= trim; excess -= trim
                       if excess <= 0: break
        counts = {cat: max(0, int(count)) for cat, count in counts.items()}
        self.logger.debug(f"Calculated question counts for chunk {analysis.chunk_id}: {counts}")
        return counts
