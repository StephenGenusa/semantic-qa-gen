# semantic_qa_gen/question/generator.py

import logging
import datetime
import uuid
import asyncio
import json
import re
import sys
import traceback
from typing import Dict, Any, Optional, List, Tuple
from pydantic import ValidationError, TypeAdapter

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError, ConfigurationError

# Pydantic TypeAdapter for robust list validation
DictListValidator = TypeAdapter(List[Dict[str, Any]])


class QuestionGenerator:
    """
    Generator for creating questions based on document chunks.
    """

    def __init__(self, config_manager: ConfigManager,
                 task_router: TaskRouter,
                 prompt_manager: PromptManager):
        """Initialize the question generator."""
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
        """Generate questions for a document chunk based on analysis results."""
        task_name = "generation"
        prompt_key = "question_generation"
        response_text: Optional[str] = None

        try:
            # Calculate question counts
            category_counts = self._calculate_question_counts(analysis)
            total_questions_to_generate = sum(category_counts.values())
            self.logger.debug(f"Question distribution: {category_counts} (Total: {total_questions_to_generate})")

            if total_questions_to_generate <= 0:
                self.logger.info(f"Skipping question generation for chunk {chunk.id}: Calculated question count is 0.")
                return []

            # Get LLM service
            try:
                llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)
                self.logger.debug(f"Using adapter: {type(llm_service.adapter).__name__}")
            except LLMServiceError as e:
                self.logger.error(f"Failed to get LLM service: {e}")
                raise ValidationError(f"Failed to get LLM service for generation task: {e}") from e

            # Format prompt
            prompt_vars = {
                "chunk_content": chunk.content,
                "total_questions": total_questions_to_generate,
                "factual_count": category_counts.get("factual", 0),
                "inferential_count": category_counts.get("inferential", 0),
                "conceptual_count": category_counts.get("conceptual", 0),
                "key_concepts": ", ".join(analysis.key_concepts) if analysis and analysis.key_concepts else "N/A",
                "analysis_details": json.dumps(analysis.model_dump(), indent=2) if analysis else "N/A"
            }

            formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
            expects_json = llm_service.prompt_manager.is_json_output(prompt_key)
            self.logger.debug(f"Expects JSON response: {expects_json}")

            # Generate completion
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted_prompt,
                model_config=llm_service.task_model_config
            )

            if not response_text:
                self.logger.warning("Empty response from LLM")
                raise ValidationError("LLM returned empty response during question generation.",
                                      details={"chunk_id": chunk.id})

            # Parse response
            parsed_response = self._parse_question_response(response_text, chunk.id, expected_json=expects_json)
            self.logger.debug(f"Parsed {len(parsed_response)} question items")

            # Create Question objects - UPDATED: now passing chunk and analysis
            questions = self._create_questions_from_parsed(parsed_response, chunk.id, chunk, analysis)
            self.logger.debug(f"Created {len(questions)} Question objects with enhanced metadata")

            return questions

        except (LLMServiceError, ConfigurationError) as e:
            self.logger.error(f"LLM/Config error during question generation for chunk {chunk.id}: {e}")
            if response_text:
                self.logger.debug(f"Raw response that caused LLM/Config error:\n{response_text}")
            raise ValidationError(f"LLM or Config error during question generation: {str(e)}") from e

        except ValidationError as e:
            self.logger.error(f"Validation error during question generation for chunk {chunk.id}: {e}")
            if 'response_text' in locals() and response_text:
                self.logger.debug(f"Response that caused validation error:\n{response_text[:1000]}...")

                # Log analysis detection warning
                if '"information_density"' in response_text and '"topic_coherence"' in response_text:
                    self.logger.error("Response appears to be an ANALYSIS result instead of QUESTIONS. "
                                      "Check that question_generation prompt is requesting questions.")
            raise

        except Exception as e:
            self.logger.exception(f"Unexpected error during question generation for chunk {chunk.id}: {e}")
            if 'response_text' in locals() and response_text:
                self.logger.debug(f"Raw response that caused unexpected error:\n{response_text[:1000]}...")
            raise ValidationError(f"An unexpected error occurred during question generation: {str(e)}") from e

    def _parse_question_response(self, response_text: str, chunk_id: str, expected_json: bool) -> List[Dict[str, Any]]:
        """
        Parses the LLM response with detailed diagnostics.
        """
        response_text = response_text.strip()
        self.logger.debug(f"Parsing response: length={len(response_text)}, expected_json={expected_json}")

        if not response_text:
            raise ValidationError("LLM returned an empty response for question generation.",
                                  details={"chunk_id": chunk_id})

        # Attempt to parse as a general JSON - this is a diagnostic step to understand what we're dealing with
        try:
            any_json = json.loads(response_text)
            self.logger.debug(f"Successfully parsed as {type(any_json).__name__}")

            # If we have a direct JSON object (not array)
            if isinstance(any_json, dict):
                self.logger.debug(f"JSON object keys: {list(any_json.keys())}")

                # Check if it has common analysis keys
                analysis_keys = ["information_density", "topic_coherence", "complexity",
                                 "estimated_question_yield", "key_concepts"]
                if any(k in any_json for k in analysis_keys):
                    self.logger.warning("JSON appears to be an analysis result, not a list of questions")

                # Check if it has a questions field we can use
                if "questions" in any_json and isinstance(any_json["questions"], list):
                    self.logger.debug(f"Found 'questions' field with {len(any_json['questions'])} items")
                    return any_json["questions"]

                # Check if JSON is an analysis object passed incorrectly
                if "information_density" in any_json and "estimated_question_yield" in any_json:
                    self.logger.error("Received an analysis object instead of questions. "
                                      "Check that prompt requests questions, not analysis.")

                    raise ValidationError(
                        "Received an analysis object instead of questions. Check that your prompt instructs the LLM "
                        "to return a list of question objects, not an analysis result.",
                        details={"object_keys": list(any_json.keys()), "chunk_id": chunk_id}
                    )

            # If it's already a list, check if it's a list of questions
            if isinstance(any_json, list):
                self.logger.debug(f"Found JSON array with {len(any_json)} items")
                if any_json and isinstance(any_json[0], dict):
                    self.logger.debug(f"First item keys: {list(any_json[0].keys())}")

                    # Look for expected question/answer fields
                    has_question = "question" in any_json[0] or "Q" in any_json[0]
                    has_answer = "answer" in any_json[0] or "A" in any_json[0]

                    if has_question and has_answer:
                        # Clean the list and return
                        cleaned_list = [item for item in any_json if isinstance(item, dict)]
                        return cleaned_list

        except json.JSONDecodeError as e:
            self.logger.debug(f"Not valid JSON: {e}. Trying to extract JSON from markdown code blocks")

        # Try extracting from code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_block_pattern, response_text, re.IGNORECASE)

        if matches:
            self.logger.debug(f"Found {len(matches)} code blocks")

            for i, block in enumerate(matches):
                try:
                    block = block.strip()
                    self.logger.debug(f"Parsing code block {i + 1}")

                    # Try parsing as JSON
                    parsed = json.loads(block)
                    self.logger.debug(f"Successfully parsed block {i + 1} as {type(parsed).__name__}")

                    # Handle list case
                    if isinstance(parsed, list):
                        if all(isinstance(item, dict) for item in parsed):
                            self.logger.debug(f"Block {i + 1} contains a list of {len(parsed)} dictionaries")
                            # Check if these look like questions
                            if parsed and ("question" in parsed[0] or "Q" in parsed[0]):
                                self.logger.debug(f"Block {i + 1} contains valid question items")
                                return parsed

                    # Handle dict case with questions field
                    elif isinstance(parsed, dict) and "questions" in parsed:
                        if isinstance(parsed["questions"], list):
                            self.logger.debug(
                                f"Block {i + 1} contains a questions array with {len(parsed['questions'])} items")
                            return parsed["questions"]

                except json.JSONDecodeError as e:
                    self.logger.debug(f"Block {i + 1} is not valid JSON: {e}")

        # If we've reached this point, comprehensive failure analysis
        self.logger.error("FAILED TO EXTRACT QUESTIONS FROM RESPONSE")
        self.logger.debug(f"Full response for inspection:\n{response_text}")

        raise ValidationError(
            "Could not parse LLM response as questions. The response did not contain a valid list of question objects.",
            details={
                "chunk_id": chunk_id,
                "response_start": response_text[:100],
                "response_format": "Expected JSON array or code block containing questions"
            }
        )


    def _create_questions_from_parsed(self, parsed_list: List[Dict[str, Any]], chunk_id: str,
                                      chunk: Optional[Chunk] = None,
                                      analysis: Optional[AnalysisResult] = None) -> List[Question]:
        """Converts parsed dictionary items into Question objects with comprehensive metadata for fine-tuning."""
        questions: List[Question] = []
        valid_categories = {"factual", "inferential", "conceptual", "implementation", "architectural",
                            "troubleshooting"}

        self.logger.debug(f"Creating questions from {len(parsed_list)} parsed items")

        # Get chunk context and analysis data for metadata enrichment
        chunk_context = {}
        analysis_data = {}

        if chunk:
            chunk_context = chunk.context.copy() if chunk.context else {}

        if analysis:
            # Extract ALL useful fields from analysis for fine-tuning
            analysis_data = analysis.model_dump(exclude={"chunk_id"})

        # Get generation timestamp for tracking
        generation_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Get model information if available for fine-tuning
        model_info = {}
        try:
            # Access info via task_router if available
            if hasattr(self, 'task_router'):
                task_details = self.task_router._mapped_task_details.get('generation')
                if task_details and len(task_details) >= 2:
                    _, model_config = task_details
                    model_info = {
                        'model_name': model_config.name,
                        'temperature': model_config.temperature,
                        'max_tokens': model_config.max_tokens
                    }
        except Exception as e:
            self.logger.debug(f"Could not capture model info: {e}")

        for i, item in enumerate(parsed_list):
            try:
                # Extract question and answer
                question_text = item.get("question", item.get("Q"))
                answer_text = item.get("answer", item.get("A"))
                category_raw = item.get("category", "unknown")
                category = str(category_raw).lower().strip()

                if not question_text or not answer_text or not isinstance(question_text, str) or not isinstance(
                        answer_text, str):
                    self.logger.warning(f"Item {i}: Invalid question/answer - Q: {question_text}, A: {answer_text}")
                    continue

                # Normalize and validate
                question_text = question_text.strip()
                answer_text = answer_text.strip()

                if not question_text or not answer_text:
                    self.logger.warning(f"Item {i}: Empty question/answer after stripping")
                    continue

                # Check category
                if category not in valid_categories:
                    self.logger.warning(f"Item {i}: Unknown category '{category_raw}', using 'unknown'")
                    category = "unknown"

                # Create comprehensive metadata dictionary combining ALL sources
                enhanced_metadata = {}

                # Start with original metadata if available
                if isinstance(item.get('metadata', {}), dict):
                    enhanced_metadata.update(item.get('metadata', {}))

                # Add standardized generation metadata for fine-tuning
                enhanced_metadata['generation_index'] = i
                enhanced_metadata['generation_timestamp'] = generation_timestamp

                # Add model information for fine-tuning
                if model_info:
                    enhanced_metadata['generation_model'] = model_info

                # Add document metadata
                if chunk_context.get('document_metadata'):
                    enhanced_metadata['document_metadata'] = chunk_context['document_metadata']

                # Add standardized page information
                if 'page_numbers' in chunk_context:
                    enhanced_metadata['page_numbers'] = chunk_context['page_numbers']
                if 'page_number' in chunk_context:
                    enhanced_metadata['page_number'] = chunk_context['page_number']

                # Add section path for context
                if 'section_path' in chunk_context:
                    enhanced_metadata['section_path'] = chunk_context['section_path']

                # Add document title if available
                if 'title' in chunk_context:
                    enhanced_metadata['document_title'] = chunk_context['title']

                # Add text statistics for ML features
                if 'text_stats' in chunk_context:
                    enhanced_metadata['text_stats'] = chunk_context['text_stats']

                # Add font/style information if available
                if 'font_info' in chunk_context:
                    enhanced_metadata['font_info'] = chunk_context['font_info']
                if 'style_info' in chunk_context:
                    enhanced_metadata['style_info'] = chunk_context['style_info']

                # Add ALL analysis data with standardized names
                if analysis_data:
                    enhanced_metadata['analysis'] = analysis_data

                    # Also flat-map key metrics for direct access
                    if 'key_concepts' in analysis_data:
                        enhanced_metadata['key_concepts'] = analysis_data['key_concepts']
                    if 'information_density' in analysis_data:
                        enhanced_metadata['information_density'] = analysis_data['information_density']
                    if 'topic_coherence' in analysis_data:
                        enhanced_metadata['topic_coherence'] = analysis_data['topic_coherence']
                    if 'complexity' in analysis_data:
                        enhanced_metadata['complexity'] = analysis_data['complexity']

                # Create Question object with comprehensive metadata
                questions.append(Question(
                    text=question_text,
                    answer=answer_text,
                    chunk_id=chunk_id,
                    category=category,
                    metadata=enhanced_metadata
                ))
                self.logger.debug(f"Created question {i} with comprehensive metadata")

            except Exception as item_err:
                self.logger.error(f"Failed to create question from item {i}: {item_err}")
                self.logger.debug(f"Problem item: {item}")

        self.logger.debug(f"Created {len(questions)} questions from {len(parsed_list)} items with enhanced metadata")
        return questions

    def _calculate_question_counts(self, analysis: AnalysisResult) -> Dict[str, int]:
        """Calculate how many questions to generate for each category based on configuration and analysis."""
        # Get category configuration
        category_config_section = getattr(self.config, 'categories', {})
        if not isinstance(category_config_section, dict):
            self.logger.warning("'categories' config is not a dictionary, using defaults")
            category_config_section = {}

        # Get basic settings
        max_questions = getattr(self.config, 'max_questions_per_chunk', 10)
        adaptive_enabled = getattr(self.config, 'adaptive_generation', True)
        self.logger.debug(f"Max questions: {max_questions}, Adaptive: {adaptive_enabled}")

        # Extract analysis data
        density = getattr(analysis, 'information_density', 0.5) if analysis else 0.5
        yield_estimates = getattr(analysis, 'estimated_question_yield', {}) if analysis and isinstance(
            getattr(analysis, 'estimated_question_yield', None), dict) else {}
        self.logger.debug(f"Analysis density: {density}, Yield estimates: {yield_estimates}")

        # Determine valid categories
        valid_categories_from_config = set(category_config_section.keys())
        if not valid_categories_from_config:
            valid_categories_from_config = {"factual", "inferential", "conceptual"}
            self.logger.debug(f"Using default categories: {valid_categories_from_config}")
        else:
            self.logger.debug(f"Config categories: {valid_categories_from_config}")

        # 1. Start with configured minimums
        counts = {}
        for cat in valid_categories_from_config:
            cat_cfg = category_config_section.get(cat, {})
            if not isinstance(cat_cfg, dict):
                min_q = 0
            else:
                min_q = cat_cfg.get('min_questions', 0)
            counts[cat] = int(max(0, min_q))

        self.logger.debug(f"Min counts from config: {counts}")

        # 2. Consider yield estimates from analysis
        for cat, estimated_yield in yield_estimates.items():
            if cat in counts:
                try:
                    counts[cat] = max(counts[cat], int(estimated_yield))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid yield estimate '{estimated_yield}' for '{cat}'")

        self.logger.debug(f"Counts after yield estimates: {counts}")

        # 3. Apply adaptive scaling
        if adaptive_enabled:
            density_factor = max(0.5, min(2.0, 1.0 + (density - 0.5)))
            self.logger.debug(f"Density factor: {density_factor}")

            for cat in counts:
                cat_cfg = category_config_section.get(cat, {})
                if not isinstance(cat_cfg, dict):
                    weight = 1.0
                    min_q = 0
                else:
                    weight = cat_cfg.get('weight', 1.0)
                    min_q = cat_cfg.get('min_questions', 0)

                try:
                    current_count = float(counts[cat])
                    weight = float(weight)
                    adjusted_count = current_count * density_factor * weight
                    counts[cat] = max(int(min_q), int(round(adjusted_count)))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid numeric value for category '{cat}'")
                    counts[cat] = max(int(min_q), int(counts.get(cat, 0)))

        self.logger.debug(f"Counts after adaptive scaling: {counts}")

        # 4. Enforce total max questions
        current_total = sum(counts.values())
        if current_total > max_questions:
            self.logger.debug(f"Total {current_total} exceeds max {max_questions}, scaling down")

            if current_total > 0:
                scale_factor = max_questions / current_total
                scaled_counts = {}

                for cat, count in counts.items():
                    cat_cfg = category_config_section.get(cat, {})
                    min_q = int(cat_cfg.get('min_questions', 0)) if isinstance(cat_cfg, dict) else 0
                    scaled_counts[cat] = max(min_q, int(round(count * scale_factor)))

                counts = scaled_counts
                final_total = sum(counts.values())
                excess = final_total - max_questions

                if excess > 0:
                    self.logger.debug(f"After scaling, still {excess} excess, trimming")
                    cats_to_trim = sorted(
                        counts.keys(),
                        key=lambda k: (
                            float(category_config_section.get(k, {}).get('weight', 1.0))
                            if isinstance(category_config_section.get(k), dict) else 1.0,
                            counts[k],
                            k
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
                            self.logger.debug(f"Trimmed {trim_amount} from {cat}")

                        if excess <= 0:
                            break

        # Final cleanup
        counts = {cat: max(0, int(count)) for cat, count in counts.items()}
        self.logger.debug(f"Final question counts: {counts}")

        return counts
