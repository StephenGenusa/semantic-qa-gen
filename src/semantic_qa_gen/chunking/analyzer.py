# filename: semantic_qa_gen/chunking/analyzer.py

"""Component for performing semantic analysis of document chunks using LLMs."""

import logging
import re
from typing import Dict, Any, Optional

import yaml # Keep for fallback parsing if robust parser needs it
import json # Added for JSON parsing

from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.document.models import AnalysisResult, Chunk
from semantic_qa_gen.utils.error import AnalyzerError, LLMServiceError, ConfigurationError

class SemanticAnalyzer:
    """
    Analyzes document chunks for semantic properties using an LLM.

    This component retrieves the appropriate LLM service via the TaskRouter,
    formats a specific analysis prompt, calls the LLM, and parses the
    structured response to create an AnalysisResult object.
    """

    def __init__(self, task_router: TaskRouter, prompt_manager: PromptManager):
        """
        Initialize the SemanticAnalyzer.

        Args:
            task_router: The router to get appropriate LLM services.
            prompt_manager: The manager for retrieving prompt templates.
        """
        self.task_router = task_router
        # PromptManager is needed to format the prompt template correctly here.
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)

        # Verify existence of the required prompt template during initialization
        try:
            self.prompt_manager.get_prompt("chunk_analysis")
            self.logger.debug("Chunk analysis prompt verified.")
        except Exception as e:
            # If the essential prompt is missing, analyzer cannot function.
            raise ConfigurationError(f"SemanticAnalyzer init failed: Required prompt 'chunk_analysis' not found. Details: {e}")


    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Perform semantic analysis on a single chunk using an LLM via TaskRouter.

        Args:
            chunk: The Chunk object to analyze.

        Returns:
            An AnalysisResult object containing the analysis metrics.

        Raises:
            AnalyzerError: If the analysis fails due to LLM errors, prompt issues,
                           parsing errors, or configuration issues.
        """
        self.logger.debug(f"Requesting analysis for chunk {chunk.id} (sequence {chunk.sequence})...")
        try:
            # 1. Get the appropriate LLM task handler for "analysis"
            llm_task_service: LLMTaskService = self.task_router.get_task_handler("analysis")

            self.logger.debug(f"Using adapter {type(llm_task_service.adapter).__name__} "
                             f"with model config {llm_task_service.model_config.name} for analysis.")

            # 2. Get and format the analysis prompt
            try:
                analysis_prompt_template = self.prompt_manager.get_prompt("chunk_analysis")
                prompt_vars = {"chunk_content": chunk.content}
                formatted_prompt = analysis_prompt_template.format(**prompt_vars)
                # Check if prompt/adapter expects JSON mode
                expects_json = self.prompt_manager.is_json_output("chunk_analysis")
                # TODO: Pass `expects_json` hint to `generate_completion` if the adapter implements it
            except KeyError as e:
                 self.logger.error(f"Missing variable '{e}' for chunk_analysis prompt template.")
                 raise AnalyzerError(f"Prompt formatting error: Missing variable '{e}'")
            except LLMServiceError as e: # Catch prompt not found from get_prompt
                 raise AnalyzerError(f"Analysis failed: {e}") from e

            # 3. Call the LLM via the adapter's core completion method
            response_text = await llm_task_service.adapter.generate_completion(
                prompt=formatted_prompt,
                model_config=llm_task_service.model_config # Pass specific model config for this task
            )

            if not response_text:
                 self.logger.warning(f"LLM returned empty response during analysis for chunk {chunk.id}.")
                 # Return default/fallback AnalysisResult on empty response
                 return self._create_default_analysis_result(chunk.id, "LLM returned empty response")


            # 4. Parse the structured response (YAML or JSON)
            # Moved parsing logic here from adapter base
            analysis_data = self._parse_analysis_response(response_text, chunk.id)

            # 5. Create and return the AnalysisResult object
            return AnalysisResult(
                chunk_id=chunk.id,
                information_density=analysis_data.get('information_density', 0.5),
                topic_coherence=analysis_data.get('topic_coherence', 0.5),
                complexity=analysis_data.get('complexity', 0.5),
                estimated_question_yield=analysis_data.get('estimated_question_yield', {"factual": 1, "inferential": 0, "conceptual": 0}),
                key_concepts=analysis_data.get('key_concepts', []),
                notes=analysis_data.get('notes')
            )

        except LLMServiceError as e:
            self.logger.error(f"LLM service error during analysis for chunk {chunk.id}: {e}")
            # Depending on config, could return default or raise
            # Raise AnalyzerError to let pipeline decide on retry/skip
            raise AnalyzerError(f"LLM service failed during chunk analysis: {str(e)}") from e
        except AnalyzerError as e: # Catch formatting/parsing errors raised internally
             self.logger.error(f"Analyzer error processing chunk {chunk.id}: {e}")
             raise # Re-raise specific AnalyzerErrors
        except ConfigurationError as e:
             self.logger.error(f"Configuration error preventing analysis for chunk {chunk.id}: {e}")
             raise # Re-raise config errors immediately
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing chunk {chunk.id}: {e}", exc_info=True)
            raise AnalyzerError(f"An unexpected error occurred during chunk analysis: {str(e)}") from e


    def _parse_analysis_response(self, response_text: str, chunk_id: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response (YAML or JSON), attempting recovery.

        Args:
            response_text: The raw string response from the LLM.
            chunk_id: The ID of the chunk analyzed (for logging).

        Returns:
            A dictionary containing parsed analysis data with defaults for errors/missing fields.

        Raises:
            AnalyzerError: If parsing fails definitively after attempting recovery.
        """
        parsed_data: Optional[Dict[str, Any]] = None
        response_text = response_text.strip()
        source = "unknown"

        # Attempt 1: Try parsing as JSON first
        try:
            # Check if it looks like a JSON object/array first
            if response_text.startswith('{') or response_text.startswith('['):
                parsed_data = json.loads(response_text)
                source = "direct JSON parse"
        except json.JSONDecodeError:
            self.logger.debug(f"Direct JSON parse failed for chunk {chunk_id}. Trying YAML.")
            pass # Fall through to YAML

        # Attempt 2: Try parsing as YAML if JSON failed or wasn't attempted
        if parsed_data is None:
            try:
                parsed_data = yaml.safe_load(response_text)
                # Check if YAML parsing resulted in a primitive type instead of dict
                if not isinstance(parsed_data, dict):
                     self.logger.warning(f"YAML parse for chunk {chunk_id} did not yield a dictionary (got {type(parsed_data)}). Treating as parse failure.")
                     parsed_data = None # Reset parsed_data to trigger code block extraction
                else:
                     source = "direct YAML parse"
            except yaml.YAMLError as yaml_err:
                self.logger.warning(f"Direct YAML parse failed for chunk {chunk_id}: {yaml_err}. Trying code block extraction.")
                pass # Fall through to code block extraction

        # Attempt 3: Extract from Markdown code blocks if direct parsing failed
        if parsed_data is None:
             code_block_json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
             code_block_yaml_match = re.search(r"```(?:yaml|yml)?\s*\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)

             if code_block_json_match:
                 try:
                     parsed_data = json.loads(code_block_json_match.group(1))
                     source = "extracted JSON code block"
                     self.logger.info(f"Successfully parsed analysis from extracted JSON block for chunk {chunk_id}.")
                 except json.JSONDecodeError as e:
                     self.logger.error(f"Failed to parse extracted JSON block for analysis (Chunk {chunk_id}): {e}")
                     parsed_data = None # Indicate failure
             elif code_block_yaml_match:
                 try:
                     parsed_data = yaml.safe_load(code_block_yaml_match.group(1).strip())
                     if not isinstance(parsed_data, dict): # Verify type again
                          parsed_data = None
                     else:
                          source = "extracted YAML code block"
                          self.logger.info(f"Successfully parsed analysis from extracted YAML block for chunk {chunk_id}.")
                 except yaml.YAMLError as e:
                     self.logger.error(f"Failed to parse extracted YAML block for analysis (Chunk {chunk_id}): {e}")
                     parsed_data = None # Indicate failure

        # Final check and raise if still failed
        if parsed_data is None or not isinstance(parsed_data, dict):
            self.logger.error(f"Failed to parse analysis response structure for chunk {chunk_id} after all attempts.")
            raise AnalyzerError(f"Could not parse LLM analysis response for chunk {chunk_id}.", details={"response": response_text[:500]}) # Include snippet of response

        # --- Data Validation & Normalization with Defaults ---
        final_data: Dict[str, Any] = {}
        # Scores (float, 0.0-1.0, default 0.5)
        for key in ['information_density', 'topic_coherence', 'complexity']:
            value = parsed_data.get(key, 0.5)
            try:
                value_f = float(value)
                final_data[key] = max(0.0, min(1.0, value_f))
                if not (0.0 <= value_f <= 1.0):
                     self.logger.warning(f"analysis:{source}: Metric '{key}' value ({value_f}) for chunk {chunk_id} outside [0, 1]. Clamped.")
            except (ValueError, TypeError):
                 self.logger.warning(f"analysis:{source}: Could not parse metric '{key}' ('{value}') as float for chunk {chunk_id}. Defaulting to 0.5.")
                 final_data[key] = 0.5

        # Estimated Yield (dict[str, int], default 0)
        yield_dict = parsed_data.get('estimated_question_yield', {})
        final_yield: Dict[str, int] = {}
        # Ensure default categories exist
        default_categories = ["factual", "inferential", "conceptual"]
        for cat in default_categories:
            final_yield[cat] = 0

        if isinstance(yield_dict, dict):
            for cat, count in yield_dict.items():
                 cat_lower = str(cat).lower()
                 # Only take expected categories or warn? Let's be flexible but clean keys.
                 valid_key = next((c for c in default_categories if c == cat_lower), None)
                 if not valid_key:
                     self.logger.debug(f"analysis:{source}: Ignoring unexpected yield category '{cat}' for chunk {chunk_id}.")
                     continue
                 try:
                    final_yield[valid_key] = max(0, int(count))
                 except (ValueError, TypeError):
                    self.logger.warning(f"analysis:{source}: Could not parse yield '{cat}' ('{count}') as int for chunk {chunk_id}. Defaulting to 0.")
                    # Keep the default 0 set above
        else:
             self.logger.warning(f"analysis:{source}: 'estimated_question_yield' for chunk {chunk_id} not a dict. Using defaults.")
        final_data['estimated_question_yield'] = final_yield

        # Key Concepts (list[str], default [])
        concepts = parsed_data.get('key_concepts', [])
        if isinstance(concepts, list):
            final_concepts = [str(item).strip() for item in concepts if isinstance(item, (str, int, float)) and str(item).strip()]
            if len(final_concepts) != len(concepts): # Indicates some items were skipped/cleaned
                 self.logger.debug(f"analysis:{source}: Cleaned 'key_concepts' list for chunk {chunk_id}.")
            final_data['key_concepts'] = final_concepts
        else:
             self.logger.warning(f"analysis:{source}: 'key_concepts' for chunk {chunk_id} not a list. Using empty.")
             final_data['key_concepts'] = []

        # Notes (str, optional)
        notes_val = parsed_data.get('notes')
        final_data['notes'] = str(notes_val).strip() if notes_val else None # Store None if empty/None

        self.logger.debug(f"Parsed analysis data for chunk {chunk_id} via {source}: {final_data}")
        return final_data

    def _create_default_analysis_result(self, chunk_id: str, notes: str = "Analysis failed") -> AnalysisResult:
        """Creates a fallback AnalysisResult when processing fails."""
        self.logger.warning(f"Creating default/fallback analysis result for chunk {chunk_id}. Reason: {notes}")
        return AnalysisResult(
            chunk_id=chunk_id,
            information_density=0.5,
            topic_coherence=0.5,
            complexity=0.5,
            estimated_question_yield={"factual": 1, "inferential": 0, "conceptual": 0}, # Minimum default yield
            key_concepts=[],
            notes=notes
        )

