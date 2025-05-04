# filename: semantic_qa_gen/chunking/analyzer.py

import logging
import re
import json

# Make sure yaml is imported if used as a fallback
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Logging will warn if YAML parsing is attempted without the library

from typing import Dict, Any, Optional

# Use LLMTaskService as data holder from router
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.document.models import AnalysisResult, Chunk
from semantic_qa_gen.utils.error import AnalyzerError, LLMServiceError, ConfigurationError


class SemanticAnalyzer:
    """
    Analyzes document chunks for semantic properties using an LLM.

    Gets the required LLM adapter and model config via the TaskRouter,
    formats the prompt, calls the adapter's generate_completion,
    parses the structured response, and creates an AnalysisResult object.
    """

    def __init__(self, task_router: TaskRouter, prompt_manager: PromptManager):
        """
        Initialize the SemanticAnalyzer.

        Args:
            task_router: The router to get appropriate LLM services.
            prompt_manager: The manager for retrieving prompt templates.
        """
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        # Verify the prompt key used in analyze_chunk exists
        self._analysis_prompt_key = "chunk_analysis"  # Store the key used
        try:
            self.prompt_manager.get_prompt(self._analysis_prompt_key)
            self.logger.debug(f"Chunk analysis prompt '{self._analysis_prompt_key}' verified.")
        except Exception as e:
            raise ConfigurationError(
                f"SemanticAnalyzer init failed: Required analysis prompt '{self._analysis_prompt_key}' not found. Details: {e}")

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
        task_name = "analysis"
        prompt_key = self._analysis_prompt_key  # Use the stored key
        self.logger.debug(f"Requesting analysis for chunk {chunk.id} using prompt '{prompt_key}'...")
        response_text: Optional[str] = None  # Initialize response_text

        try:
            # 1. Get the LLMTaskService
            try:
                llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)
            except LLMServiceError as e:
                raise AnalyzerError(f"Failed to get LLM service for analysis task: {e}") from e

            self.logger.debug(f"Using adapter {type(llm_service.adapter).__name__} "
                              f"with model config {llm_service.task_model_config.name} for analysis.")

            # 2. Format the analysis prompt
            try:
                prompt_vars = {"chunk_content": chunk.content}
                formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
                expects_json = llm_service.prompt_manager.is_json_output(prompt_key)
            except KeyError as e:
                raise AnalyzerError(f"Prompt formatting error: Missing variable '{e}' for prompt '{prompt_key}'")
            except LLMServiceError as e:
                raise AnalyzerError(f"Analysis failed during prompt handling: {e}") from e

            # 3. Call the adapter's generic completion method
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted_prompt,
                model_config=llm_service.task_model_config  # Use the config from the service
            )

            if not response_text:
                self.logger.warning(f"LLM returned empty response during analysis for chunk {chunk.id}.")
                raise AnalyzerError("LLM returned empty response during analysis.", details={"chunk_id": chunk.id})

            # 4. Parse the structured response with enhanced logic
            analysis_data = self._parse_analysis_response(response_text, chunk.id)

            # 5. Create and return the AnalysisResult object
            #    Map parsed keys to AnalysisResult fields
            return AnalysisResult(
                chunk_id=chunk.id,
                information_density=analysis_data.get('information_density', 0.5),  # Use normalized data
                topic_coherence=analysis_data.get('topic_coherence', 0.5),  # Use normalized data
                complexity=analysis_data.get('complexity', 0.5),  # Use normalized data
                estimated_question_yield=analysis_data.get('estimated_question_yield',
                                                           {"factual": 1, "inferential": 0, "conceptual": 0}),
                # Use normalized data
                key_concepts=analysis_data.get('key_concepts', []),  # Use normalized data
                notes=analysis_data.get('notes')  # Use normalized data
            )

        # --- Exception Handling (Includes Debug Logging of Raw Response) ---
        except LLMServiceError as e:
            self.logger.error(f"LLM service error during analysis for chunk {chunk.id}: {e}")
            if response_text:
                self.logger.error(
                    f"[DEBUG] Raw LLM response during LLMServiceError for chunk {chunk.id}:\n>>>\n{response_text}\n<<<")
            # Wrap in AnalyzerError for consistent pipeline handling
            raise AnalyzerError(f"LLM service failed during chunk analysis: {str(e)}") from e
        except AnalyzerError as e:
            # Log potentially more details if it's a parsing error
            log_raw_response = "parse" in str(e).lower() or "empty response" in str(e).lower()
            self.logger.error(f"Analyzer error processing chunk {chunk.id}: {e}")
            if log_raw_response and response_text is not None:
                self.logger.error(
                    f"[DEBUG] Raw LLM response associated with AnalyzerError for chunk {chunk.id}:\n"
                    f">>>\n{response_text}\n<<<"
                )
            elif log_raw_response:
                self.logger.error(f"[DEBUG] AnalyzerError for chunk {chunk.id}, but raw response was not captured.")
            raise  # Re-raise the specific AnalyzerError
        except ConfigurationError as e:
            self.logger.error(f"Configuration error preventing analysis for chunk {chunk.id}: {e}")
            raise  # Re-raise config errors directly
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing chunk {chunk.id}: {e}", exc_info=True)
            if response_text:
                self.logger.error(
                    f"[DEBUG] Raw LLM response during unexpected exception for chunk {chunk.id}:\n>>>\n{response_text}\n<<<")
            # Wrap unexpected errors in AnalyzerError
            raise AnalyzerError(f"An unexpected error occurred during chunk analysis: {str(e)}") from e

    def _clean_json_string(self, text: str, chunk_id: str) -> str:
        """Helper to strip comments, trailing commas, and potentially problematic trailing characters."""
        try:
            # Remove // comments
            cleaned = re.sub(r"\s*//.*$", "", text, flags=re.MULTILINE)
            # Remove /* ... */ block comments (less common but possible)
            cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r",\s*(}|\])", r"\1", cleaned)

            # Attempt to remove simple extra characters after the last '}' or ']' (more general)
            # Find the index of the last potential closing character
            last_brace_index = cleaned.rfind('}')
            last_bracket_index = cleaned.rfind(']')
            last_char_index = max(last_brace_index, last_bracket_index)

            if last_char_index != -1:
                # Keep only up to and including the last brace/bracket, strip trailing whitespace/junk
                possible_json = cleaned[:last_char_index + 1].rstrip()
                # Basic check: Ensure start matches end (e.g., starts '{' ends '}', starts '[' ends ']')
                if (possible_json.startswith('{') and possible_json.endswith('}')) or \
                        (possible_json.startswith('[') and possible_json.endswith(']')):
                    cleaned = possible_json
                else:
                    # If start/end don't match after trimming, maybe the trim was wrong, log and use previous state
                    self.logger.debug(
                        f"Suffix stripping for chunk {chunk_id} resulted in mismatched start/end. Reverting.")
                    # cleaned remains as it was before suffix stripping attempt

            return cleaned.strip()

        except Exception as re_err:
            self.logger.warning(
                f"Regex cleaning failed for chunk {chunk_id}: {re_err}. Using original text for this block.")
            return text.strip()

    def _parse_analysis_response(self, response_text: str, chunk_id: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response (JSON/YAML), attempting recovery,
        stripping comments, and handling potential extra characters.

        Args:
            response_text: The raw string response from the LLM.
            chunk_id: The ID of the chunk analyzed (for logging).
            expects_json: Whether the prompt requested JSON output.

        Returns:
            A dictionary containing parsed analysis data.

        Raises:
            AnalyzerError: If parsing fails definitively after attempting recovery.
        """
        response_text = response_text.strip()
        if not response_text:
            raise AnalyzerError("LLM returned empty response.", details={"chunk_id": chunk_id})

        parsed_data: Optional[Dict] = None
        cleaned_text_attempted: str = response_text  # Keep track of the final text tried
        source = "unknown"

        # --- Parsing Attempts ---

        # Attempt 1: Extract JSON object from markdown code block ```json ... ``` first
        # Match only if the content inside looks like an OBJECT ({...})
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            code_block_text = match.group(1).strip()
            cleaned_block = self._clean_json_string(code_block_text, chunk_id)  # Use helper
            cleaned_text_attempted = cleaned_block  # This is the primary text we try if block exists
            try:
                parsed_data = json.loads(cleaned_block)
                source = "extracted block json (cleaned)"
                if not isinstance(parsed_data, dict):
                    self.logger.warning(
                        f"Parsed JSON from code block for chunk {chunk_id} is not a dict (got {type(parsed_data)}). Resetting.")
                    parsed_data = None  # Needs to be a dict
                else:
                    self.logger.debug(f"Successfully parsed cleaned JSON object from code block for chunk {chunk_id}.")
                    # Proceed to normalization using parsed_data
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Failed to parse cleaned JSON block for chunk {chunk_id}: {e}. Raw Block:\n>>>\n{code_block_text}\n<<<\nCleaned Block:\n>>>\n{cleaned_block}\n<<<")
                # Fall through to try parsing the whole response

        # Attempt 2: Try parsing the entire response as JSON (if block failed or not found)
        if parsed_data is None:
            cleaned_full_response = self._clean_json_string(response_text, chunk_id)  # Use helper
            cleaned_text_attempted = cleaned_full_response  # Update the text we tried
            try:
                # Check if it looks like a JSON object
                if cleaned_full_response.startswith('{') and cleaned_full_response.endswith('}'):
                    parsed_data = json.loads(cleaned_full_response)
                    source = "direct json (cleaned)"
                    if not isinstance(parsed_data, dict):
                        self.logger.warning(
                            f"Parsed direct JSON for chunk {chunk_id} is not a dict (got {type(parsed_data)}). Resetting.")
                        parsed_data = None  # Needs to be a dict
                    else:
                        self.logger.debug(
                            f"Successfully parsed cleaned full response as JSON object for chunk {chunk_id}.")
                        # Proceed to normalization
                else:
                    self.logger.debug(
                        f"Cleaned full response for chunk {chunk_id} doesn't start/end with {{}}. Skipping direct JSON parse attempt.")
            except json.JSONDecodeError as e:
                self.logger.warning(f"Direct JSON parsing failed for cleaned full response for chunk {chunk_id}: {e}")
                # Fall through to YAML attempt

        # Attempt 3: Try parsing the cleaned full response as YAML (last resort)
        if parsed_data is None and YAML_AVAILABLE:  # Only try if PyYAML is installed
            # Use the same cleaned text as the last JSON attempt
            cleaned_full_response = cleaned_text_attempted
            try:
                # We use the cleaned text primarily to remove comments/trailing chars
                parsed_yaml = yaml.safe_load(cleaned_full_response)
                if isinstance(parsed_yaml, dict):
                    parsed_data = parsed_yaml
                    source = "yaml parse (cleaned)"
                    self.logger.debug(f"Successfully parsed cleaned full response as YAML dict for chunk {chunk_id}.")
                    # Proceed to normalization
                else:
                    self.logger.debug(
                        f"YAML parse for chunk {chunk_id} did not yield a dict (got {type(parsed_yaml)}).")
            except yaml.YAMLError as yaml_err:
                # Log the specific YAML error
                self.logger.warning(f"YAML parsing failed for chunk {chunk_id}: {yaml_err}")
            except Exception as general_yaml_err:  # Catch broader errors during YAML load
                self.logger.warning(f"Unexpected error during YAML parsing for chunk {chunk_id}: {general_yaml_err}")
        elif parsed_data is None and not YAML_AVAILABLE:
            self.logger.debug("PyYAML not installed. Skipping YAML parse attempt.")

        # --- Final Check and Error ---
        if parsed_data is None or not isinstance(parsed_data, dict):
            # Include the cleaned text that was last attempted in the preview
            self.logger.error(
                f"Cannot parse analysis dict for chunk {chunk_id} after cleaning/extraction (last source attempt: {source})."
            )
            # Log the raw response as well for maximum context
            self.logger.error(
                f"[DEBUG] Raw LLM response for failed parse of chunk {chunk_id}:\n>>>\n{response_text}\n<<<")
            raise AnalyzerError(
                f"Cannot parse analysis dict for chunk {chunk_id} after all attempts.",
                details={"final_cleaned_attempt": cleaned_text_attempted[:500], "parse_source_attempt": source}
            )

        # --- Normalization Logic (if parsing succeeded) ---
        # Use helper methods and handle potential missing keys safely
        final_data = {}
        final_data['information_density'] = self._normalize_float(
            parsed_data.get('information_density', parsed_data.get('technical_depth'))
        )
        final_data['topic_coherence'] = self._normalize_float(
            parsed_data.get('topic_coherence', parsed_data.get('architectural_clarity'))
        )
        final_data['complexity'] = self._normalize_float(
            parsed_data.get('complexity', parsed_data.get('implementation_specificity'))
        )

        # Handle estimated_question_yield (prefer specific keys)
        yield_dict = parsed_data.get('estimated_question_yield', {})
        if isinstance(yield_dict, dict):
            final_data['estimated_question_yield'] = {
                'factual': self._normalize_int(yield_dict.get('factual')),
                'inferential': self._normalize_int(yield_dict.get('inferential')),
                'conceptual': self._normalize_int(yield_dict.get('conceptual')),
            }
        else:  # Fallback if 'estimated_question_yield' is missing or not a dict
            self.logger.warning(f"Missing or invalid 'estimated_question_yield' for chunk {chunk_id}. Using default.")
            final_data['estimated_question_yield'] = {"factual": 1, "inferential": 0, "conceptual": 0}

        # Handle key concepts (ensure it's a list of strings)
        concepts_list = parsed_data.get('key_concepts', parsed_data.get('key_technical_concepts', []))
        if isinstance(concepts_list, list):
            final_data['key_concepts'] = [str(c).strip() for c in concepts_list if
                                          isinstance(c, (str, int, float)) and str(c).strip()]
        else:
            self.logger.warning(
                f"Invalid 'key_concepts' format for chunk {chunk_id} (expected list). Using empty list.")
            final_data['key_concepts'] = []

        # Handle notes (ensure it's a string or None)
        notes_val = parsed_data.get('notes')
        final_data['notes'] = str(notes_val).strip() if notes_val is not None else None

        self.logger.debug(f"Successfully parsed and normalized analysis data for chunk {chunk_id} via {source}.")
        return final_data

    def _normalize_float(self, value: Any, default: float = 0.5) -> float:
        """Safely convert value to float between 0.0 and 1.0."""
        try:
            # Handle potential numeric strings
            val = float(str(value).strip())
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError, SystemError) as e:
            self.logger.debug(f"Could not normalize value '{value}' to float: {e}. Using default {default}.")
            return default

    def _normalize_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to a non-negative integer."""
        try:
            # Handle potential numeric strings or floats that can be int
            val = int(float(str(value).strip()))
            return max(0, val)
        except (ValueError, TypeError, SystemError) as e:
            self.logger.debug(f"Could not normalize value '{value}' to int: {e}. Using default {default}.")
            return default

    def _create_default_analysis_result(self, chunk_id: str,
                                        notes: str = "Analysis failed or used fallback.") -> AnalysisResult:
        """Creates a fallback AnalysisResult when processing fails."""
        self.logger.warning(f"Creating default/fallback analysis result for chunk {chunk_id}. Reason: {notes}")
        return AnalysisResult(
            chunk_id=chunk_id,
            information_density=0.5,
            topic_coherence=0.5,
            complexity=0.5,
            estimated_question_yield={"factual": 1, "inferential": 0, "conceptual": 0},
            key_concepts=[],
            notes=notes  # Include the reason in the notes
        )

