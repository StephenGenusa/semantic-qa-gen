# filename: semantic_qa_gen/chunking/analyzer.py

import logging
import re
import json
import yaml
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
        try:
            self.prompt_manager.get_prompt("chunk_analysis")
            self.logger.debug("Chunk analysis prompt verified.")
        except Exception as e:
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
        task_name = "analysis"
        prompt_key = "chunk_analysis"
        self.logger.debug(f"Requesting analysis for chunk {chunk.id}...")
        try:
            # 1. Get the LLMTaskService
            try:
                llm_service: LLMTaskService = self.task_router.get_task_handler(task_name)
            except LLMServiceError as e:
                 raise AnalyzerError(f"Failed to get LLM service for analysis task: {e}") from e

            # Use the renamed field task_model_config
            self.logger.debug(f"Using adapter {type(llm_service.adapter).__name__} "
                             f"with model config {llm_service.task_model_config.name} for analysis.")

            # 2. Format the analysis prompt
            try:
                prompt_vars = {"chunk_content": chunk.content}
                formatted_prompt = llm_service.prompt_manager.format_prompt(prompt_key, **prompt_vars)
                expects_json = llm_service.prompt_manager.is_json_output(prompt_key)
            except KeyError as e:
                 raise AnalyzerError(f"Prompt formatting error: Missing variable '{e}'")
            except LLMServiceError as e:
                 raise AnalyzerError(f"Analysis failed: {e}") from e

            # 3. Call the adapter's generic completion method
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted_prompt,
                model_config=llm_service.task_model_config
            )

            if not response_text:
                 self.logger.warning(f"LLM returned empty response during analysis for chunk {chunk.id}.")
                 return self._create_default_analysis_result(chunk.id, "LLM returned empty response")

            # 4. Parse the structured response using local parser
            analysis_data = self._parse_analysis_response(response_text, chunk.id, expected_json=expects_json)

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
            raise AnalyzerError(f"LLM service failed during chunk analysis: {str(e)}") from e
        except AnalyzerError as e:
             self.logger.error(f"Analyzer error processing chunk {chunk.id}: {e}")
             raise
        except ConfigurationError as e:
             self.logger.error(f"Configuration error preventing analysis for chunk {chunk.id}: {e}")
             raise
        except Exception as e:
            self.logger.exception(f"Unexpected error analyzing chunk {chunk.id}: {e}", exc_info=True)
            raise AnalyzerError(f"An unexpected error occurred during chunk analysis: {str(e)}") from e

    def _parse_analysis_response(self, response_text: str, chunk_id: str, expected_json: bool) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response (YAML or JSON), attempting recovery.

        Args:
            response_text: The raw string response from the LLM.
            chunk_id: The ID of the chunk analyzed (for logging).
            expected_json: Whether the prompt requested JSON output.

        Returns:
            A dictionary containing parsed analysis data with defaults for errors/missing fields.

        Raises:
            AnalyzerError: If parsing fails definitively after attempting recovery.
        """
        parsed_data: Optional[Dict] = None; source = "unknown"; response_text = response_text.strip()
        if not response_text: raise AnalyzerError("LLM empty response.", {"chunk_id": chunk_id})
        if expected_json: # Try JSON first
            try:
                 if response_text.startswith('{') and response_text.endswith('}'): parsed_data = json.loads(response_text); source = "direct json"
            except json.JSONDecodeError: pass
            if parsed_data is None: # Code block JSON
                match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
                if match:
                    try: parsed_data = json.loads(match.group(1).strip()); source = "block json"
                    except json.JSONDecodeError: pass
                    if not isinstance(parsed_data, dict): parsed_data = None # Verify type
        if parsed_data is None: # Try YAML
            try:
                 import yaml; parsed_yaml = yaml.safe_load(response_text)
                 if isinstance(parsed_yaml, dict): parsed_data = parsed_yaml; source = source + "+yaml" if source != "unknown" else "yaml"
            except (ImportError, yaml.YAMLError): pass
        if parsed_data is None or not isinstance(parsed_data, dict): raise AnalyzerError(f"Cannot parse analysis dict for chunk {chunk_id}.", {"preview": response_text[:500]})
        # Validate/Normalize data
        final_data = {}
        for key in ['information_density', 'topic_coherence', 'complexity']:
            try: val = float(parsed_data.get(key, 0.5)); final_data[key] = max(0.0, min(1.0, val))
            except (ValueError, TypeError, SystemError): final_data[key] = 0.5
        yield_dict = parsed_data.get('estimated_question_yield', {}); final_yield = {"factual": 0, "inferential": 0, "conceptual": 0}
        if isinstance(yield_dict, dict):
            for cat, count in yield_dict.items():
                 cat_lower = str(cat).lower();
                 if cat_lower in final_yield:
                       try: final_yield[cat_lower] = max(0, int(count))
                       except (ValueError, TypeError): final_yield[cat_lower] = 0
        final_data['estimated_question_yield'] = final_yield
        concepts = parsed_data.get('key_concepts', [])
        final_data['key_concepts'] = [str(c).strip() for c in concepts if isinstance(c, (str, int, float)) and str(c).strip()] if isinstance(concepts, list) else []
        notes = parsed_data.get('notes'); final_data['notes'] = str(notes).strip() if notes else None
        self.logger.debug(f"Parsed analysis data for chunk {chunk_id} via {source}.")
        return final_data

    def _create_default_analysis_result(self, chunk_id: str, notes: str = "Analysis failed") -> AnalysisResult:
        """Creates a fallback AnalysisResult when processing fails."""
        self.logger.warning(f"Creating default/fallback analysis result for chunk {chunk_id}. Reason: {notes}")
        return AnalysisResult(chunk_id=chunk_id, information_density=0.5, topic_coherence=0.5, complexity=0.5, estimated_question_yield={"factual": 1, "inferential": 0, "conceptual": 0}, key_concepts=[], notes=notes)

