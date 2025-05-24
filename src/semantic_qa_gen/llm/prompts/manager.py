"""Prompt management system for SemanticQAGen."""

import os
import yaml
import logging
import string
from typing import Dict, Any, Optional, List, Set

from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError


class PromptTemplate:
    """
    Template for LLM prompts with variable substitution.

    Prompt templates allow for consistent prompt formatting with
    dynamic content insertion and metadata management.
    """

    def __init__(self, template: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt template.

        Args:
            template: Template string with {variable} placeholders.
            metadata: Optional metadata about the prompt.
        """
        self.template = template
        self.metadata = metadata or {}

    def format(self, **kwargs) -> str:
        """
        Format the template by substituting variables.

        Args:
            **kwargs: Variables to substitute.

        Returns:
            Formatted prompt string.

        Raises:
            KeyError: If a required variable is missing.
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise KeyError(f"Missing required variable in prompt template: {missing_var}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {str(e)}")


class PromptManager:
    """
    Manager for organizing and retrieving prompt templates.

    This class handles loading prompt templates from files and
    providing them on demand for various LLM tasks.
    """

    # Define essential prompt keys that the system requires
    ESSENTIAL_PROMPTS = {
        "chunk_analysis",
        "question_generation"
    }

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Optional directory for loading prompts.
        """
        self.prompts: Dict[str, PromptTemplate] = {}
        self.logger = logging.getLogger(__name__)

        # Default prompts directory within the package
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )

        # Track which prompts were loaded from files
        self.loaded_from_files: Set[str] = set()

        # Register built-in prompts
        self._register_builtin_prompts()

        # Verify essential prompts
        self._verify_essential_prompts()

    def _register_builtin_prompts(self) -> None:
        """Register built-in prompt templates."""
        # Load from YAML files if directory exists
        loaded_any = False
        missing_essential = set()

        if os.path.exists(self.prompts_dir):
            for filename in os.listdir(self.prompts_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    try:
                        path = os.path.join(self.prompts_dir, filename)
                        keys_loaded = self._load_from_file(path)
                        if keys_loaded:
                            loaded_any = True
                            self.logger.info(f"Loaded {len(keys_loaded)} prompt templates from {filename}")

                            # Check if we loaded essential prompts
                            for key in self.ESSENTIAL_PROMPTS:
                                if key in keys_loaded:
                                    self.logger.info(f"Loaded essential prompt '{key}' from {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to load prompt from {filename}: {str(e)}")

        # Register fallback prompts if none were loaded
        if not loaded_any:
            self.logger.warning("No prompt templates loaded from files. Using fallback prompts.")
            self._register_fallback_prompts()
        else:
            # Check for missing essential prompts
            missing_essential = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
            if missing_essential:
                self.logger.warning(f"Missing essential prompts after loading files: {missing_essential}")
                self._register_selected_fallbacks(missing_essential)

    def _load_from_file(self, path: str) -> Set[str]:
        """
        Load prompt templates from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Set of keys loaded from this file.

        Raises:
            ConfigurationError: If the file contains invalid prompt definitions.
        """
        loaded_keys = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)

            if not isinstance(prompt_data, dict):
                raise ConfigurationError(f"Invalid prompt file format: {path}. Expected a dictionary.")

            for name, data in prompt_data.items():
                if not isinstance(data, dict):
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: Not a dictionary")
                    continue

                if 'template' not in data:
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: Missing 'template' field")
                    continue

                template = data.pop('template')
                if not isinstance(template, str):
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: 'template' is not a string")
                    continue

                metadata = data

                # Add file source to metadata
                metadata['source'] = os.path.basename(path)

                # Log template preview for debugging
                template_preview = template[:100] + '...' if len(template) > 100 else template
                self.logger.debug(f"Loading prompt '{name}' from {path}: {template_preview}")

                # Check for essential prompt
                if name in self.ESSENTIAL_PROMPTS:
                    if name == "question_generation" and "analyze" in template.lower()[:100]:
                        self.logger.warning(
                            f"WARNING: The prompt '{name}' in {path} appears to be an analysis prompt, "
                            f"not a question generation prompt. This may cause failures."
                        )

                self.register_prompt(name, template, metadata)
                loaded_keys.add(name)
                self.loaded_from_files.add(name)

            return loaded_keys

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load prompts from {path}: {str(e)}")

    def _register_selected_fallbacks(self, missing_keys: Set[str]) -> None:
        """Register only the specified fallback prompts."""
        if "chunk_analysis" in missing_keys:
            self.logger.info("Registering fallback chunk_analysis prompt")
            self.register_prompt(
                "chunk_analysis",
                """
                Please analyze the following text passage and provide information about its 
                educational value for generating quiz questions. Focus on aspects like information
                density (0.0-1.0), topic coherence (0.0-1.0), complexity (0.0-1.0), and how many
                questions of different types could be generated from it.

                Text passage:
                ---
                {chunk_content}
                ---
                
                Include estimates for:
                - factual questions (direct information in the text)
                - inferential questions (requiring connecting information)
                - conceptual questions (dealing with broader principles)
                
                Return your analysis in the following JSON format EXACTLY:
                {{
                    "information_density":  0.0-1.0,
                    "topic_coherence": 0.0-1.0,
                    "complexity": 0.0-1.0,
                    "estimated_question_yield": {{
                        "factual": number,
                        "inferential": number,
                        "conceptual": number,
                    }},
                    "key_concepts": ["concept1", "concept2", ...],
                    "notes": string
                }}

                **CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
                **DO NOT include any comments (like // or #) or any other text outside the JSON structure.**
                Adjust the numerical and string values based on the actual content provided, maintaining the precise JSON structure.
    """,
                {
                    "description": "Analyzes a text chunk for information density and question potential",
                    "json_output": True,
                    "system_prompt": "You are an AI assistant specialized in analyzing text passages for educational content."
                }
            )

        if "question_generation" in missing_keys:
            self.logger.info("Registering fallback question_generation prompt")
            self.register_prompt(
                "question_generation",
                """
                Generate questions and answers based on the following text. Create {total_questions} questions total:
                - {factual_count} factual questions (based directly on information in the text)
                - {inferential_count} inferential questions (requiring connecting information from the text)
                - {conceptual_count} conceptual questions (addressing broader principles or ideas)
                
                Text:
                ---
                {chunk_content}
                ---
                
                Key concepts: {key_concepts}
                
                Format your response as a JSON array of question objects with the following structure:
                [
                    {{
                        "question": "The question text",
                        "answer": "The comprehensive answer",
                        "category": "factual|inferential|conceptual"
                    }},
                    ...
                ]
                
                Make the answers comprehensive and accurate based on the text. Each answer should fully explain the concept
                being asked about, not just provide a short answer.
                """,
                {
                    "description": "Generates questions and answers based on a text chunk",
                    "json_output": True,
                    "system_prompt": "You are an AI assistant specialized in creating educational questions and answers."
                }
            )

        # Add other essential prompts as needed

    def _register_fallback_prompts(self) -> None:
        """Register fallback prompts if no prompts were loaded from files."""
        # Analysis prompt
        self.register_prompt(
            "chunk_analysis",
            """
            Please analyze the following text passage and provide information about its 
            educational value for generating quiz questions. Focus on aspects like information
            density (0.0-1.0), topic coherence (0.0-1.0), complexity (0.0-1.0), and how many
            questions of different types could be generated from it.

            Text passage:
            ---
            {chunk_content}
            ---
            
            Include estimates for:
            - factual questions (direct information in the text)
            - inferential questions (requiring connecting information)
            - conceptual questions (dealing with broader principles)
            
            Return your analysis in the following JSON format EXACTLY:
            {{
                "information_density":  0.0-1.0,
                "topic_coherence": 0.0-1.0,
                "complexity": 0.0-1.0,
                "estimated_question_yield": {{
                    "factual": number,
                    "inferential": number,
                    "conceptual": number,
                }},
                "key_concepts": ["concept1", "concept2", ...],
                "notes": string
            }}

            **CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
            **DO NOT include any comments (like // or #) or any other text outside the JSON structure.**
            Adjust the numerical and string values based on the actual content provided, maintaining the precise JSON structure.
""",
            {
                "description": "Analyzes a text chunk for information density and question potential",
                "json_output": True,
                "system_prompt": "You are an AI assistant specialized in analyzing text passages for educational content."
            }
        )

        # Question generation prompt
        self.register_prompt(
            "question_generation",
            """
            Generate questions and answers based on the following text. Create {total_questions} questions total:
            - {factual_count} factual questions (based directly on information in the text)
            - {inferential_count} inferential questions (requiring connecting information from the text)
            - {conceptual_count} conceptual questions (addressing broader principles or ideas)
            
            Text:
            ---
            {chunk_content}
            ---
            
            Format your response as a JSON array of question objects with the following structure:
            [
                {{
                    "question": "The question text",
                    "answer": "The comprehensive answer",
                    "category": "factual|inferential|conceptual"
                }},
                ...
            ]
            
            Make the answers comprehensive and accurate based on the text. Each answer should fully explain the concept
            being asked about, not just provide a short answer.
            """,
            {
                "description": "Generates questions and answers based on a text chunk",
                "json_output": True,
                "system_prompt": "You are an AI assistant specialized in creating educational questions and answers."
            }
        )

        # Question validation prompt
        self.register_prompt(
            "question_validation",
            """
            Evaluate the following question and answer based on the provided source text.
            
            Source text:
            ---
            {chunk_content}
            ---
            
            Question: {question_text}
            
            Answer: {answer_text}
            
            Please verify:
            1. Factual accuracy: Is the answer factually correct according to the source text?
            2. Answer completeness: Does the answer fully address the question?
            3. Question clarity: Is the question clear and unambiguous?
            
            Return your analysis in the following JSON format EXACTLY:
            {{
                "is_valid": true/false,
                "factual_accuracy": float (0.0-1.0),
                "answer_completeness": float (0.0-1.0),
                "question_clarity": float (0.0-1.0),
                "reasons": [string],
                "suggested_improvements": string (optional)
            }}

            **CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
            **DO NOT include any comments (like // or #) or any other text outside the JSON structure.**
            Adjust the numerical and string values based on the actual content provided, maintaining the precise JSON structure.
""",
            {
                "description": "Validates a question-answer pair against the source text",
                "json_output": True,
                "system_prompt": "You are an AI assistant specialized in evaluating educational questions and answers."
            }
        )

    def _verify_essential_prompts(self) -> None:
        """Verify that all essential prompts are available."""
        missing = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
        if missing:
            error_msg = f"Critical prompt templates missing: {', '.join(missing)}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)

        # Verify question_generation is actually generating questions, not analysis
        if "question_generation" in self.prompts:
            template = self.prompts["question_generation"].template.lower()
            if "analyze" in template[:100] and "density" in template and "coherence" in template:
                warning = (
                    "WARNING: The 'question_generation' prompt appears to be an analysis prompt, "
                    "not a question generation prompt. This will cause generation failures."
                )
                self.logger.error(warning)
                # Consider raising an error here if you want to fail fast
    
    def register_prompt(self, name: str, template: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new prompt template.
        
        Args:
            name: Prompt name/identifier.
            template: Template string.
            metadata: Optional metadata.
        """
        self.prompts[name] = PromptTemplate(template, metadata)
        self.logger.info(f"Registered prompt template: {name}")
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.
        
        Args:
            name: Prompt name/identifier.
            
        Returns:
            Prompt template.
            
        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        if name not in self.prompts:
            raise LLMServiceError(f"Prompt template not found: {name}")
        
        return self.prompts[name]
    
    def format_prompt(self, name: str, **kwargs) -> str:
        """
        Format a prompt template with variable substitution.
        
        Args:
            name: Prompt name/identifier.
            **kwargs: Variables to substitute.
            
        Returns:
            Formatted prompt string.
            
        Raises:
            LLMServiceError: If the prompt doesn't exist or formatting fails.
        """
        try:
            template = self.get_prompt(name)
            return template.format(**kwargs)
        except KeyError as e:
            raise LLMServiceError(f"Missing variable in prompt template {name}: {str(e)}")
        except Exception as e:
            raise LLMServiceError(f"Error formatting prompt {name}: {str(e)}")
    
    def get_system_prompt(self, name: str) -> Optional[str]:
        """
        Get the system prompt for a template, if defined.
        
        Args:
            name: Prompt name/identifier.
            
        Returns:
            System prompt string if defined, otherwise None.
            
        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        template = self.get_prompt(name)
        return template.metadata.get('system_prompt')
    
    def is_json_output(self, name: str) -> bool:
        """
        Check if a prompt expects JSON output.
        
        Args:
            name: Prompt name/identifier.
            
        Returns:
            True if the prompt expects JSON output.
            
        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        template = self.get_prompt(name)
        return template.metadata.get('json_output', False)
