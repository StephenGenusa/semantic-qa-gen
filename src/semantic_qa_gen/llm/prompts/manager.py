"""Prompt management system for SemanticQAGen."""

import os
import yaml
import logging
import string
from typing import Dict, Any, Optional, List

from semantic_qa_gen.utils.error import LLMServiceError


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
        
        # Register built-in prompts
        self._register_builtin_prompts()
    
    def _register_builtin_prompts(self) -> None:
        """Register built-in prompt templates."""
        # Load from YAML files if directory exists
        if os.path.exists(self.prompts_dir):
            for filename in os.listdir(self.prompts_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    try:
                        path = os.path.join(self.prompts_dir, filename)
                        self._load_from_file(path)
                    except Exception as e:
                        self.logger.error(f"Failed to load prompt from {filename}: {str(e)}")
        
        # Register fallback prompts if none were loaded
        if not self.prompts:
            self._register_fallback_prompts()
    
    def _load_from_file(self, path: str) -> None:
        """
        Load prompt templates from a YAML file.
        
        Args:
            path: Path to the YAML file.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)
                
            if not isinstance(prompt_data, dict):
                self.logger.error(f"Invalid prompt file format: {path}")
                return
                
            for name, data in prompt_data.items():
                if not isinstance(data, dict) or 'template' not in data:
                    self.logger.error(f"Invalid prompt definition for {name} in {path}")
                    continue
                    
                template = data.pop('template')
                metadata = data
                
                # Add file source to metadata
                metadata['source'] = os.path.basename(path)
                
                self.register_prompt(name, template, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to load prompts from {path}: {str(e)}")
    
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
            
            Format your response as JSON with the following structure:
            {{
                "information_density": float,
                "topic_coherence": float,
                "complexity": float,
                "estimated_question_yield": {{
                    "factual": int,
                    "inferential": int,
                    "conceptual": int
                }},
                "key_concepts": [string],
                "notes": string
            }}
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
            
            Format your response as JSON with the following structure:
            {{
                "is_valid": true/false,
                "factual_accuracy": float (0.0-1.0),
                "answer_completeness": float (0.0-1.0),
                "question_clarity": float (0.0-1.0),
                "reasons": [string],
                "suggested_improvements": string (optional)
            }}
            """,
            {
                "description": "Validates a question-answer pair against the source text",
                "json_output": True,
                "system_prompt": "You are an AI assistant specialized in evaluating educational questions and answers."
            }
        )
    
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
        self.logger.debug(f"Registered prompt template: {name}")
    
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
