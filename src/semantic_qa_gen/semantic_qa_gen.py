# filename: semantic_qa_gen/semantic_qa_gen.py (updated)

import asyncio
import logging
import json
import os

from typing import Optional, Dict, Any, List, Union

# Use Pydantic V2 models
from pydantic import BaseModel

# Import necessary components and utilities
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Chunk, Question
from semantic_qa_gen.pipeline.semantic import SemanticPipeline
from semantic_qa_gen.utils.logging import setup_logger
from semantic_qa_gen.utils.error import SemanticQAGenError, DocumentError, ConfigurationError, OutputError
from semantic_qa_gen.utils.project import ProjectManager
from semantic_qa_gen.version import __version__


class SemanticQAGen:
    """
    Main interface for the SemanticQAGen library using Pydantic V2.
    """

    def __init__(self, config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 verbose: bool = False,
                 project_path: Optional[str] = None):
        """
        Initialize SemanticQAGen.

        Sets up configuration, logging, and the main processing pipeline.

        Args:
            config_path: Optional path to a YAML configuration file.
                         Overrides default configuration.
            config_dict: Optional dictionary representing configuration.
                         Overrides file and default configurations.
            verbose: If True, sets logging level to DEBUG initially.
            project_path: Optional path to the QAGenProject directory.
                          If provided, uses this as the project root.

        Raises:
            ConfigurationError: If configuration loading or validation fails.
            SemanticQAGenError: For other initialization errors.
        """
        # Setup project structure manager
        self.project_manager = ProjectManager()

        # Initialize or locate project structure
        if project_path:
            if os.path.exists(project_path):
                if self.project_manager._is_project_directory(project_path):
                    self.project_path = project_path
                else:
                    # Create project at specified path
                    self.project_path = self.project_manager.create_project_structure(project_path)
            else:
                # Create project at specified path
                self.project_path = self.project_manager.create_project_structure(project_path)
        else:
            # Try to find existing project from current directory
            detected_project = self.project_manager.find_project_root()
            if detected_project:
                self.project_path = detected_project
            else:
                # Create default project in current directory
                self.project_path = self.project_manager.create_project_structure()

        # Determine initial log level
        log_level = "DEBUG" if verbose else "INFO"

        # Set up log file in project logs directory
        log_file = os.path.join(self.project_path, "logs", "semantic_qa_gen.log")

        # Setup root logger level initially - gets adjusted after config load
        self.logger = setup_logger(level=log_level, log_file=log_file)
        self.logger.info(f"Using project directory: {self.project_path}")

        try:
            # If no config_path provided, check for project config
            if not config_path:
                project_config = os.path.join(self.project_path, "config", "system.yaml")
                if os.path.exists(project_config):
                    self.logger.info(f"Using project configuration: {project_config}")
                    config_path = project_config

            # Load configuration manager (handles validation with Pydantic V2 schema)
            self.config_manager = ConfigManager(config_path, config_dict)
            # Store the validated Pydantic config object
            self.config = self.config_manager.config

            # Adjust logger level based on final config, unless verbose flag is already DEBUG
            final_log_level_str = self.config.processing.log_level
            if not verbose:
                numeric_level = getattr(logging, final_log_level_str.upper(), logging.INFO)
                logging.getLogger("semantic_qa_gen").setLevel(numeric_level)
            else:
                # If verbose=True, ensure debug_mode in config is also True potentially
                if hasattr(self.config.processing, 'debug_mode'):
                    self.config.processing.debug_mode = True

            # Initialize the processing pipeline
            self.pipeline = SemanticPipeline(self.config_manager)

            self.logger.info(f"SemanticQAGen v{__version__} initialized successfully.")
            effective_level = logging.getLogger('semantic_qa_gen').getEffectiveLevel()
            self.logger.debug(f"Effective log level set to {logging.getLevelName(effective_level)}.")

        except ConfigurationError as e:
            self.logger.critical(f"Configuration Error during initialization: {e}",
                                 exc_info=False)  # Don't need full trace usually
            raise  # Re-raise specific config error
        except Exception as e:
            self.logger.critical(f"Unexpected error during SemanticQAGen initialization: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to initialize SemanticQAGen: {str(e)}") from e

    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a single document to generate question-answer pairs.

        Args:
            document_path: Path to the document. Can be absolute or relative to the project input directory.

        Returns:
            Dictionary with processed questions, document info, and statistics.

        Raises:
            FileNotFoundError: If document not found.
            DocumentError: If document is invalid or can't be processed.
            ConfigurationError: If configuration is invalid.
            SemanticQAGenError: For other processing errors.
        """
        # Resolve document path relative to project if needed
        resolved_path = document_path
        if not os.path.isabs(document_path):
            # Check if file exists relative to current directory
            if os.path.exists(document_path):
                resolved_path = os.path.abspath(document_path)
            else:
                # Check in project input directory
                project_input_path = os.path.join(self.project_path, "input", document_path)
                if os.path.exists(project_input_path):
                    resolved_path = project_input_path
                else:
                    raise FileNotFoundError(f"Document not found: {document_path}")

        if not os.path.isfile(resolved_path):
            raise DocumentError(f"Path is not a file: {resolved_path}")

        self.logger.info(f"Starting processing for document: {resolved_path}")

        # Handle asyncio event loop
        try:
            loop = asyncio.get_running_loop()
            is_new_loop = False
        except RuntimeError:
            self.logger.debug("No running asyncio event loop found, creating new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_new_loop = True

        try:
            # Run the pipeline's main processing coroutine
            # Ensure pipeline exists before running
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                raise SemanticQAGenError("Processing pipeline was not initialized correctly.")

            result = loop.run_until_complete(self.pipeline.process_document(resolved_path))
            self.logger.info(f"Successfully processed document: {resolved_path}")
            return result
        except (FileNotFoundError, DocumentError, ConfigurationError, OutputError) as e:
            # Log specific pipeline errors and re-raise them
            self.logger.error(f"Error processing document {resolved_path}: {e}")
            raise
        except Exception as e:
            # Wrap unexpected runtime errors
            self.logger.critical(f"Critical unexpected error processing {resolved_path}: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to process document '{resolved_path}': {str(e)}") from e
        finally:
            # Close the loop only if we created it in this function scope
            # This avoids closing loops potentially managed externally (e.g., in notebooks)
            if is_new_loop:
                try:
                    # Close async generators and transports
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)  # Set loop policy back to default
                    self.logger.debug("Closed locally created asyncio event loop.")

    def save_questions(self, result: Dict[str, Any],
                       output_path: str,
                       format_name: Optional[str] = None) -> str:
        """
        Save the generated questions to a file in the specified format.

        Args:
            result: The dictionary returned by `process_document`.
            output_path: The path for the output file. If not absolute,
                         saves to project output directory.
            format_name: The desired output format ('json', 'csv', etc.).
                         Uses config default if None.

        Returns:
            The full path to the saved output file.

        Raises:
            SemanticQAGenError: If the result dictionary is invalid or saving fails.
            OutputError: If the format adapter fails or is not found.
        """
        if not isinstance(result, dict) or not all(k in result for k in ["questions", "document", "statistics"]):
            raise SemanticQAGenError("Invalid result format: Missing 'questions', 'document', or 'statistics'.")

        questions: List[Question] = result.get("questions", [])  # Should be List[Question] from pipeline
        document_info = result.get("document")
        statistics = result.get("statistics")

        # Validate questions list content (assuming Question is a known type)
        if not isinstance(questions, list) or (questions and not isinstance(questions[0], Question)):
            # If it failed validation, maybe pipeline returned dicts? check that too.
            if not (questions and isinstance(questions[0], dict) and 'text' in questions[0]):
                raise SemanticQAGenError(
                    "Invalid result: 'questions' key is not a list of Question objects or compatible dictionaries.")

        try:
            # Resolve output path relative to project if needed
            resolved_output_path = output_path
            if not os.path.isabs(output_path):
                # Save to project output directory by default
                resolved_output_path = os.path.join(self.project_path, "output", output_path)

            # Get the OutputFormatter from the initialized pipeline
            output_formatter = getattr(self.pipeline, 'output_formatter', None)
            if not output_formatter:
                raise SemanticQAGenError("Internal state error: OutputFormatter not initialized.")

            # Determine the format to use
            effective_format = (format_name or self.config.output.format).lower()

            # Serialize Question objects to dictionaries just before formatting/saving
            questions_dicts = []
            for q in questions:
                if isinstance(q, BaseModel):  # Check if using Pydantic Question model
                    # Exclude fields if needed, mode='json' for JSON-compatible types
                    questions_dicts.append(q.model_dump(mode='json'))
                elif hasattr(q, 'to_dict'):  # Check for custom dict conversion method
                    questions_dicts.append(q.to_dict())
                elif hasattr(q, '__dict__'):  # Fallback for simple objects/dataclasses
                    # Basic dict conversion, might need refinement for complex attributes
                    q_dict = {k: v for k, v in q.__dict__.items() if not k.startswith('_')}
                    questions_dicts.append(q_dict)
                else:
                    self.logger.warning(f"Cannot serialize question object of type {type(q)} to dictionary. Skipping.")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(resolved_output_path)), exist_ok=True)

            # Delegate formatting and saving to the OutputFormatter
            saved_path = output_formatter.format_and_save(
                questions=questions_dicts,
                document_info=document_info,
                statistics=statistics,
                output_path_base=resolved_output_path,  # Pass resolved path
                format_name=effective_format
            )

            self.logger.info(f"Questions saved to {saved_path} in '{effective_format}' format.")
            return saved_path

        except OutputError as e:  # Catch specific output errors
            self.logger.error(f"Output error saving questions to {output_path}: {e}")
            raise  # Re-raise OutputError
        except Exception as e:
            self.logger.error(f"Unexpected error saving questions to {output_path}: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to save questions: {str(e)}") from e

    def create_default_config_file(self, output_path: str, include_comments: bool = True) -> None:
        """
        Creates a default configuration YAML file with comments.

        Args:
            output_path: Path for the config file. If not absolute, saves to project config directory.
            include_comments: Whether to include comments in the config file.
        """
        try:
            # Resolve output path relative to project if needed
            resolved_output_path = output_path
            if not os.path.isabs(output_path):
                # Save to project config directory by default
                resolved_output_path = os.path.join(self.project_path, "config", output_path)

            self.config_manager.create_default_config_file(resolved_output_path, include_comments)
            self.logger.info(f"Default configuration file created at: {resolved_output_path}")
        except ConfigurationError as e:
            self.logger.error(f"Failed creating default config: {e}")
            raise  # Re-raise specific config error
        except Exception as e:
            self.logger.error(f"Unexpected error creating default config: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to create default configuration: {str(e)}") from e

    def create_project(self, project_path: Optional[str] = None) -> str:
        """
        Create a new QAGenProject structure.

        This creates directories and default configuration files
        for a new SemanticQAGen project.

        Args:
            project_path: Optional custom path for the project.

        Returns:
            Path to the created project directory.
        """
        try:
            project_dir = self.project_manager.create_project_structure(project_path)
            self.project_path = project_dir
            self.logger.info(f"Created new project at: {project_dir}")
            return project_dir
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            raise SemanticQAGenError(f"Project creation failed: {str(e)}") from e
