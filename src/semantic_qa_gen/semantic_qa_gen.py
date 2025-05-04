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

    Provides methods to initialize the system, process individual documents,
    or process all documents within the project's input directory.
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
            # Use provided path, create if necessary
            project_path = os.path.abspath(project_path)
            if not os.path.exists(project_path) or not self.project_manager._is_project_directory(project_path):
                self.logger = logging.getLogger(__name__) # Basic logger for project creation
                self.logger.info(f"Creating project structure at specified path: {project_path}")
                try:
                    self.project_path = self.project_manager.create_project_structure(project_path)
                except ConfigurationError as e:
                    self.logger.critical(f"Failed to create project at specified path {project_path}: {e}")
                    raise
            else:
                 self.project_path = project_path
        else:
             # Try to find existing project from current directory
            detected_project = self.project_manager.find_project_root()
            if detected_project:
                self.project_path = detected_project
            else:
                self.logger = logging.getLogger(__name__) # Basic logger for project creation
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
        # Path resolution is now more robustly handled within this method
        if not os.path.isabs(document_path):
            # Check relative to current dir FIRST
            if os.path.exists(os.path.abspath(document_path)):
                 resolved_path = os.path.abspath(document_path)
                 self.logger.debug(f"Resolved document path relative to CWD: {resolved_path}")
            else:
                 # Then check relative to project input directory
                project_input_path = os.path.join(self.project_path, "input", document_path)
                if os.path.exists(project_input_path):
                    resolved_path = project_input_path
                    self.logger.debug(f"Resolved document path relative to project input: {resolved_path}")
                else:
                     # If not found in either place
                    raise FileNotFoundError(f"Document not found: '{document_path}' "
                                           f"(checked absolute and relative to project input dir '{os.path.join(self.project_path, 'input')}').")
        elif not os.path.exists(resolved_path):
             # If absolute path was given but doesn't exist
             raise FileNotFoundError(f"Absolute document path does not exist: {resolved_path}")

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

    def process_input_directory(self, output_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes all readable files in the project's input directory.

        Args:
            output_format: Optional output format ('json', 'csv', etc.) to override
                           the default from the configuration for saving results.

        Returns:
            A dictionary summarizing the batch processing results, including counts
            of processed and failed files, and lists of output/failed file paths.
            Example:
            {
                "processed_count": 3,
                "failed_count": 1,
                "output_files": ["/path/to/proj/output/doc1.json", ...],
                "failed_files": [{"file": "inaccessible.pdf", "error": "Permission denied"}]
            }

        Raises:
            SemanticQAGenError: If the input directory cannot be accessed.
        """
        input_dir = os.path.join(self.project_path, "input")

        if not os.path.isdir(input_dir):
            message = f"Input directory not found or is not a directory: {input_dir}"
            self.logger.error(message)
            raise SemanticQAGenError(message)

        self.logger.info(f"Starting batch processing for directory: {input_dir}")

        processed_count = 0
        failed_count = 0
        output_files = []
        failed_files_info = []

        # Determine the output format to use for saving
        # If argument is provided, it overrides the config default
        save_format = (output_format or self.config.output.format).lower()

        try:
            files_in_dir = os.listdir(input_dir)
        except OSError as e:
             message = f"Cannot list files in input directory {input_dir}: {e}"
             self.logger.error(message)
             raise SemanticQAGenError(message) from e

        if not files_in_dir:
             self.logger.warning(f"Input directory is empty: {input_dir}")
             return {
                 "processed_count": 0,
                 "failed_count": 0,
                 "output_files": [],
                 "failed_files": []
             }

        self.logger.info(f"Found {len(files_in_dir)} potential items in input directory.")

        for filename in files_in_dir:
            file_path = os.path.join(input_dir, filename)

            # Simple check if it's a file before attempting to process
            if not os.path.isfile(file_path):
                self.logger.debug(f"Skipping non-file entry: {filename}")
                continue

            # Optionally add a quick check for obviously unsupported extensions?
            # Or rely solely on DocumentProcessor's loader selection?
            # Relying on DP is cleaner, avoids duplicating logic.

            self.logger.info(f"--- Processing file: {filename} ---")
            try:
                # 1. Process the document
                results = self.process_document(file_path) # Pass the full path
                self.logger.info(f"  Successfully processed {filename}.")

                # 2. Save the results
                try:
                    # Generate base output filename from input filename
                    base_name = os.path.splitext(filename)[0]
                    output_filename_base = base_name # save_questions handles extension/dir

                    self.logger.info(f"  Saving results for {filename} as {save_format.upper()}...")

                    # Use the determined save format
                    saved_path = self.save_questions(
                        result=results,
                        output_path=output_filename_base, # Pass base name
                        format_name=save_format # Specify format
                    )
                    self.logger.info(f"  Results saved to: {saved_path}")
                    output_files.append(saved_path)
                    processed_count += 1

                except (OutputError, SemanticQAGenError, IOError) as save_err:
                    self.logger.error(f"  Error saving results for {filename}: {save_err}")
                    failed_count += 1
                    failed_files_info.append({"file": filename, "error": f"Save Error: {save_err}"})

            except (DocumentError, FileNotFoundError, ConfigurationError, SemanticQAGenError) as proc_err:
                # Catch expected errors during processing
                self.logger.error(f"  Error processing {filename}: {proc_err}")
                failed_count += 1
                failed_files_info.append({"file": filename, "error": f"Processing Error: {proc_err}"})
            except Exception as unexpected_err:
                 # Catch unexpected errors
                self.logger.exception(f"  Unexpected critical error processing {filename}: {unexpected_err}", exc_info=True)
                failed_count += 1
                failed_files_info.append({"file": filename, "error": f"Unexpected Error: {unexpected_err}"})

        # --- Batch Completion Summary ---
        self.logger.info("--- Batch processing complete ---")
        self.logger.info(f"Successfully processed and saved: {processed_count} file(s)")
        self.logger.info(f"Failed to process or save: {failed_count} file(s)")
        if failed_files_info:
             self.logger.warning("Details of failed files:")
             for failed in failed_files_info:
                  self.logger.warning(f"  - {failed['file']}: {failed['error']}")

        return {
            "processed_count": processed_count,
            "failed_count": failed_count,
            "output_files": output_files,
            "failed_files": failed_files_info
        }


    def save_questions(self, result: Dict[str, Any],
                       output_path: str, # Changed name to clarify it's base path/name
                       format_name: Optional[str] = None) -> str:
        """
        Save the generated questions to a file in the specified format.

        Args:
            result: The dictionary returned by `process_document`.
            output_path: The base path or filename for the output file.
                         If relative, saves inside the project's 'output' directory.
                         The correct file extension will be appended based on format.
                         Example: "mydoc_results" -> "/path/to/proj/output/mydoc_results.json"
            format_name: The desired output format ('json', 'csv', etc.).
                         Uses config default if None.

        Returns:
            The full, absolute path to the saved output file.

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
        # Allow list of dicts as pipeline might return that before full model instantiation in some error paths
        is_valid_question_list = isinstance(questions, list) and \
                                  (not questions or isinstance(questions[0], (Question, dict)))
        if not is_valid_question_list:
               raise SemanticQAGenError(
                   "Invalid result: 'questions' key is not a list of Question objects or compatible dictionaries.")

        try:
            # Get the OutputFormatter from the initialized pipeline
            output_formatter = getattr(self.pipeline, 'output_formatter', None)
            if not output_formatter:
                raise SemanticQAGenError("Internal state error: OutputFormatter not initialized.")

            # Determine the format to use
            effective_format = (format_name or self.config.output.format).lower()

            # Resolve the base output path relative to project output dir if needed
            resolved_output_base = output_path
            if not os.path.isabs(output_path):
                resolved_output_base = os.path.join(self.project_path, "output", output_path)
            else: # If absolute, ensure base directory exists
                 base_dir = os.path.dirname(output_path)
                 if base_dir: os.makedirs(base_dir, exist_ok=True)


            # Serialize Question objects to dictionaries just before formatting/saving
            questions_dicts = []
            for q in questions:
                if isinstance(q, BaseModel):  # Check if using Pydantic Question model
                    try:
                        # Mode='json' ensures complex types are JSON-serializable
                        questions_dicts.append(q.model_dump(mode='json', exclude_none=True))
                    except Exception as dump_err:
                         self.logger.warning(f"Failed to dump Question model {getattr(q, 'id', '?')}: {dump_err}. Skipping.")
                elif isinstance(q, dict): # Handle case where pipeline result was already dicts
                     questions_dicts.append(q)
                elif hasattr(q, 'to_dict'):  # Check for custom dict conversion method
                    questions_dicts.append(q.to_dict())
                # Removed the __dict__ fallback - too unreliable. Prefer model_dump or pass dicts.
                else:
                    self.logger.warning(f"Cannot serialize question object of type {type(q)} to dictionary. Skipping.")

            # Delegate formatting and saving to the OutputFormatter
            # Pass the resolved *base* path; formatter/adapter adds extension
            saved_path = output_formatter.format_and_save(
                questions=questions_dicts,
                document_info=document_info,
                statistics=statistics,
                output_path_base=resolved_output_base,
                format_name=effective_format
            )

            self.logger.info(f"Questions saved to {saved_path} in '{effective_format}' format.")
            # Return the absolute path to the saved file
            return os.path.abspath(saved_path)

        except OutputError as e:  # Catch specific output errors
            self.logger.error(f"Output error saving questions for base path {output_path}: {e}")
            raise  # Re-raise OutputError
        except Exception as e:
            self.logger.error(f"Unexpected error saving questions for base path {output_path}: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to save questions: {str(e)}") from e

    def create_default_config_file(self, output_path: str, include_comments: bool = True) -> None:
        """
        Creates a default configuration YAML file with comments.

        Args:
            output_path: Path for the config file. If not absolute, saves relative to project config directory.
            include_comments: Whether to include comments in the config file.
        """
        try:
            # Resolve output path relative to project config directory if needed
            resolved_output_path = output_path
            if not os.path.isabs(output_path):
                resolved_output_path = os.path.join(self.project_path, "config", output_path)

            # Ensure directory exists before calling config manager's save method
            os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

            self.config_manager.create_default_config_file(resolved_output_path, include_comments)
            self.logger.info(f"Default configuration file created at: {resolved_output_path}")
        except ConfigurationError as e:
            self.logger.error(f"Failed creating default config: {e}")
            raise  # Re-raise specific config error
        except Exception as e:
            self.logger.error(f"Unexpected error creating default config at {output_path}: {e}", exc_info=True)
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
            # Update self.project_path if a new project was created successfully
            # Or potentially re-initialize if necessary? Simpler to just update path.
            self.project_path = project_dir
            self.logger.info(f"Created new project at: {project_dir}")
            return project_dir
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            raise SemanticQAGenError(f"Project creation failed: {str(e)}") from e

