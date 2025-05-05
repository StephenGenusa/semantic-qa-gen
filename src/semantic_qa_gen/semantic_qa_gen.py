# semantic_qa_gen/semantic_qa_gen.py

import asyncio
import logging
import json
import os
import glob
from typing import Optional, Dict, Any, List, Union

# Use Pydantic V2 models
from pydantic import BaseModel

# Import necessary components and utilities
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Chunk, Question
from semantic_qa_gen.pipeline.semantic import SemanticPipeline
from semantic_qa_gen.utils.logging import setup_logger
from semantic_qa_gen.utils.error import SemanticQAGenError, DocumentError, ConfigurationError, OutputError
from semantic_qa_gen.utils.project import ProjectManager # Use the enhanced ProjectManager
from semantic_qa_gen.version import __version__


class SemanticQAGen:
    """
    Main interface for the SemanticQAGen library using Pydantic V2.

    Provides methods to initialize the system, process individual documents,
    or process all documents within the project's input directory.
    Relies on ProjectManager for filesystem structure and path resolution.
    """

    def __init__(self, config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 verbose: bool = False,
                 project_path: Optional[str] = None):
        """
        Initializes SemanticQAGen.

        Sets up project management, configuration, logging, and the main
        processing pipeline. Finds or creates a project structure.

        Args:
            config_path: Optional path to a YAML configuration file.
                         Can be absolute or relative to the project config dir.
                         Overrides default project configuration.
            config_dict: Optional dictionary representing configuration.
                         Overrides file and default configurations.
            verbose: If True, sets logging level to DEBUG initially.
                     Overrides the level set in the configuration file.
            project_path: Optional path to the QAGenProject directory or where
                          to create it. If None, searches from CWD or creates
                          a default project.

        Raises:
            ConfigurationError: If project setup, config loading, or validation fails.
            SemanticQAGenError: For other initialization errors.
        """
        # Need a basic logger setup for ProjectManager initialization messages
        # This might log to console initially if file path isn't known yet
        temp_logger = logging.getLogger(__name__)
        initial_log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=initial_log_level)


        try:
            # Initialize ProjectManager - handles finding/creating project root
            self.project_manager = ProjectManager(project_path)
            self.project_path = self.project_manager.get_project_root() # Store determined project root
        except ConfigurationError as e:
            temp_logger.critical(f"Project initialization failed: {e}", exc_info=False)
            raise
        except Exception as e:
            temp_logger.critical(f"Unexpected error during ProjectManager initialization: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to initialize project manager: {e}") from e

        # Setup logger properly now that project path is known
        log_file_path = self.project_manager.resolve_log_path()
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Use project path and desired level for the final logger setup
        self.logger = setup_logger(name="semantic_qa_gen",
                                   level="DEBUG" if verbose else "INFO",
                                   log_file=log_file_path)
        self.logger.info(f"Using project directory: {self.project_path}")

        try:
            # Determine the configuration file path to use
            effective_config_path = None
            if config_path:
                # Resolve provided config_path relative to project config dir if not absolute
                if not os.path.isabs(config_path):
                     potential_path = self.project_manager.resolve_config_path(config_path)
                     if os.path.exists(potential_path):
                         effective_config_path = potential_path
                         self.logger.debug(f"Resolved relative config path to: {potential_path}")
                     else: # If relative path doesn't exist in config dir, try relative to CWD
                          potential_cwd_path = os.path.abspath(config_path)
                          if os.path.exists(potential_cwd_path):
                               effective_config_path = potential_cwd_path
                               self.logger.debug(f"Using config path relative to CWD: {potential_cwd_path}")
                          else:
                               raise ConfigurationError(f"Config file not found at relative path: {config_path}")
                elif os.path.exists(config_path): # Absolute path
                     effective_config_path = config_path
                else:
                    raise ConfigurationError(f"Config file not found at absolute path: {config_path}")

            else:
                # No config path provided, use default in project if it exists
                default_project_config = self.project_manager.resolve_config_path()
                if os.path.exists(default_project_config):
                    self.logger.info(f"Using default project configuration: {default_project_config}")
                    effective_config_path = default_project_config
                else:
                     self.logger.info("No config file specified and default not found. Using internal defaults.")


            # Initialize ConfigManager with the determined path (or None) and dict override
            self.config_manager = ConfigManager(effective_config_path, config_dict)
            self.config = self.config_manager.config # Get the final merged config

            # Apply final log level based on config, unless overridden by verbose flag
            final_log_level_str = self.config.processing.log_level
            if not verbose:
                # Get numeric level from config string (default to INFO)
                numeric_level = getattr(logging, final_log_level_str.upper(), logging.INFO)
                # Set level for the root logger of the package
                logging.getLogger("semantic_qa_gen").setLevel(numeric_level)
                # Also update self.logger's level if needed
                self.logger.setLevel(numeric_level)
            else:
                # Verbose flag forces DEBUG level
                logging.getLogger("semantic_qa_gen").setLevel(logging.DEBUG)
                self.logger.setLevel(logging.DEBUG)
                # Ensure debug_mode reflects verbosity if present
                if hasattr(self.config.processing, 'debug_mode'):
                    self.config.processing.debug_mode = True


            # Initialize the main processing pipeline
            self.pipeline = SemanticPipeline(self.config_manager)

            self.logger.info(f"SemanticQAGen v{__version__} initialized successfully.")
            # Log the effective level after all adjustments
            effective_level = logging.getLogger('semantic_qa_gen').getEffectiveLevel()
            self.logger.debug(f"Effective log level set to: {logging.getLevelName(effective_level)}")

        except ConfigurationError as e:
            # Log config errors specifically during config loading/pipeline init stage
            self.logger.critical(f"Configuration Error during initialization: {e}", exc_info=False)
            raise
        except Exception as e:
            # Catch any other unexpected errors during initialization
            self.logger.critical(f"Unexpected error during SemanticQAGen initialization: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to initialize SemanticQAGen: {e}") from e


    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Processes a single document to generate question-answer pairs.

        Args:
            document_path: Path to the document. Can be absolute, relative to CWD,
                           or relative to the project input directory.

        Returns:
            Dictionary containing:
                - 'questions': List[Question] - The generated Question objects.
                - 'document': Dict[str, Any] - Information about the processed document.
                - 'statistics': Dict[str, Any] - Statistics about the processing run.

        Raises:
            FileNotFoundError: If the document cannot be found.
            DocumentError: If the path is not a file or the document is invalid.
            ConfigurationError: If the processing pipeline configuration is invalid.
            OutputError: If an error occurs during an intermediate output step.
            SemanticQAGenError: For other processing errors.
        """
        resolved_path = None

        # 1. Check if absolute path exists
        if os.path.isabs(document_path):
            if os.path.exists(document_path):
                resolved_path = document_path
                self.logger.debug(f"Using absolute document path: {resolved_path}")
            else:
                raise FileNotFoundError(f"Absolute document path does not exist: {document_path}")
        else:
            # 2. Check relative to CWD
            cwd_path = os.path.abspath(document_path)
            if os.path.exists(cwd_path):
                 resolved_path = cwd_path
                 self.logger.debug(f"Resolved document path relative to CWD: {resolved_path}")
            else:
                 # 3. Check relative to project input directory
                 project_input_path = self.project_manager.resolve_input_path(document_path)
                 if os.path.exists(project_input_path):
                     resolved_path = project_input_path
                     self.logger.debug(f"Resolved document path relative to project input: {resolved_path}")
                 else:
                    # Document not found in any expected location
                    raise FileNotFoundError(
                        f"Document not found: '{document_path}' "
                        f"(Checked absolute, relative to CWD '{os.getcwd()}', and "
                        f"relative to project input '{self.project_manager.get_input_dir()}')."
                    )

        # Ensure the resolved path is a file
        if not os.path.isfile(resolved_path):
            raise DocumentError(f"Resolved path is not a file: {resolved_path}")

        self.logger.info(f"Starting processing for document: {resolved_path}")

        # --- Asyncio Event Loop Handling ---
        try:
            # Try to get the existing loop if running in an async context
            loop = asyncio.get_running_loop()
            is_new_loop = False
        except RuntimeError:
            # No loop running, create a new one for this operation
            self.logger.debug("No running asyncio event loop found, creating new one for process_document.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_new_loop = True
        # --- End Asyncio Event Loop Handling ---

        try:
            # Check if pipeline was initialized correctly
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                raise SemanticQAGenError("Processing pipeline was not initialized correctly.")

            # Run the asynchronous pipeline processing within the event loop
            # Result should contain List[Question] objects, doc info, stats
            result: Dict[str, Any] = loop.run_until_complete(self.pipeline.process_document(resolved_path))

            self.logger.info(f"Successfully processed document: {resolved_path}")
            return result # Return the results dictionary

        except (FileNotFoundError, DocumentError, ConfigurationError, OutputError) as e:
            # Catch specific, expected errors during processing
            self.logger.error(f"Error processing document {resolved_path}: {e}")
            raise # Re-raise the specific error
        except Exception as e:
            # Catch unexpected errors during pipeline execution
            self.logger.critical(f"Critical unexpected error processing {resolved_path}: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to process document '{resolved_path}': {e}") from e

        finally:
            # Clean up the event loop ONLY if we created it locally
            if is_new_loop:
                try:
                    # Shutdown async generators
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    # Close the loop and reset the policy's loop
                    loop.close()
                    asyncio.set_event_loop(None)
                    self.logger.debug("Closed locally created asyncio event loop.")


    def process_input_directory(self, output_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes all readable files in the project's input directory.

        Iterates through files, calls `process_document` for each, saves the
        results using `save_questions`, and optionally compiles JSONL outputs
        into a master file.

        Args:
            output_format: Optional output format ('json', 'csv', 'jsonl') to
                           override the config default for saving results.

        Returns:
            A dictionary summarizing the batch processing results:
                - 'processed_count': Number of files successfully processed and saved.
                - 'failed_count': Number of files that failed processing or saving.
                - 'output_files': List of absolute paths to the generated output files.
                - 'failed_files': List of dicts {'file': str, 'error': str} for failures.
                - 'master_compiled_file': Path to the compiled master JSONL file (or None).
                - 'compilation_errors': List of errors encountered during compilation.
        """
        input_dir = self.project_manager.get_input_dir()

        if not os.path.isdir(input_dir):
            message = f"Input directory not found or is not a directory: {input_dir}"
            self.logger.error(message)
            # Perhaps should be a FileNotFoundError or ConfigurationError?
            raise SemanticQAGenError(message)

        self.logger.info(f"Starting batch processing for directory: {input_dir}")

        processed_count: int = 0
        failed_count: int = 0
        output_files: List[str] = [] # Store paths of successfully generated individual files
        failed_files_info: List[Dict[str, str]] = []

        # Determine the format used for saving individual files
        save_format = (output_format or self.config.output.format).lower()

        try:
            # List directory contents
            files_in_dir = os.listdir(input_dir)
            if not files_in_dir:
                self.logger.warning(f"Input directory is empty: {input_dir}")
                # Return empty summary Dictionaries are mutable default arguments
                return {
                    "processed_count": 0, "failed_count": 0,
                    "output_files": [], "failed_files": [],
                    "master_compiled_file": None, "compilation_errors": []
                }

            self.logger.info(f"Found {len(files_in_dir)} potential items in input directory.")

        except OSError as e:
             message = f"Cannot list files in input directory {input_dir}: {e}"
             self.logger.error(message)
             raise SemanticQAGenError(message) from e

             self.logger.warning(f"Input directory is empty: {input_dir}")
             return {
                 "processed_count": 0,
                 "failed_count": 0,
                 "output_files": [],
                 "failed_files": []
             }

        # Process each item in the directory
        for filename in files_in_dir:
            file_path = os.path.join(input_dir, filename) # Already absolute path

            # Skip if not a file
            if not os.path.isfile(file_path):
                self.logger.debug(f"Skipping non-file entry: {filename}")
                continue

            self.logger.info(f"--- Processing file: {filename} ---")
            try:
                # Process the document
                results = self.process_document(file_path) # Returns dict with List[Question]
                self.logger.info(f"  Successfully processed {filename}.")

                # Save the results
                try:
                    # Use filename without extension as base for output
                    base_name = os.path.splitext(filename)[0]
                    self.logger.info(f"  Saving results for {filename} as {save_format.upper()}...")

                    # save_questions handles path resolution within output dir
                    saved_path = self.save_questions(
                        result=results,
                        output_path=base_name, # Pass base name, save_questions resolves it
                        format_name=save_format
                    )
                    self.logger.info(f"  Results saved to: {saved_path}")
                    output_files.append(saved_path)
                    processed_count += 1

                except (OutputError, SemanticQAGenError, IOError) as save_err:
                    # Handle errors during saving
                    self.logger.error(f"  Error saving results for {filename}: {save_err}")
                    failed_count += 1
                    failed_files_info.append({"file": filename, "error": f"Save Error: {str(save_err)}"})

            except (DocumentError, FileNotFoundError, ConfigurationError, OutputError, SemanticQAGenError) as proc_err:
                # Handle specific errors during processing
                self.logger.error(f"  Error processing {filename}: {proc_err}")
                failed_count += 1
                failed_files_info.append({"file": filename, "error": f"Processing Error: {str(proc_err)}"})
            except Exception as unexpected_err:
                # Handle unexpected errors during processing
                self.logger.exception(f"  Unexpected critical error processing {filename}: {unexpected_err}", exc_info=True)
                failed_count += 1
                failed_files_info.append({"file": filename, "error": f"Unexpected Error: {str(unexpected_err)}"})

        # --- Batch processing summary logging ---
        self.logger.info("--- Batch processing complete ---")
        self.logger.info(f"Successfully processed and saved: {processed_count} file(s)")
        self.logger.info(f"Failed to process or save: {failed_count} file(s)")
        if failed_files_info:
             self.logger.warning("Details of failed files:")
             for failed in failed_files_info:
                  self.logger.warning(f"  - {failed['file']}: {failed['error']}")
        # --- End Batch processing summary logging ---


        # --- Compile JSONL outputs ---
        master_file_path: Optional[str] = None
        compile_errors: List[str] = []

        # Only compile if the chosen save format was JSONL and there were successful outputs
        if save_format == 'jsonl' and output_files:
            # Define master filename and resolve its path in the output directory
            master_filename = "_master_qa_generated.jsonl"
            master_file_path = self.project_manager.resolve_output_path(master_filename)

            self.logger.info(f"Compiling {len(output_files)} JSONL files into master file: {master_file_path}")

            try:
                # Ensure output directory exists for the master file
                os.makedirs(os.path.dirname(master_file_path), exist_ok=True)

                # Open master file for writing
                with open(master_file_path, 'w', encoding='utf-8') as master_file:
                    # Iterate through each successfully created individual JSONL file
                    for individual_file_path in output_files:
                        if not os.path.exists(individual_file_path):
                             err_msg = f"Individual file not found during compilation: {individual_file_path}"
                             self.logger.warning(err_msg)
                             compile_errors.append(err_msg)
                             continue # Skip to next file

                        # Read individual file and write valid lines to master
                        try:
                            with open(individual_file_path, 'r', encoding='utf-8') as infile:
                                for line_num, line in enumerate(infile, 1):
                                     line_stripped = line.strip()
                                     # Basic validation: check if line is a non-empty JSON object
                                     if line_stripped.startswith('{') and line_stripped.endswith('}'):
                                         try:
                                             # Optionally do a full JSON parse for validation? Might slow down significantly.
                                             # json.loads(line_stripped)
                                             master_file.write(line) # Write original line (with newline)
                                         except json.JSONDecodeError:
                                              self.logger.warning(f"Skipping invalid JSON line {line_num} in {os.path.basename(individual_file_path)} during compilation.")
                                              compile_errors.append(f"Invalid JSON in {os.path.basename(individual_file_path)} line {line_num}")
                                     elif line_stripped: # Log if line has content but isn't a JSON object
                                         self.logger.warning(f"Skipping non-object line {line_num} in {os.path.basename(individual_file_path)} during compilation: {line_stripped[:100]}...")
                                         compile_errors.append(f"Non-object line in {os.path.basename(individual_file_path)} line {line_num}")
                        except (IOError, OSError) as read_err:
                            err_msg = f"Error reading file {os.path.basename(individual_file_path)} during compilation: {read_err}"
                            self.logger.error(err_msg)
                            compile_errors.append(err_msg)
                        except Exception as comp_err:
                            # Catch unexpected errors reading/processing an individual file
                            err_msg = f"Unexpected error processing file {os.path.basename(individual_file_path)} during compilation: {comp_err}"
                            self.logger.error(err_msg, exc_info=self.config.processing.debug_mode)
                            compile_errors.append(err_msg)

                # Log compilation outcome
                if not compile_errors:
                     self.logger.info(f"Successfully compiled master JSONL file: {master_file_path}")
                else:
                     self.logger.warning(f"Master JSONL compilation finished with {len(compile_errors)} errors/warnings.")

            except (IOError, OSError) as master_write_err:
                 # Handle errors opening/writing the master file itself
                 err_msg = f"Failed to write master JSONL file {master_file_path}: {master_write_err}"
                 self.logger.error(err_msg)
                 compile_errors.append(err_msg)
                 master_file_path = None # Indicate master file creation failed
            except Exception as master_err:
                 # Catch unexpected errors during master file creation
                 err_msg = f"Unexpected error writing master JSONL file {master_file_path}: {master_err}"
                 self.logger.error(err_msg, exc_info=self.config.processing.debug_mode)
                 compile_errors.append(err_msg)
                 master_file_path = None # Indicate master file creation failed
        # --- End Compile JSONL outputs ---


        # Prepare final summary dictionary
        result_summary = {
            "processed_count": processed_count,
            "failed_count": failed_count,
            "output_files": output_files,         # List of individual file paths
            "failed_files": failed_files_info,  # List of {'file': name, 'error': msg}
            "master_compiled_file": master_file_path, # Path or None
            "compilation_errors": compile_errors  # List of error messages
        }

        return result_summary


    def save_questions(self, result: Dict[str, Any],
                       output_base_filename: str,
                       format_name: Optional[str] = None) -> str:
        """
        Saves the generated Question objects to a file in the specified format.

        Formats the Question objects into dictionaries suitable for AI training
        (input, output, context, metadata) before passing them to the
        OutputFormatter.

        Args:
            result: The dictionary returned by `process_document`. Must contain:
                - 'questions': List[Question] - The generated questions.
                - 'document': Dict[str, Any] - Info about the source document.
                - 'statistics': Dict[str, Any] - Processing statistics.
            output_base_filename: The base filename for the output file (e.g., "my_doc").
                              This will be resolved relative to the project's
                              output directory. The correct extension is added
                              by the formatter.
            format_name: The desired output format ('json', 'csv', 'jsonl').
                         Uses config default if None.

        Returns:
            The full, absolute path to the saved output file.

        Raises:
            SemanticQAGenError: If the result dict format is invalid, questions
                                are not Question objects, or the formatter is missing.
            OutputError: If the OutputFormatter encounters an error during
                         formatting or saving.
            ValueError: If output_path_base is invalid.
        """
        # Validate input dictionary structure
        if not isinstance(result, dict) or not all(k in result for k in ["questions", "document", "statistics"]):
            raise SemanticQAGenError("Invalid result format: Must contain 'questions', 'document', and 'statistics'.")

        questions_list: List[Question] = result.get("questions", [])
        document_info: Dict[str, Any] = result.get("document", {}) # Default to empty dict
        statistics: Dict[str, Any] = result.get("statistics", {})

        # Basic validation of output path base
        if not output_base_filename or not isinstance(output_base_filename, str):
             raise ValueError("Invalid output_path_base provided.")

        # Validate that 'questions' contains Question objects
        is_valid_question_list = isinstance(questions_list, list) and \
                                  (not questions_list or isinstance(questions_list[0], Question))
        if not is_valid_question_list:
            first_item_type = type(questions_list[0]).__name__ if questions_list else "None"
            self.logger.error(f"Invalid result type for 'questions': Expected List[Question], but first item is {first_item_type}")
            raise SemanticQAGenError(
                "Invalid result: 'questions' key must contain a list of Question model objects.")

        # ---- Get source filename from document info ----
        source_filename = "unknown_source.ext"
        doc_filename_from_info = document_info.get("filename")
        if doc_filename_from_info and isinstance(doc_filename_from_info, str):
             try:
                 source_filename = os.path.basename(doc_filename_from_info)
             except Exception as e: # Catch potential errors with weird paths
                 self.logger.warning(f"Could not extract basename from document filename '{doc_filename_from_info}': {e}")
        # ---- End Get source filename ----


        try:
            # Get the configured output formatter from the pipeline
            output_formatter = getattr(self.pipeline, 'output_formatter', None)
            if not output_formatter:
                raise SemanticQAGenError("Internal state error: OutputFormatter not found in pipeline.")

            # Determine the output format to use (override or config default)
            effective_format = (format_name or self.config.output.format).lower()

            # Resolve the final output path using ProjectManager
            # Pass the base name (e.g., "my_doc"), let formatter add ext.
            resolved_output_base_path = self.project_manager.resolve_output_path(output_base_filename)


            # ---- Prepare list of dictionaries for the OutputFormatter ----
            output_data_list: List[Dict[str, Any]] = []
            for i, q in enumerate(questions_list):
                # Double-check type just in case (though validated above)
                if not isinstance(q, Question):
                    self.logger.warning(f"Skipping item {i} in questions list - not a Question object: {type(q)}. Item: {q}")
                    continue

                try:
                    # Start building the output dictionary for this question
                    output_dict: Dict[str, Any] = {}
                    output_dict['id'] = str(q.id)        # Ensure ID is string
                    output_dict['input'] = q.text         # RENAME 'text' to 'input'
                    output_dict['output'] = q.answer      # RENAME 'answer' to 'output'

                    # Extract context and other metadata added by QuestionGenerator
                    q_metadata = q.metadata if isinstance(q.metadata, dict) else {}
                    output_dict['context'] = q_metadata.get('context', "") # Mandatory field, default empty
                    output_dict['category'] = q.category     # Mandatory field from Question

                    # Prepare the nested 'metadata' dictionary
                    nested_metadata: Dict[str, Any] = {
                        'chunk_id': str(q.chunk_id), # Ensure ID is string
                        'source_filename': source_filename,
                        # Get other fields added by QuestionGenerator, default to None if missing
                        'page_number': q_metadata.get('page_number'),
                        'preceding_headings': q_metadata.get('preceding_headings', []), # Default empty list
                        'generation_order': q_metadata.get('generation_order')
                    }

                    # Add any *other* key-value pairs from the original question metadata
                    # that weren't explicitly handled above. Avoids losing extra info.
                    standard_meta_keys = {'context', 'page_number', 'preceding_headings', 'generation_order'}
                    for meta_key, meta_val in q_metadata.items():
                        if meta_key not in standard_meta_keys and meta_key not in nested_metadata:
                            nested_metadata[meta_key] = meta_val


                    # Store the cleaned nested metadata (remove None values)
                    output_dict['metadata'] = {k: v for k, v in nested_metadata.items() if v is not None}

                    output_data_list.append(output_dict)

                except AttributeError as attr_err:
                     # Catch errors if Question object is missing expected fields
                     self.logger.error(f"Attribute error formatting Question {getattr(q, 'id', '?')}: {attr_err}. Skipping question.")
                     continue # Skip this question
                except Exception as format_err:
                    # General catch for unexpected errors formatting a single question
                    self.logger.warning(f"Failed to format Question {getattr(q, 'id', '?')} for output: {format_err}. Skipping question.")
                    continue # Skip this question
            # ---- End preparing list ----

            # Log if no questions were successfully formatted
            if not output_data_list and questions_list:
                 self.logger.warning(f"No questions were successfully formatted for output from the {len(questions_list)} provided for {output_path_base}.")
                 # Allow formatter to proceed, it might handle empty lists gracefully (e.g., create empty file)

            # Call the OutputFormatter to handle actual file writing
            # Pass the list of formatted dictionaries
            saved_path = output_formatter.format_and_save(
                questions=output_data_list,          # The list of dicts we just created
                document_info=document_info,        # Pass doc info for potential header/metadata
                statistics=statistics,              # Pass stats for potential header/metadata
                output_path_base=resolved_output_base_path, # Base path for output file
                format_name=effective_format        # The format ('json', 'csv', 'jsonl')
            )

            self.logger.info(f"Questions saved to {saved_path} in '{effective_format}' format.")
            # Return the absolute path to the saved file
            return os.path.abspath(saved_path) # Ensure path is absolute

        except OutputError as e:
            # Catch specific output errors from the formatter/adapter
            self.logger.error(f"Output error saving questions for {output_base_filename}: {e}")
            raise # Re-raise OutputError
        except ValueError as e:
            # Catch specific ValueErrors (e.g., from path validation)
            self.logger.error(f"Value error during question saving for {output_base_filename}: {e}")
            raise OutputError(f"Value error during save: {e}") from e # Wrap as OutputError
        except Exception as e:
            # Catch unexpected errors during the saving process setup
            self.logger.error(f"Unexpected error saving questions for {output_base_filename}: {e}", exc_info=True)
            raise SemanticQAGenError(f"Failed to save questions: {e}") from e


    def create_default_config_file(self, output_path: Optional[str] = None,
                                   include_comments: bool = True) -> None:
        """
        Creates a default configuration YAML file with comments in the project.

        Args:
            output_path: Optional path for the config file. If None, uses the
                         default 'system.yaml' in the project's config dir.
                         If relative, treated relative to project's config dir.
            include_comments: Whether to include comments in the config file.

        Raises:
            ConfigurationError: If config generation or saving fails.
            SemanticQAGenError: For unexpected errors.
        """
        try:
            # Determine the target path for the config file
            if output_path:
                # Resolve relative to config dir if not absolute
                target_path = self.project_manager.resolve_config_path(os.path.basename(output_path)) \
                               if not os.path.isabs(output_path) else output_path
            else:
                # Default path
                target_path = self.project_manager.resolve_config_path() # Gets default system.yaml path

            # Ensure the directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Check if file already exists (ConfigManager create method handles this, but good practice)
            if os.path.exists(target_path):
                 self.logger.info(f"Configuration file already exists at {target_path}. Not overwriting.")
                 return

            # Delegate the actual creation to ConfigManager
            # It will generate the default content appropriately
            # We instantiate a *new* ConfigManager here just for this purpose
            temp_config_manager = ConfigManager() # Gets default config content
            temp_config_manager.create_default_config_file(target_path, include_comments)

            self.logger.info(f"Default configuration file created at: {target_path}")

        except ConfigurationError as e:
             # ConfigManager raises this on save failure
             self.logger.error(f"Failed creating default config file at '{output_path or 'default'}': {e}")
             raise
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

