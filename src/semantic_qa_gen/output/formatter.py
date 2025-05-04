# filename: semantic_qa_gen/output/formatter.py

"""Output formatting system for SemanticQAGen."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Type

# Use Pydantic V2 schema
from semantic_qa_gen.config.schema import OutputConfig
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Question # Keep for type hint flexibility, though input is dict
from semantic_qa_gen.utils.error import OutputError


class FormatAdapter(ABC):
    """
    Abstract base class for output format adapters.
    """
    # Store strongly-typed config subsection if needed
    # def __init__(self, config: Optional[OutputConfig] = None):
    # Or keep flexible dict for simpler adapter implementations initially
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the format adapter.

        Args:
            config: Optional configuration dictionary (relevant parts of OutputConfig).
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def format(self, questions: List[Dict[str, Any]],
              document_info: Dict[str, Any],
              statistics: Dict[str, Any]) -> Any:
        """
        Format question-answer pairs and related data.

        Args:
            questions: List of question dictionaries (serialized from Question objects).
            document_info: Information about the source document.
            statistics: Processing statistics.

        Returns:
            Formatted output ready for saving (e.g., Dict for JSON, structure for CSV).

        Raises:
            OutputError: If formatting fails.
        """
        pass

    @abstractmethod
    def save(self, formatted_data: Any, output_path: str) -> str:
        """
        Save formatted data to a file.

        Args:
            formatted_data: Data structured by the `format` method.
            output_path: Base path where to save the output (adapter adds extension).

        Returns:
            Full path to the saved file.

        Raises:
            OutputError: If saving fails.
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this format (e.g., '.json')."""
        pass


class OutputFormatter:
    """
    Manages formatting and exporting question-answer pairs.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the output formatter.

        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        # Get the specific OutputConfig section
        self.config: OutputConfig = config_manager.get_section("output")
        self.logger = logging.getLogger(__name__)

        # Initialize adapters
        self.adapters: Dict[str, FormatAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        """Initialize standard format adapters based on availability."""
        # Import adapters here to avoid potential circular imports at module level
        try:
            from semantic_qa_gen.output.adapters.json import JSONAdapter
            # Pass the raw dict view of the config section
            self.register_adapter("json", JSONAdapter(self.config.model_dump()))
        except ImportError:
            self.logger.warning("JSONAdapter not found or failed to import.")

        try:
            from semantic_qa_gen.output.adapters.csv import CSVAdapter
            self.register_adapter("csv", CSVAdapter(self.config.model_dump()))
        except ImportError:
            self.logger.warning("CSVAdapter not found or failed to import.")

        self.logger.info(f"Initialized {len(self.adapters)} output format adapter(s).")


    def register_adapter(self, name: str, adapter: FormatAdapter) -> None:
        """
        Register a format adapter.

        Args:
            name: Name for the adapter (case-insensitive).
            adapter: Adapter instance.

        Raises:
            OutputError: If an adapter with the same name already exists.
        """
        name_lower = name.lower()
        if name_lower in self.adapters:
            # Allow overwrite but log warning
            self.logger.warning(f"Overwriting previously registered format adapter: {name_lower}")
            # raise OutputError(f"Format adapter already registered: {name_lower}")

        if not isinstance(adapter, FormatAdapter):
             raise TypeError(f"Cannot register adapter '{name_lower}': Object must be instance of FormatAdapter.")

        self.adapters[name_lower] = adapter
        self.logger.debug(f"Registered format adapter: {name_lower} ({type(adapter).__name__})")


    def _get_adapter(self, format_name: Optional[str] = None) -> FormatAdapter:
        """Retrieves the requested or default format adapter."""
        target_format = (format_name or self.config.format).lower()
        adapter = self.adapters.get(target_format)
        if not adapter:
            available = list(self.adapters.keys())
            raise OutputError(f"Unknown or unavailable output format: '{target_format}'. Available: {available}")
        return adapter

    def format_questions(self,
                       questions: List[Dict[str, Any]],
                       document_info: Dict[str, Any],
                       statistics: Dict[str, Any],
                       format_name: Optional[str] = None) -> Any:
        """
        Format question-answer pairs using the specified adapter.

        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.
            format_name: Name of the format to use (defaults to config).

        Returns:
            Formatted output data structured by the adapter.

        Raises:
            OutputError: If formatting fails or the adapter doesn't exist.
        """
        adapter = self._get_adapter(format_name)
        try:
            formatted_data = adapter.format(
                questions=questions,
                document_info=document_info,
                statistics=statistics
            )
            return formatted_data
        except Exception as e:
            # Catch potential errors within adapter's format method
            self.logger.error(f"Adapter '{type(adapter).__name__}' failed during formatting: {e}", exc_info=True)
            raise OutputError(f"Failed to format output using {type(adapter).__name__}: {str(e)}") from e

    def save_to_file(self,
                     formatted_data: Any,
                     output_path_base: str, # Base path without extension
                     format_name: Optional[str] = None) -> str:
        """
        Save formatted data to a file using the appropriate adapter.

        Args:
            formatted_data: Data formatted by a format method.
            output_path_base: Base path for the output file (e.g., 'results/doc1').
                              The adapter will append the correct file extension.
            format_name: Name of the format to use (defaults to config).

        Returns:
            Full path to the saved file.

        Raises:
            OutputError: If saving fails or the adapter doesn't exist.
        """
        adapter = self._get_adapter(format_name)
        # Construct path with correct extension BEFORE creating directories
        final_output_path = f"{output_path_base}{adapter.file_extension}"

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(os.path.abspath(final_output_path))
            if output_dir: # Avoid trying to create dir if path has no directory part
                 os.makedirs(output_dir, exist_ok=True)

            # Delegate saving to the adapter, passing the full path
            saved_path = adapter.save(formatted_data, final_output_path)

            self.logger.info(f"Output saved successfully to {saved_path}")
            return saved_path # Return the path confirmed by the adapter

        except (IOError, OSError) as e:
            self.logger.error(f"File system error saving output to {final_output_path}: {e}")
            raise OutputError(f"Cannot write output file: {e}") from e
        except Exception as e:
            # Catch potential errors within adapter's save method
            self.logger.error(f"Adapter '{type(adapter).__name__}' failed during save: {e}", exc_info=True)
            raise OutputError(f"Failed to save output using {type(adapter).__name__}: {str(e)}") from e

    def format_and_save(self,
                      questions: List[Dict[str, Any]],
                      document_info: Dict[str, Any],
                      statistics: Dict[str, Any],
                      output_path_base: str,
                      format_name: Optional[str] = None) -> str:
        """
        Format and save question-answer pairs in one operation.

        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.
            output_path_base: Base path for the output file (extension added automatically).
            format_name: Name of the format to use (defaults to config).

        Returns:
            Full path to the saved file.

        Raises:
            OutputError: If formatting or saving fails.
        """
        effective_format = (format_name or self.config.format).lower()
        adapter = self._get_adapter(effective_format) # Get adapter once

        formatted_data = self.format_questions(
            questions=questions,
            document_info=document_info,
            statistics=statistics,
            format_name=effective_format # Pass effective format
        )

        return self.save_to_file(
            formatted_data=formatted_data,
            output_path_base=output_path_base,
            format_name=effective_format # Pass effective format again
        )

