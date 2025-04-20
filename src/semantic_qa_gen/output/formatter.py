"""Output formatting system for SemanticQAGen."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Type

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Question
from semantic_qa_gen.utils.error import OutputError


class FormatAdapter(ABC):
    """
    Abstract base class for output format adapters.
    
    Format adapters are responsible for converting question-answer pairs
    and related data into specific output formats like JSON, CSV, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the format adapter.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def format(self, questions: List[Dict[str, Any]], 
              document_info: Dict[str, Any],
              statistics: Dict[str, Any]) -> Any:
        """
        Format question-answer pairs and related data.
        
        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.
            
        Returns:
            Formatted output in the adapter's format.
            
        Raises:
            OutputError: If formatting fails.
        """
        pass
    
    @abstractmethod
    def save(self, formatted_data: Any, output_path: str) -> str:
        """
        Save formatted data to a file.
        
        Args:
            formatted_data: Data formatted by the format method.
            output_path: Path where to save the output.
            
        Returns:
            Path to the saved file.
            
        Raises:
            OutputError: If saving fails.
        """
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this format."""
        pass


class OutputFormatter:
    """
    Manager for formatting and exporting question-answer pairs.
    
    This class coordinates the process of formatting question-answer pairs
    and related data using different format adapters, and manages saving
    the formatted output to files.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the output formatter.
        
        Args:
            config_manager: Configuration manager.
        """
        self.config_manager = config_manager
        self.config = config_manager.get_section("output")
        self.logger = logging.getLogger(__name__)
        
        # Initialize adapters
        self.adapters: Dict[str, FormatAdapter] = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self) -> None:
        """Initialize standard format adapters."""
        # Import adapters here to avoid circular imports
        from semantic_qa_gen.output.adapters.json import JSONAdapter
        from semantic_qa_gen.output.adapters.csv import CSVAdapter
        
        # Register standard adapters
        self.register_adapter("json", JSONAdapter(self.config.dict()))
        self.register_adapter("csv", CSVAdapter(self.config.dict()))
    
    def register_adapter(self, name: str, adapter: FormatAdapter) -> None:
        """
        Register a format adapter.
        
        Args:
            name: Name for the adapter.
            adapter: Adapter instance.
            
        Raises:
            OutputError: If an adapter with the same name already exists.
        """
        if name in self.adapters:
            raise OutputError(f"Format adapter already registered: {name}")
            
        self.adapters[name] = adapter
        self.logger.debug(f"Registered format adapter: {name}")
    
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
            Formatted output.
            
        Raises:
            OutputError: If formatting fails or the adapter doesn't exist.
        """
        # Use the specified format or the one from config
        format_name = format_name or self.config.format
        
        # Get the adapter
        adapter = self.adapters.get(format_name)
        if not adapter:
            raise OutputError(f"Unknown output format: {format_name}")
        
        try:
            # Format the data
            formatted_data = adapter.format(
                questions=questions,
                document_info=document_info,
                statistics=statistics
            )
            
            return formatted_data
            
        except Exception as e:
            raise OutputError(f"Failed to format output: {str(e)}")

    def save_to_file(self,
                     formatted_data: Any,
                     output_path: str,
                     format_name: Optional[str] = None) -> str:
        """
        Save formatted data to a file using the appropriate adapter.

        Args:
            formatted_data: Data formatted by a format method.
            output_path: Path where to save the output.
            format_name: Name of the format to use (defaults to config).

        Returns:
            Path to the saved file.

        Raises:
            OutputError: If saving fails or the adapter doesn't exist.
        """
        # Use the specified format or the one from config
        format_name = format_name or self.config.format

        # Get the adapter
        adapter = self.adapters.get(format_name)
        if not adapter:
            raise OutputError(f"Unknown output format: {format_name}")

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save the data
            saved_path = adapter.save(formatted_data, output_path)

            self.logger.info(f"Output saved to {saved_path}")
            return saved_path

        except Exception as e:
            raise OutputError(f"Failed to save output: {str(e)}")
    
    def format_and_save(self, 
                      questions: List[Dict[str, Any]], 
                      document_info: Dict[str, Any],
                      statistics: Dict[str, Any],
                      output_path: str,
                      format_name: Optional[str] = None) -> str:
        """
        Format and save question-answer pairs in one operation.
        
        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.
            output_path: Path where to save the output.
            format_name: Name of the format to use (defaults to config).
            
        Returns:
            Path to the saved file.
            
        Raises:
            OutputError: If formatting or saving fails.
        """
        # Use the specified format or the one from config
        format_name = format_name or self.config.format
        
        # Format the data
        formatted_data = self.format_questions(
            questions=questions,
            document_info=document_info,
            statistics=statistics,
            format_name=format_name
        )
        
        # Save the data
        return self.save_to_file(
            formatted_data=formatted_data,
            output_path=output_path,
            format_name=format_name
        )
    
    def generate_statistics(self, 
                          questions: List[Dict[str, Any]], 
                          processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the question-answer pairs.
        
        Args:
            questions: List of question dictionaries.
            processing_stats: Processing statistics from earlier stages.
            
        Returns:
            Dictionary containing detailed statistics.
        """
        stats = {
            "total_questions": len(questions),
            "categories": {},
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "processing_info": processing_stats
        }
        
        # Calculate statistics
        if questions:
            # Count by category
            for question in questions:
                category = question.get("category")
                if category:
                    stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Calculate average lengths
            total_q_length = sum(len(q.get("text", "")) for q in questions)
            total_a_length = sum(len(q.get("answer", "")) for q in questions)
            
            stats["avg_question_length"] = total_q_length / len(questions)
            stats["avg_answer_length"] = total_a_length / len(questions)
        
        return stats
