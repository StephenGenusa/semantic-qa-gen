"""JSON format adapter for output formatting."""

import json
import os
from typing import Dict, List, Any, Optional

from semantic_qa_gen.output.formatter import FormatAdapter
from semantic_qa_gen.utils.error import OutputError


class JSONAdapter(FormatAdapter):
    """
    Format adapter for JSON output.
    
    This adapter formats question-answer pairs and related data
    as JSON for easy consumption by other systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON adapter.
        
        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.indent = self.config.get('indent', 2)
        self.ensure_ascii = self.config.get('ensure_ascii', False)
    
    def format(self, questions: List[Dict[str, Any]], 
              document_info: Dict[str, Any],
              statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format question-answer pairs as a JSON-serializable dictionary.
        
        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.
            
        Returns:
            Dictionary structured for JSON output.
            
        Raises:
            OutputError: If formatting fails.
        """
        try:
            # Create the output structure
            output = {
                "document": document_info,
                "questions": questions,
                "statistics": statistics
            }
            
            # Include metadata if configured
            if self.config.get('include_metadata', True):
                import datetime
                output["metadata"] = {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "generator": "SemanticQAGen",
                    "format_version": "1.0"
                }
                
            return output
            
        except Exception as e:
            raise OutputError(f"Failed to format JSON output: {str(e)}")
    
    def save(self, formatted_data: Dict[str, Any], output_path: str) -> str:
        """
        Save JSON data to a file.
        
        Args:
            formatted_data: Data formatted by the format method.
            output_path: Path where to save the output.
            
        Returns:
            Path to the saved file.
            
        Raises:
            OutputError: If saving fails.
        """
        try:
            # Ensure the file extension is correct
            if not output_path.endswith(self.file_extension):
                output_path = f"{output_path}{self.file_extension}"
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
            # Write the file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(formatted_data, file, indent=self.indent, ensure_ascii=self.ensure_ascii)
                
            return output_path
            
        except Exception as e:
            raise OutputError(f"Failed to save JSON output: {str(e)}")
    
    @property
    def file_extension(self) -> str:
        """Get the file extension for JSON format."""
        return ".json"
