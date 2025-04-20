"""CSV format adapter for output formatting."""

import csv
import os
from typing import Dict, List, Any, Optional

from semantic_qa_gen.output.formatter import FormatAdapter
from semantic_qa_gen.utils.error import OutputError


class CSVAdapter(FormatAdapter):
    """
    Format adapter for CSV output.
    
    This adapter formats question-answer pairs as CSV for easy import
    into spreadsheets and other data processing tools.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CSV adapter.
        
        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.delimiter = self.config.get('csv_delimiter', ',')
        self.quotechar = self.config.get('csv_quotechar', '"')

    def format(self, questions: List[Dict[str, Any]],
               document_info: Dict[str, Any],
               statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format question-answer pairs for CSV output.

        Since CSV is a flat format, we need to structure the data differently.
        We'll return a dictionary with the rows and headers.

        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.

        Returns:
            Dictionary with CSV rows and headers.

        Raises:
            OutputError: If formatting fails.
        """
        try:
            # Create headers
            standard_headers = ['id', 'question', 'answer', 'category', 'chunk_id']

            # Sample the first question to get additional metadata fields
            metadata_headers = []
            if questions and self.config.get('include_metadata', True):
                if questions[0] and isinstance(questions[0], dict) and 'metadata' in questions[0]:
                    # Extract metadata fields from the first question
                    sample_metadata = questions[0].get('metadata', {})
                    if isinstance(sample_metadata, dict):
                        metadata_headers = list(sample_metadata.keys())

            # Document info headers (prefixed for clarity)
            doc_headers = []
            if self.config.get('include_document_info', True) and document_info:
                doc_headers = [f"document_{key}" for key in document_info.keys()]

            # Combined headers
            all_headers = standard_headers + metadata_headers + doc_headers

            # Create rows
            rows = []
            for q in questions:
                if not isinstance(q, dict):
                    continue

                # Standard fields
                row = [
                    q.get('id', ''),
                    q.get('text', q.get('question', '')),  # Accept either text or question field
                    q.get('answer', ''),
                    q.get('category', ''),
                    q.get('chunk_id', '')
                ]

                # Add metadata fields
                if metadata_headers:
                    q_metadata = q.get('metadata', {})
                    if isinstance(q_metadata, dict):
                        for h in metadata_headers:
                            row.append(q_metadata.get(h, ''))
                    else:
                        # Handle case where metadata is not a dict
                        for h in metadata_headers:
                            row.append('')

                # Add document info fields
                if doc_headers and document_info:
                    for key in document_info.keys():
                        row.append(document_info.get(key, ''))

                rows.append(row)

            return {
                'headers': all_headers,
                'rows': rows,
                'statistics': statistics  # Store this separately
            }

        except Exception as e:
            raise OutputError(f"Failed to format CSV output: {str(e)}")

    def save(self, formatted_data: Dict[str, Any], output_path: str) -> str:
        """
        Save CSV data to a file.
        
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
                
            # Write the main CSV file
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=self.delimiter, quotechar=self.quotechar)
                
                # Write headers
                writer.writerow(formatted_data['headers'])
                
                # Write data rows
                for row in formatted_data['rows']:
                    writer.writerow(row)
            
            # If configured, write statistics to a separate file
            if self.config.get('include_statistics', True):
                stats_path = f"{os.path.splitext(output_path)[0]}_stats.csv"
                self._write_statistics(formatted_data['statistics'], stats_path)
                
            return output_path
            
        except Exception as e:
            raise OutputError(f"Failed to save CSV output: {str(e)}")
    
    def _write_statistics(self, statistics: Dict[str, Any], output_path: str) -> None:
        """
        Write statistics to a separate CSV file.
        
        Args:
            statistics: Statistics dictionary.
            output_path: Path where to save the statistics.
            
        Raises:
            OutputError: If saving fails.
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=self.delimiter, quotechar=self.quotechar)
                
                # Write header
                writer.writerow(['Statistic', 'Value'])
                
                # Write general statistics
                for key, value in statistics.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (like categories)
                        for sub_key, sub_value in value.items():
                            writer.writerow([f"{key}_{sub_key}", sub_value])
                    else:
                        writer.writerow([key, value])
                        
        except Exception as e:
            self.logger.error(f"Failed to write statistics file: {str(e)}")
    
    @property
    def file_extension(self) -> str:
        """Get the file extension for CSV format."""
        return ".csv"
