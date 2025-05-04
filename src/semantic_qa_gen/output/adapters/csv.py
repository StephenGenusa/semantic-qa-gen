# filename: semantic_qa_gen/output/adapters/csv.py

"""CSV format adapter for output formatting."""

import csv
import os
import json # For serializing complex metadata/stats if needed
from typing import Dict, List, Any, Optional

from semantic_qa_gen.output.formatter import FormatAdapter
from semantic_qa_gen.utils.error import OutputError


class CSVAdapter(FormatAdapter):
    """
    Format adapter for CSV output. Writes questions to a main CSV file,
    and optionally statistics to a separate stats file.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CSV adapter.

        Args:
            config: Optional configuration dictionary (expects OutputConfig structure).
        """
        super().__init__(config)
        self.delimiter = self.config.get('csv_delimiter', ',')
        self.quotechar = self.config.get('csv_quotechar', '"')
        # Fetch CSV specific options
        self.include_doc_info = self.config.get('csv_include_document_info', False) # Default False
        self.include_stats = self.config.get('include_statistics', True)


    def format(self, questions: List[Dict[str, Any]],
               document_info: Dict[str, Any],
               statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format question-answer pairs into lists suitable for CSV writing.

        Args:
            questions: List of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.

        Returns:
            Dictionary containing 'headers' (list) and 'rows' (list of lists).

        Raises:
            OutputError: If formatting fails.
        """
        try:
            standard_headers = ['id', 'question', 'answer', 'category', 'chunk_id']
            # Simplify metadata handling: potentially serialize the whole metadata dict as JSON string
            metadata_headers = ['metadata_json']
            doc_headers = []
            if self.include_doc_info:
                # Flatten document info keys
                doc_headers = [f"document_{key}" for key in document_info.keys()]

            all_headers = standard_headers + metadata_headers + doc_headers

            rows = []
            for q in questions:
                if not isinstance(q, dict): continue # Skip invalid entries

                # Standard fields
                row_data = {
                    'id': q.get('id', ''),
                    'question': q.get('text', ''), # Use 'text' as canonical question field from dict
                    'answer': q.get('answer', ''),
                    'category': q.get('category', ''),
                    'chunk_id': q.get('chunk_id', '')
                }

                # Serialize metadata dict to JSON string
                metadata = q.get('metadata', {})
                try:
                    metadata_json = json.dumps(metadata) if metadata else ""
                except TypeError:
                    metadata_json = json.dumps({"error": "cannot serialize metadata"}) # Fallback
                row_data['metadata_json'] = metadata_json

                # Add document info if configured
                if self.include_doc_info:
                    for key, value in document_info.items():
                         doc_key = f"document_{key}"
                         # Ensure value is string representable for CSV
                         row_data[doc_key] = str(value) if value is not None else ""

                # Build row list in header order
                row = [row_data.get(h, '') for h in all_headers]
                rows.append(row)

            # Return structured data for saving
            return {
                'headers': all_headers,
                'rows': rows,
                # Pass stats through, saving handles writing separately
                'statistics': statistics
            }

        except Exception as e:
             self.logger.error(f"Internal error formatting data for CSV: {e}", exc_info=True)
             raise OutputError(f"Failed to structure CSV output rows: {str(e)}")

    def save(self, formatted_data: Dict[str, Any], output_path: str) -> str:
        """
        Save the data rows as a CSV file. Optionally saves stats separately.

        Args:
            formatted_data: Dictionary returned by the format method ('headers', 'rows', 'statistics').
            output_path: Full path where to save the main CSV file (including extension).

        Returns:
            Path to the saved main CSV file.

        Raises:
            OutputError: If file writing fails.
        """
        try:
            # Directory creation handled by OutputFormatter
            # Write the main CSV file
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(
                    file,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                    quoting=csv.QUOTE_MINIMAL # Quote only when necessary
                 )
                writer.writerow(formatted_data.get('headers', []))
                writer.writerows(formatted_data.get('rows', []))

            # Conditionally write statistics to a separate file
            if self.include_stats and 'statistics' in formatted_data:
                stats_path_base = os.path.splitext(output_path)[0]
                stats_path = f"{stats_path_base}_stats.json" # Save stats as JSON for easier parsing
                self._write_statistics_json(formatted_data['statistics'], stats_path)

            return output_path # Return the path of the main saved file

        except (IOError, OSError) as e:
             self.logger.error(f"File write error saving CSV to {output_path}: {e}")
             raise OutputError(f"Failed to write CSV output file: {str(e)}")
        except Exception as e:
             self.logger.exception(f"Unexpected error saving CSV output to {output_path}: {e}", exc_info=True)
             raise OutputError(f"Unexpected error saving CSV output: {str(e)}")

    # Changed to save stats as JSON for better structure preservation
    def _write_statistics_json(self, statistics: Dict[str, Any], output_path: str) -> None:
        """Write statistics dictionary to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(statistics, file, indent=2)
            self.logger.info(f"Statistics saved separately to {output_path}")
        except Exception as e:
            # Statistics saving is non-critical, just log error
            self.logger.error(f"Failed to write separate statistics file to {output_path}: {str(e)}")

    @property
    def file_extension(self) -> str:
        """Get the file extension for CSV format."""
        return ".csv"

