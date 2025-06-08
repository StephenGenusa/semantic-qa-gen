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
        """Formats Q&A pairs into CSV rows, including a 'context' column.

        Args:
            questions: A list of question dictionaries.
            document_info: Information about the source document.
            statistics: Processing statistics.

        Returns:
            A dictionary containing 'headers' and 'rows' for the CSV file,
            and the original 'statistics'.

        Raises:
            OutputError: If structuring the CSV data fails.
        """
        try:
            fine_tuning_format = self.config.get('fine_tuning_format', 'default')
            if fine_tuning_format == 'openai_legacy':
                standard_headers = ['id', 'prompt', 'completion', 'context', 'category', 'chunk_id']
            elif fine_tuning_format == 'standard':
                standard_headers = ['id', 'input', 'output', 'context', 'category', 'chunk_id']
            else:
                standard_headers = ['id', 'question', 'answer', 'context', 'category', 'chunk_id']

            key_metadata_fields = [
                'page_number', 'generation_index', 'information_density',
                'topic_coherence', 'complexity', 'validation_is_valid'
            ]

            metadata_direct_headers = []
            if questions:
                sample_size = min(10, len(questions))
                for q in questions[:sample_size]:
                    metadata = q.get('metadata', {})
                    for field in key_metadata_fields:
                        if field in metadata and field not in metadata_direct_headers:
                            metadata_direct_headers.append(field)
                    for field in ['validation', 'analysis']:
                        if field in metadata and isinstance(metadata[field], dict):
                            for nested_field, value in metadata[field].items():
                                if isinstance(value, (str, int, float,
                                                      bool)) and f"{field}_{nested_field}" not in metadata_direct_headers:
                                    metadata_direct_headers.append(f"{field}_{nested_field}")
                    if 'key_concepts' in metadata and isinstance(metadata['key_concepts'], list):
                        if 'key_concepts' not in metadata_direct_headers:
                            metadata_direct_headers.append('key_concepts')
                    if 'position_info' in metadata and isinstance(metadata['position_info'], dict):
                        for pos_field, value in metadata['position_info'].items():
                            pos_field_name = f"position_{pos_field}"
                            if pos_field_name not in metadata_direct_headers:
                                metadata_direct_headers.append(pos_field_name)

            metadata_headers = ['metadata_json']
            doc_headers = [f"document_{key}" for key in document_info.keys()] if self.include_doc_info else []
            all_headers = standard_headers + metadata_direct_headers + metadata_headers + doc_headers

            rows = []
            for q in questions:
                if not isinstance(q, dict): continue

                question_text = q.get('text', q.get('question', ''))
                answer_text = q.get('answer', '')
                metadata = self._process_metadata(q.get('metadata', {}))

                if fine_tuning_format == 'openai_legacy':
                    row_data = {
                        'id': q.get('id', ''), 'prompt': question_text, 'completion': answer_text,
                        'context': q.get('context', ''), 'category': q.get('category', ''),
                        'chunk_id': q.get('chunk_id', '')
                    }
                elif fine_tuning_format == 'standard':
                    row_data = {
                        'id': q.get('id', ''), 'input': question_text, 'output': answer_text,
                        'context': q.get('context', ''), 'category': q.get('category', ''),
                        'chunk_id': q.get('chunk_id', '')
                    }
                else:
                    row_data = {
                        'id': q.get('id', ''), 'question': question_text, 'answer': answer_text,
                        'context': q.get('context', ''), 'category': q.get('category', ''),
                        'chunk_id': q.get('chunk_id', '')
                    }

                for field in metadata_direct_headers:
                    if '_' in field and field.split('_', 1)[0] in ['validation', 'analysis']:
                        top_level, nested_field = field.split('_', 1)
                        if top_level in metadata and isinstance(metadata[top_level], dict):
                            row_data[field] = metadata[top_level].get(nested_field, '')
                    elif field == 'key_concepts' and field in metadata and isinstance(metadata[field], list):
                        row_data[field] = ', '.join(str(concept) for concept in metadata[field])
                    elif field.startswith('position_') and 'position_info' in metadata:
                        pos_field = field[9:]
                        if isinstance(metadata['position_info'], dict) and pos_field in metadata['position_info']:
                            pos_value = metadata['position_info'][pos_field]
                            row_data[field] = json.dumps(pos_value) if isinstance(pos_value, dict) else str(pos_value)
                    elif field in metadata:
                        row_data[field] = metadata.get(field, '')

                try:
                    metadata_json = json.dumps(metadata) if metadata else ""
                except TypeError:
                    metadata_json = json.dumps({"error": "cannot serialize metadata"})
                row_data['metadata_json'] = metadata_json

                if self.include_doc_info:
                    for key, value in document_info.items():
                        doc_key = f"document_{key}"
                        row_data[doc_key] = str(value) if value is not None else ""

                row = [row_data.get(h, '') for h in all_headers]
                rows.append(row)

            return {'headers': all_headers, 'rows': rows, 'statistics': statistics}

        except Exception as e:
            self.logger.error(f"Internal error formatting data for CSV: {e}", exc_info=True)
            raise OutputError(f"Failed to structure CSV output rows: {str(e)}")

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to standardize fields and paths.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Processed metadata dictionary with standardized fields
        """
        if not isinstance(metadata, dict):
            return metadata

        result = metadata.copy()

        # Convert source to basename if it exists in document_metadata
        if 'document_metadata' in result and isinstance(result['document_metadata'], dict):
            if 'source' in result['document_metadata']:
                import os
                result['document_metadata']['source'] = os.path.basename(result['document_metadata']['source'])

        # Ensure position info is captured and structured properly
        position_info = {}

        # Get page position from context if it exists
        if 'page_numbers' in result:
            position_info['page_numbers'] = result['page_numbers']
        if 'page_number' in result:
            position_info['page_number'] = result['page_number']

        # Look for position data in font_info or other fields
        for source_field in ['font_info', 'style_info']:
            if source_field in result and isinstance(result[source_field], dict):
                for key, value in result[source_field].items():
                    if isinstance(value, dict) and 'position' in value:
                        position_info[f"{key}_position"] = value['position']

        # Add position info if we found any
        if position_info and 'position_info' not in result:
            result['position_info'] = position_info

        return result

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

