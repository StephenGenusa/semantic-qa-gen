# filename: semantic_qa_gen/output/adapters/json.py

"""JSON format adapter for output formatting."""

import json
import os
import datetime # Keep for default metadata
from typing import Dict, List, Any, Optional

from semantic_qa_gen.output.formatter import FormatAdapter
from semantic_qa_gen.utils.error import OutputError


class JSONAdapter(FormatAdapter):
    """
    Format adapter for JSON output. Formats data as a single JSON object.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON adapter.

        Args:
            config: Optional configuration dictionary (expects OutputConfig structure).
        """
        super().__init__(config)
        # Access config keys safely with defaults
        self.indent = self.config.get('json_indent', 2)
        self.ensure_ascii = self.config.get('json_ensure_ascii', False)
        self.include_metadata = self.config.get('include_metadata', True)
        self.include_stats = self.config.get('include_statistics', True)

    def format(self, questions: List[Dict[str, Any]],
               document_info: Dict[str, Any],
               statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Structures the data into a dictionary suitable for JSON serialization.

        Args:
            questions: A list of question dictionaries, now including the 'context' field.
            document_info: Information about the source document.
            statistics: Processing statistics.

        Returns:
            A dictionary structured for JSON output, including the source context in each question record.

        Raises:
            OutputError: If structuring the data fails.
        """
        try:
            # Get fine-tuning format preference
            fine_tuning_format = self.config.get('fine_tuning_format', 'default')

            # Process questions for the selected fine-tuning format
            formatted_questions = []
            for q in questions:
                if not isinstance(q, dict):
                    continue

                # Get the question and answer text
                question_text = q.get('text', q.get('question', ''))
                answer_text = q.get('answer', '')
                # MODIFICATION: Get the context field
                source_context = q.get('context', '')

                # Create a copy of the original record for modification
                record = q.copy()

                # Apply standardized field mapping based on selected format
                if fine_tuning_format == 'openai_chat':
                    # Format for OpenAI chat fine-tuning (GPT models)
                    record = {
                        "messages": [
                            {"role": "user", "content": question_text},
                            {"role": "assistant", "content": answer_text}
                        ]
                    }
                    # MODIFICATION: Explicitly add context for this format
                    record['context'] = source_context
                    # Add metadata as a separate field
                    if 'metadata' in q:
                        record['metadata'] = self._process_metadata(q['metadata'])

                elif fine_tuning_format == 'openai_legacy':
                    # Format for OpenAI legacy fine-tuning
                    record = {
                        "prompt": question_text,
                        "completion": answer_text
                    }
                    # MODIFICATION: Explicitly add context for this format
                    record['context'] = source_context
                    # Add metadata as a separate field
                    if 'metadata' in q:
                        record['metadata'] = self._process_metadata(q['metadata'])

                elif fine_tuning_format == 'standard':
                    # Common standard format - context is already included via q.copy()
                    record = {
                        "input": question_text,
                        "output": answer_text,
                        # Keep other fields except text/question/answer
                        **{k: v for k, v in q.items() if k not in ['text', 'question', 'answer']}
                    }
                    if 'metadata' in record:
                        record['metadata'] = self._process_metadata(record['metadata'])

                else:
                    # Default: Keep original fields but standardize naming - context is already included
                    if 'text' in record:
                        record['question'] = record.pop('text')
                    if 'metadata' in record:
                        record['metadata'] = self._process_metadata(record['metadata'])

                formatted_questions.append(record)

            # Create the core output structure
            output = {
                "document": document_info,
                "questions": formatted_questions,
            }

            # Conditionally include statistics
            if self.include_stats:
                output["statistics"] = statistics

            # Conditionally include generator metadata
            if self.include_metadata:
                output["generation_metadata"] = {
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "generator": "SemanticQAGen",
                    "format_version": "1.2",
                    "fine_tuning_format": fine_tuning_format
                }

            return output

        except Exception as e:
            self.logger.error(f"Internal error formatting data for JSON: {e}", exc_info=True)
            raise OutputError(f"Failed to structure JSON output data: {str(e)}")

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to standardize fields and paths.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Processed metadata dictionary
        """
        if not isinstance(metadata, dict):
            return metadata

        result = metadata.copy()

        # Convert source to basename if it exists in document_metadata
        if 'document_metadata' in result and isinstance(result['document_metadata'], dict):
            if 'source' in result['document_metadata']:
                import os
                result['document_metadata']['source'] = os.path.basename(result['document_metadata']['source'])

        # Ensure position info is included when available
        if 'font_info' in result and 'position_info' not in result:
            # Extract any position data that might be in other fields
            position_data = {}

            # Get page position from context if it exists
            if 'page_numbers' in result:
                position_data['page_numbers'] = result['page_numbers']
            if 'page_number' in result:
                position_data['page_number'] = result['page_number']

            # Check for PDF-specific position info
            if result.get('document_metadata', {}).get('custom', {}).get('producer', '').startswith('PDF'):
                # Extract position from font_info if it contains position data
                for key, value in result.get('font_info', {}).items():
                    if isinstance(value, dict) and 'position' in value:
                        position_data[f"{key}_position"] = value['position']

            # Add position info if we found any
            if position_data:
                result['position_info'] = position_data

        return result

    def save(self, formatted_data: Dict[str, Any], output_path: str) -> str:
        """
        Save the dictionary data as a JSON file.

        Args:
            formatted_data: Dictionary returned by the format method.
            output_path: Full path where to save the output file (including extension).

        Returns:
            Path to the saved file (the input path).

        Raises:
            OutputError: If JSON serialization or file writing fails.
        """
        try:
            # Directory creation is handled by OutputFormatter.save_to_file
            # Write the file using json.dump
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(
                    formatted_data,
                    file,
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii
                 )
            return output_path # Return the confirmed save path

        except TypeError as e:
            # More specific error for serialization issues
             self.logger.error(f"JSON serialization failed for {output_path}: {e}. Check data types.", exc_info=True)
             raise OutputError(f"Failed to serialize data to JSON: {str(e)}")
        except (IOError, OSError) as e:
             self.logger.error(f"File write error saving JSON to {output_path}: {e}")
             raise OutputError(f"Failed to write JSON output file: {str(e)}")
        except Exception as e:
            self.logger.exception(f"Unexpected error saving JSON to {output_path}: {e}", exc_info=True)
            raise OutputError(f"Unexpected error saving JSON output: {str(e)}")

    @property
    def file_extension(self) -> str:
        """Get the file extension for JSON format."""
        return ".json"

