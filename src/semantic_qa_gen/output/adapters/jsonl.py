# semantic_qa_gen/output/adapters/jsonl.py

"""JSON Lines (JSONL) format adapter for output formatting."""

import json
import os
import logging
from typing import Dict, List, Any, Optional

from semantic_qa_gen.output.formatter import FormatAdapter
from semantic_qa_gen.utils.error import OutputError


class JSONLAdapter(FormatAdapter):
    """
    Format adapter for JSON Lines (JSONL) output.

    Each question object is serialized as a separate JSON object on a new line.
    Document metadata and statistics are not included in the JSONL file itself,
    as the format is oriented towards record-based data processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSONL adapter.

        Args:
            config: Optional configuration dictionary (expects OutputConfig structure).
                    Relevant keys: json_ensure_ascii.
        """
        super().__init__(config)
        self.ensure_ascii = self.config.get('json_ensure_ascii', False)
        self.logger = logging.getLogger(__name__) # Use adapter-specific logger

    def format(self, questions: List[Dict[str, Any]],
               document_info: Dict[str, Any],
               statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formats questions for JSONL with field mapping for AI fine-tuning.

        Each question, now including its source context, is prepared as a dictionary
        to be written as a single line in the JSONL file.

        Args:
            questions: List of question dictionaries.
            document_info: Ignored by JSONL format.
            statistics: Ignored by JSONL format.

        Returns:
            A list of formatted question dictionaries.

        Raises:
            OutputError: If formatting the data fails.
        """
        try:
            # Get fine-tuning format from config
            fine_tuning_format = self.config.get('fine_tuning_format', 'default')

            formatted_questions = []
            for q in questions:
                if not isinstance(q, dict):
                    self.logger.warning(f"Skipping non-dictionary item in JSONL data: {type(q)}")
                    continue

                # Get the question and answer text
                question_text = q.get('text', q.get('question', ''))
                answer_text = q.get('answer', '')
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
                    record['context'] = source_context
                    if 'metadata' in q:
                        record['metadata'] = self._process_metadata(q['metadata'])

                elif fine_tuning_format == 'openai_legacy':
                    # Format for OpenAI legacy fine-tuning
                    record = {
                        "prompt": question_text,
                        "completion": answer_text
                    }
                    record['context'] = source_context
                    if 'metadata' in q:
                        record['metadata'] = self._process_metadata(q['metadata'])

                elif fine_tuning_format == 'standard':
                    # Common standard format - context is already included via q.copy()
                    record = {
                        "input": question_text,
                        "output": answer_text,
                        **{k: v for k, v in q.items() if k not in ['text', 'question', 'answer']}
                    }
                    if 'metadata' in record:
                        record['metadata'] = self._process_metadata(record['metadata'])

                else:
                    # Default: Keep original fields but ensure 'question' exists - context is already included
                    if 'text' in record and 'question' not in record:
                        record['question'] = record.pop('text')
                    if 'metadata' in record:
                        record['metadata'] = self._process_metadata(record['metadata'])

                formatted_questions.append(record)

            self.logger.debug(
                f"Prepared {len(formatted_questions)} question records for JSONL output with fine-tuning format: {fine_tuning_format}")
            return formatted_questions

        except Exception as e:
            self.logger.error(f"Internal error preparing data for JSONL: {e}", exc_info=True)
            raise OutputError(f"Failed to prepare JSONL output data: {str(e)}")

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

        # Ensure position info is included when available
        if 'position_info' not in result:
            # Extract any position data that might be in other fields
            position_data = {}

            # Get page position from context if it exists
            if 'page_numbers' in result:
                position_data['page_numbers'] = result['page_numbers']
            if 'page_number' in result:
                position_data['page_number'] = result['page_number']

            # Add specific position data if available
            for field in ['font_info', 'style_info']:
                if field in result and isinstance(result[field], dict):
                    for key, value in result[field].items():
                        if isinstance(value, dict) and any(pos_key in value for pos_key in ['position', 'x', 'y']):
                            position_data[f"{key}_position"] = value

            # Add position info if we found any
            if position_data:
                result['position_info'] = position_data
        return result

    def save(self, formatted_data: List[Dict[str, Any]], output_path: str) -> str:
        """
        Save the list of dictionaries as a JSONL file.

        Each dictionary in the list is written as a JSON string on a separate line.

        Args:
            formatted_data: List of question dictionaries returned by the format method.
            output_path: Full path where to save the output file (including extension).

        Returns:
            Path to the saved file (the input path).

        Raises:
            OutputError: If JSON serialization or file writing fails.
        """
        try:
            # Directory creation is handled by OutputFormatter.save_to_file
            with open(output_path, 'w', encoding='utf-8') as file:
                for record in formatted_data:
                    if not isinstance(record, dict):
                         self.logger.warning(f"Skipping non-dictionary item in JSONL data: {type(record)}")
                         continue
                    try:
                        # Dump each dictionary as a compact JSON string on one line
                        json_string = json.dumps(
                            record,
                            ensure_ascii=self.ensure_ascii,
                            separators=(',', ':') # Compact representation
                        )
                        file.write(json_string + '\n')
                    except TypeError as json_err:
                         # Log specific error for the record and continue if possible
                         record_id = record.get('id', 'unknown_id')
                         self.logger.error(f"Failed to serialize record {record_id} to JSON for JSONL: {json_err}. Skipping record.")
                         # Optionally, write an error placeholder? For now, skip.
                         # file.write(json.dumps({"error": "serialization_failed", "id": record_id}) + '\n')


            self.logger.info(f"Successfully saved {len(formatted_data)} records to JSONL file: {output_path}")
            return output_path # Return the confirmed save path

        except (IOError, OSError) as e:
            self.logger.error(f"File write error saving JSONL to {output_path}: {e}")
            raise OutputError(f"Failed to write JSONL output file: {str(e)}")
        except Exception as e:
            # Catch other potential errors during the save loop
            self.logger.exception(f"Unexpected error saving JSONL to {output_path}: {e}", exc_info=True)
            raise OutputError(f"Unexpected error saving JSONL output: {str(e)}")

    @property
    def file_extension(self) -> str:
        """Get the file extension for JSON Lines format."""
        return ".jsonl"


