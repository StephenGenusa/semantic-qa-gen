# filename: semantic_qa_gen/utils/checkpoint.py

"""Checkpoint management for SemanticQAGen."""

import os
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple

# Use Pydantic V2 schema and base model
from pydantic import BaseModel, ValidationError
from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.document.models import Document, Chunk, Question # Question needed for type hints/loading
from semantic_qa_gen.utils.error import SemanticQAGenError


class CheckpointError(SemanticQAGenError):
    """Exception raised for checkpoint errors."""
    pass


class CheckpointManager:
    """
    Manages checkpoints for resumable processing using Pydantic V2 models.

    Saves state including processed chunk count, statistics, and generated questions.
    Verifies config and document hashes for consistency on load.
    """

    CHECKPOINT_VERSION = "1.1" # Current checkpoint data structure version

    # In semantic_qa_gen/utils/checkpoint.py - modify the __init__ method

    def __init__(self, config: SemanticQAGenConfig):
        """
        Initialize the checkpoint manager.

        Args:
            config: The validated Pydantic V2 Application configuration object.
        """
        if not isinstance(config, SemanticQAGenConfig):
            raise CheckpointError("CheckpointManager requires a valid SemanticQAGenConfig object.")
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Access config attributes safely
        checkpoint_dir = config.processing.checkpoint_dir
        self.enable_checkpoints = config.processing.enable_checkpoints
        self.checkpoint_interval = config.processing.checkpoint_interval

        # Resolve checkpoint directory path - handle relative paths
        if not os.path.isabs(checkpoint_dir):
            # Try to find project root from current working directory
            from semantic_qa_gen.utils.project import ProjectManager
            project_manager = ProjectManager()
            project_root = project_manager.find_project_root()

            if project_root:
                # If checkpoint_dir starts with './', remove it
                if checkpoint_dir.startswith('./'):
                    checkpoint_dir = checkpoint_dir[2:]
                # Make absolute path relative to project root
                checkpoint_dir = os.path.join(project_root, checkpoint_dir)
                self.logger.debug(f"Resolved checkpoint directory relative to project: {checkpoint_dir}")

        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if enabled and doesn't exist
        if self.enable_checkpoints:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.logger.debug(f"Created checkpoint directory: {self.checkpoint_dir}")
            except OSError as e:
                # More informative error, but treat as fatal for init
                self.logger.critical(
                    f"Failed to create checkpoint directory '{self.checkpoint_dir}': {e}. Check permissions.")
                raise CheckpointError(f"Cannot create checkpoint directory: {e}")
        else:
            self.logger.info("Checkpoints are disabled in the configuration.")

    def _get_checkpoint_base_filename(self, document: Document) -> str:
        """Generates the base filename part using document hash."""
        doc_hash = self._hash_document(document)
        return f"checkpoint_{doc_hash}"

    def save_checkpoint(self,
                        document: Document,
                        processed_chunk_count: int,
                        stats: Dict[str, Any],
                        all_questions_so_far: Optional[List[Question]] = None) -> Optional[str]:
        """
        Save a processing checkpoint if enabled.

        Args:
            document: The document being processed.
            processed_chunk_count: Count of chunks fully processed (next to process is this index).
            stats: Current aggregated processing statistics.
            all_questions_so_far: List of *validated* Question objects generated so far.

        Returns:
            Path to the saved checkpoint file, or None if disabled or saving failed non-critically.

        Raises:
            CheckpointError: If checkpoint saving fails due to critical issues like invalid data.
                             Basic file I/O errors are logged and return None.
        """
        if not self.enable_checkpoints:
            return None

        try:
            timestamp = int(time.time())
            base_filename = self._get_checkpoint_base_filename(document)
            checkpoint_filename = f"{base_filename}_{timestamp}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

            # Convert Question objects to Pydantic-serializable dictionaries
            questions_data = []
            if all_questions_so_far:
                # Use model_dump for potential Pydantic V2 Question model later
                # For now, assumes Question is a dataclass/simple object
                questions_data = [q.to_dict() for q in all_questions_so_far if hasattr(q, 'to_dict')]
                # Fallback if no to_dict method - basic serialization might work
                if not questions_data and all_questions_so_far:
                     try:
                          # Attempt naive serialization, might fail for complex types in metadata
                          questions_data = [json.loads(json.dumps(q.__dict__, default=str)) for q in all_questions_so_far]
                          self.logger.warning("Used fallback serialization for Question objects in checkpoint.")
                     except Exception as serial_err:
                          self.logger.error(f"Failed to serialize Question objects for checkpoint: {serial_err}")
                          raise CheckpointError(f"Cannot serialize questions for checkpoint: {serial_err}")


            # Prepare checkpoint data structure (using current version)
            checkpoint_data = {
                "checkpoint_schema_version": self.CHECKPOINT_VERSION,
                "saved_at_timestamp": timestamp,
                "config_hash": self._hash_config(self.config), # Hash the Pydantic config model
                "document_id": document.id,
                "document_hash": self._hash_document(document),
                "processed_chunk_count": processed_chunk_count,
                "processed_qa_pairs_data": questions_data,
                "statistics": stats,
            }

            # Save checkpoint atomically (write to temp file, then rename)
            temp_checkpoint_path = checkpoint_path + ".tmp"
            try:
                with open(temp_checkpoint_path, 'w', encoding='utf-8') as f:
                    # Use Pydantic's JSON capability if data was a model, else standard json
                    json.dump(checkpoint_data, f, indent=2)

                # Atomically replace existing or create new
                os.replace(temp_checkpoint_path, checkpoint_path)

            except (IOError, OSError) as write_error:
                 self.logger.error(f"Error writing/renaming checkpoint file {checkpoint_path}: {write_error}")
                 # Cleanup temp file if it exists and saving failed
                 if os.path.exists(temp_checkpoint_path):
                     try: os.remove(temp_checkpoint_path)
                     except OSError: pass
                 return None # Non-critical IO error, allow pipeline to proceed
            except Exception as e:
                 # Catch unexpected errors during file writing/dumping
                 self.logger.exception(f"Unexpected error saving checkpoint state to {checkpoint_path}: {e}", exc_info=True)
                 if os.path.exists(temp_checkpoint_path):
                     try: os.remove(temp_checkpoint_path)
                     except OSError: pass
                 return None # Treat unexpected save errors as non-critical for pipeline continuation

            self.logger.info(f"Checkpoint saved: {checkpoint_path} (Processed chunks: {processed_chunk_count})")
            # Cleanup older checkpoints for the same document (optional)
            self._cleanup_old_checkpoints(document, keep=3) # Keep latest 3 checkpoints
            return checkpoint_path

        except CheckpointError: # Re-raise specific checkpoint logical errors
            raise
        except Exception as e:
             # Catch unexpected errors during checkpoint data *preparation*
             self.logger.exception(f"Unexpected error preparing checkpoint data: {e}", exc_info=True)
             raise CheckpointError(f"Failed preparing checkpoint data: {e}")


    def load_checkpoint(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Load the latest valid checkpoint for a document, verifying consistency.

        Args:
            document: The document to load checkpoint for.

        Returns:
            Checkpoint data dictionary if a valid one is found, otherwise None.

        Raises:
            CheckpointError: For critical errors during loading/verification.
                             Basic file IO or JSON parsing errors return None.
        """
        if not self.enable_checkpoints:
            return None

        try:
            base_filename = self._get_checkpoint_base_filename(document)
            potential_files = []

            # Scan directory for matching checkpoint files
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(base_filename) and filename.endswith(".json"):
                    try:
                        # Extract timestamp safely
                        parts = filename.rsplit('_', 1)
                        if len(parts) == 2:
                            timestamp = int(parts[1].replace('.json', ''))
                            potential_files.append((timestamp, os.path.join(self.checkpoint_dir, filename)))
                    except (ValueError, IndexError):
                         self.logger.warning(f"Skipping malformed checkpoint filename: {filename}")
                         continue

            if not potential_files:
                self.logger.info(f"No checkpoint files found for document hash {base_filename.split('_')[1]}.")
                return None

            # Sort by timestamp, newest first
            potential_files.sort(key=lambda x: x[0], reverse=True)

            # Try loading the latest valid checkpoint
            current_config_hash = self._hash_config(self.config)
            current_doc_hash = self._hash_document(document)

            for timestamp, checkpoint_path in potential_files:
                self.logger.debug(f"Attempting to load checkpoint: {checkpoint_path}")
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)

                    # --- Verification ---
                    # 1. Check schema version (optional, for future compatibility)
                    ckpt_version = checkpoint_data.get("checkpoint_schema_version")
                    if ckpt_version and ckpt_version != self.CHECKPOINT_VERSION:
                         self.logger.warning(f"Checkpoint {checkpoint_path} has schema version {ckpt_version}, expected {self.CHECKPOINT_VERSION}. Compatibility not guaranteed.")
                         # Decide: skip or attempt load? For now, attempt load.

                    # 2. Check document hash consistency
                    if checkpoint_data.get("document_hash") != current_doc_hash:
                        self.logger.warning(f"Checkpoint {checkpoint_path} document hash mismatch. File content may have changed. Skipping.")
                        continue

                    # 3. Check configuration hash
                    if checkpoint_data.get("config_hash") != current_config_hash:
                        self.logger.warning(f"Config hash mismatch in checkpoint {checkpoint_path}. "
                                            "Config likely changed since checkpoint. Proceeding cautiously (resume may be inaccurate).")
                        # Decide: skip or proceed? For now, allow proceeding with warning.

                    # 4. Basic structure check
                    required_keys = {"processed_chunk_count", "statistics", "processed_qa_pairs_data"}
                    if not required_keys.issubset(checkpoint_data.keys()):
                        self.logger.warning(f"Checkpoint {checkpoint_path} missing essential keys ({required_keys - set(checkpoint_data.keys())}). Skipping.")
                        continue

                    # All checks passed (or accepted with warnings)
                    self.logger.info(f"Successfully loaded and verified checkpoint: {os.path.basename(checkpoint_path)}")
                    return checkpoint_data

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in checkpoint {checkpoint_path}: {e}. Skipping.")
                    continue # Try the next older checkpoint
                except (IOError, OSError) as e:
                    self.logger.error(f"Cannot read checkpoint {checkpoint_path}: {e}. Skipping.")
                    continue
                except Exception as e: # Catch other potential errors during loading/verification
                    self.logger.exception(f"Unexpected error loading/verifying checkpoint {checkpoint_path}: {e}. Skipping.", exc_info=True)
                    continue

            # If loop finishes without loading a valid checkpoint
            self.logger.info(f"No valid & consistent checkpoint found for document hash {base_filename.split('_')[1]} after checking {len(potential_files)} file(s).")
            return None

        except OSError as e:
             self.logger.error(f"Error accessing checkpoint directory '{self.checkpoint_dir}': {e}")
             # Treat directory access errors as potentially critical? Or just return None?
             # Let's raise, as it indicates a filesystem issue preventing checkpointing.
             raise CheckpointError(f"Cannot access checkpoint directory: {e}")
        except Exception as e:
            # Catch other unexpected errors during the loading discovery process
            self.logger.exception(f"Unexpected error during checkpoint loading process: {e}", exc_info=True)
            raise CheckpointError(f"Failed to find/load checkpoint: {str(e)}")


    def load_questions_from_data(self, questions_data: List[Dict[str, Any]]) -> List[Question]:
        """Safely load Question objects from checkpoint data."""
        loaded_questions = []
        for i, q_data in enumerate(questions_data):
            try:
                # Requires Question model to support instantiation from dict
                # If Question uses Pydantic V2:
                # loaded_questions.append(Question.model_validate(q_data))
                # If Question is a dataclass:
                loaded_questions.append(Question(**q_data))
            except (TypeError, ValidationError, KeyError) as q_load_err: # Catch common errors
                 self.logger.warning(f"Failed to load question data item {i} from checkpoint data: {q_load_err}. Data snippet: {str(q_data)[:100]}")
            except Exception as e:
                 self.logger.error(f"Unexpected error loading question data item {i}: {e}. Data: {q_data}", exc_info=False)
        return loaded_questions


    def _cleanup_old_checkpoints(self, document: Document, keep: int = 3):
        """Remove older checkpoints for a document, keeping the specified number."""
        if not self.enable_checkpoints or keep <= 0:
            return

        try:
            base_filename = self._get_checkpoint_base_filename(document)
            checkpoint_files = []

            for filename in os.listdir(self.checkpoint_dir):
                 if filename.startswith(base_filename) and filename.endswith(".json") and ".tmp" not in filename:
                    try:
                        parts = filename.rsplit('_', 1)
                        if len(parts) == 2:
                             timestamp = int(parts[1].replace('.json', ''))
                             checkpoint_files.append((timestamp, os.path.join(self.checkpoint_dir, filename)))
                    except (ValueError, IndexError):
                         continue # Skip malformed names

            if len(checkpoint_files) <= keep:
                return # Not enough checkpoints to cleanup

            # Sort by timestamp, oldest first
            checkpoint_files.sort(key=lambda x: x[0])

            # Files to delete (all except the last 'keep' ones)
            files_to_delete = checkpoint_files[:-keep]
            delete_count = 0

            for timestamp, filepath in files_to_delete:
                 try:
                      os.remove(filepath)
                      self.logger.debug(f"Cleaned up old checkpoint: {os.path.basename(filepath)}")
                      delete_count += 1
                 except OSError as e:
                      self.logger.warning(f"Failed to cleanup old checkpoint {os.path.basename(filepath)}: {e}")

            if delete_count > 0:
                 self.logger.info(f"Cleaned up {delete_count} old checkpoint(s) for document hash {base_filename.split('_')[1]}.")

        except OSError as e:
             self.logger.error(f"Error accessing checkpoint directory for cleanup '{self.checkpoint_dir}': {e}")
        except Exception as e:
             # Catch other errors during cleanup but don't block main process
             self.logger.error(f"Unexpected error during checkpoint cleanup: {e}", exc_info=True)


    def _hash_document(self, document: Document) -> str:
        """Create a hash of document content for identification."""
        # Use path stable identifier and content hash
        # Use SHA256 for better collision resistance than MD5
        hasher = hashlib.sha256()
        # Include path for uniqueness if content is identical across files
        path_bytes = (document.path or "").encode('utf-8')
        content_bytes = document.content.encode('utf-8')
        hasher.update(path_bytes)
        hasher.update(b"::") # Separator
        hasher.update(content_bytes)
        return hasher.hexdigest()[:16] # Truncate hash for reasonable filename length


    def _hash_config(self, config: SemanticQAGenConfig) -> str:
        """Create a repeatable hash of relevant configuration Pydantic model."""
        try:
             # Use Pydantic V2's json dump for canonical representation
             # Exclude potentially volatile/irrelevant fields from the hash
             # CORRECTED exclude syntax for Pydantic V2:
             exclude_structure = {
                 'version': True,  # Exclude top-level 'version' field
                 'processing': {   # Specify exclusions within 'processing' field
                     'log_level': True,
                     'debug_mode': True
                 }
             }
             config_json_str = config.model_dump_json(
                 exclude=exclude_structure,
                 exclude_none=True # Keep excluding None values
             )
             # Use SHA256 for better collision resistance
             return hashlib.sha256(config_json_str.encode('utf-8')).hexdigest()[:16] # Truncated hash
        except Exception as e:
             self.logger.error(f"Failed to create configuration hash: {e}. Returning default hash.")
             # Return a default hash or raise? Default allows proceeding but with uncertainty.
             return "config_hash_error"
