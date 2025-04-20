"""Checkpoint management for SemanticQAGen."""

import os
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional
import logging

from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.document.models import Document, Chunk
from semantic_qa_gen.utils.error import SemanticQAGenError


class CheckpointError(SemanticQAGenError):
    """Exception raised for checkpoint errors."""
    pass


class CheckpointManager:
    """
    Manages checkpoints for resumable processing.
    
    This class allows the processing pipeline to save its state
    and resume from where it left off if interrupted.
    """
    
    def __init__(self, config: SemanticQAGenConfig):
        """
        Initialize the checkpoint manager.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.checkpoint_dir = config.processing.checkpoint_dir
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def save_checkpoint(self, document: Document, processed_chunks: List[Chunk],
                      current_chunk_idx: int, stats: Dict[str, Any]) -> str:
        """
        Save a processing checkpoint.
        
        Args:
            document: The document being processed.
            processed_chunks: Chunks that have been processed.
            current_chunk_idx: Index of the current chunk being processed.
            stats: Processing statistics.
            
        Returns:
            Path to the saved checkpoint file.
            
        Raises:
            CheckpointError: If the checkpoint cannot be saved.
        """
        try:
            # Create a uniquely identifiable checkpoint filename
            doc_hash = self._hash_document(document)
            timestamp = int(time.time())
            checkpoint_filename = f"checkpoint_{doc_hash}_{timestamp}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Prepare checkpoint data
            checkpoint_data = {
                "version": "1.0",
                "timestamp": timestamp,
                "document_id": document.id,
                "document_hash": doc_hash,
                "config_hash": self._hash_config(self.config),
                "current_chunk_idx": current_chunk_idx,
                "completed_chunks": [c.id for c in processed_chunks[:current_chunk_idx]],
                "statistics": stats,
            }
            
            # Save checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint for a document.
        
        Args:
            document: The document to load checkpoint for.
            
        Returns:
            Checkpoint data, or None if no checkpoint exists.
            
        Raises:
            CheckpointError: If the checkpoint cannot be loaded.
        """
        try:
            doc_hash = self._hash_document(document)
            checkpoint_files = []
            
            # Find all checkpoints for this document
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(f"checkpoint_{doc_hash}_") and filename.endswith(".json"):
                    checkpoint_files.append(filename)
            
            if not checkpoint_files:
                return None
                
            # Get the most recent checkpoint
            checkpoint_files.sort(reverse=True)  # Most recent first
            latest_checkpoint = checkpoint_files[0]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            
            # Load checkpoint data
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            # Verify this checkpoint matches the current configuration
            if checkpoint_data.get("config_hash") != self._hash_config(self.config):
                self.logger.warning("Configuration has changed since checkpoint was created")
                
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {str(e)}")
    
    def _hash_document(self, document: Document) -> str:
        """
        Create a hash of document content for identification.
        
        Args:
            document: Document to hash.
            
        Returns:
            Hash string.
        """
        return hashlib.md5(document.content.encode('utf-8')).hexdigest()[:10]
    
    def _hash_config(self, config: SemanticQAGenConfig) -> str:
        """
        Create a hash of configuration for verification.
        
        Args:
            config: Configuration to hash.
            
        Returns:
            Hash string.
        """
        config_str = json.dumps(config.dict(), sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:10]
