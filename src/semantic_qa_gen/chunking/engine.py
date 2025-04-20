"""Chunking engine for SemanticQAGen."""

import logging
from typing import List, Optional, Dict, Any, Type

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Section, Chunk
from semantic_qa_gen.document.processor import DocumentProcessor
from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from semantic_qa_gen.chunking.strategies.semantic import SemanticChunkingStrategy
from semantic_qa_gen.utils.error import ChunkingError


class ChunkingEngine:
    """
    Engine for document chunking.
    
    This class manages the process of breaking documents into semantically
    coherent chunks for question generation.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the chunking engine.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.config = config_manager.get_section("chunking")
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.strategies: Dict[str, BaseChunkingStrategy] = {
            "fixed_size": FixedSizeChunkingStrategy(self.config.dict()),
            "semantic": SemanticChunkingStrategy(self.config.dict())
        }
        
        # Set the active strategy based on configuration
        self.active_strategy = self.strategies.get(self.config.strategy)
        if not self.active_strategy:
            raise ChunkingError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def chunk_document(self, document: Document, 
                      document_processor: DocumentProcessor) -> List[Chunk]:
        """
        Break a document into chunks.
        
        Args:
            document: Document to chunk.
            document_processor: DocumentProcessor instance for structural analysis.
            
        Returns:
            List of document chunks.
            
        Raises:
            ChunkingError: If the document cannot be chunked.
        """
        try:
            # Extract document sections
            sections = document_processor.extract_sections(document)
            
            # Use the active strategy to chunk the document
            self.logger.info(f"Chunking document with {self.config.strategy} strategy")
            chunks = self.active_strategy.chunk_document(document, sections)
            
            # Optimize chunks if needed
            chunks = self.optimize_chunks(chunks)
            
            self.logger.info(f"Document chunked into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {str(e)}")
    
    def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Optimize chunks after initial chunking.
        
        This method can merge small chunks, split large chunks, or
        adjust chunk boundaries for better semantic coherence.
        
        Args:
            chunks: List of chunks to optimize.
            
        Returns:
            List of optimized chunks.
        """
        # If no chunks, nothing to optimize
        if not chunks:
            return []
            
        # Check for and merge small chunks
        min_chunk_size = self.config.min_chunk_size
        optimized = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue
                
            # If current chunk is small and adding the next one wouldn't make it too large
            if (len(current_chunk.content) < min_chunk_size and 
                len(current_chunk.content) + len(chunk.content) <= self.config.max_chunk_size * 1.1):
                
                # Merge the chunks
                merged_content = current_chunk.content + "\n\n" + chunk.content
                
                # Preserve the first chunk's ID and metadata
                current_chunk.content = merged_content
                
                # Update context
                self._update_context_for_merged_chunk(current_chunk, chunk)
                
                # Skip adding the second chunk separately
                continue
            
            # Add the current chunk and move to the next
            optimized.append(current_chunk)
            current_chunk = chunk
        
        # Add the last chunk
        if current_chunk is not None:
            optimized.append(current_chunk)
        
        # Update sequence numbers
        for i, chunk in enumerate(optimized):
            chunk.sequence = i
            
        self.logger.info(f"Chunk optimization: {len(chunks)} chunks -> {len(optimized)} chunks")
        return optimized
    
    def _update_context_for_merged_chunk(self, target_chunk: Chunk, source_chunk: Chunk) -> None:
        """
        Update context information when merging chunks.
        
        Args:
            target_chunk: The chunk being merged into.
            source_chunk: The chunk being merged from.
        """
        # Ensure all headings are included
        target_headings = {h.content: h for h in target_chunk.preceding_headings}
        for heading in source_chunk.preceding_headings:
            if heading.content not in target_headings:
                target_chunk.preceding_headings.append(heading)
        
        # Sort headings by level and then by sequence of appearance
        target_chunk.preceding_headings.sort(key=lambda h: (h.level, source_chunk.preceding_headings.index(h) 
                                                      if h in source_chunk.preceding_headings else 999))
    
    def set_strategy(self, strategy_name: str) -> None:
        """
        Set the active chunking strategy.
        
        Args:
            strategy_name: Name of the strategy to use.
            
        Raises:
            ChunkingError: If the strategy does not exist.
        """
        if strategy_name not in self.strategies:
            raise ChunkingError(f"Unknown chunking strategy: {strategy_name}")
            
        self.active_strategy = self.strategies[strategy_name]
        self.logger.info(f"Chunking strategy set to: {strategy_name}")
    
    def register_strategy(self, name: str, strategy: BaseChunkingStrategy) -> None:
        """
        Register a new chunking strategy.
        
        Args:
            name: Name for the strategy.
            strategy: Strategy instance.
        """
        self.strategies[name] = strategy
        self.logger.info(f"Registered new chunking strategy: {name}")
