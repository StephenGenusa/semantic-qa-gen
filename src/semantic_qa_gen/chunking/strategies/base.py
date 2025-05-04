"""Base chunking strategy for SemanticQAGen."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from semantic_qa_gen.document.models import Document, Section, Chunk
from semantic_qa_gen.utils.error import ChunkingError


class BaseChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    Chunking strategies break documents into semantically coherent chunks
    for question generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunking strategy.
        
        Args:
            config: Configuration dictionary for the strategy.
        """
        self.config = config or {}
    
    @abstractmethod
    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """
        Break a document into chunks.
        
        Args:
            document: Document to chunk.
            sections: Preprocessed document sections.
            
        Returns:
            List of document chunks.
            
        Raises:
            ChunkingError: If the document cannot be chunked.
        """
        pass
    
    def get_context_for_chunk(self, chunk_text: str, document: Document, 
                             preceding_headings: List[Section]) -> Dict[str, Any]:
        """
        Get context information for a chunk.
        
        Args:
            chunk_text: The chunk text.
            document: Source document.
            preceding_headings: Headings that precede this chunk.
            
        Returns:
            Dictionary containing context information.
        """
        # Extract title from metadata or first heading
        title = None
        if document.metadata and document.metadata.title:
            title = document.metadata.title
        elif preceding_headings and preceding_headings[0].level == 1:
            title = preceding_headings[0].content
            
        # Extract section information
        section_path = []
        current_levels = {}
        
        for heading in preceding_headings:
            level = heading.level
            current_levels[level] = heading.content
            
            # Remove any lower levels when a new heading is encountered
            for l in list(current_levels.keys()):
                if l > level:
                    del current_levels[l]
        
        # Build section path from highest to lowest level
        for level in sorted(current_levels.keys()):
            section_path.append(current_levels[level])
            
        return {
            'title': title,
            'section_path': section_path,
            'document_type': document.doc_type,
            'metadata': document.metadata.model_dump(exclude_none=True) if document.metadata else {}
        }
