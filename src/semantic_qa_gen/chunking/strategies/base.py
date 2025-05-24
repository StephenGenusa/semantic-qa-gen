"""Base chunking strategy for SemanticQAGen."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
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
        Get enhanced context information for a chunk, optimized for AI fine-tuning.

        Args:
            chunk_text: The chunk text.
            document: Source document.
            preceding_headings: Headings that precede this chunk.

        Returns:
            Dictionary containing comprehensive context information.
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

        # Track page numbers and structural metadata from sections
        page_numbers = set()
        font_info = {}
        style_info = {}
        position_info = {}

        for heading in preceding_headings:
            level = heading.level
            current_levels[level] = heading.content

            # Extract comprehensive metadata from section
            metadata = heading.metadata or {}

            # Collect page numbers from section metadata
            if 'page' in metadata:
                page_numbers.add(metadata['page'])

            # Collect font information (from PDF)
            if 'font_size' in metadata:
                font_info[f'heading_level_{level}_font_size'] = metadata['font_size']

            if 'font_name' in metadata:
                font_info[f'heading_level_{level}_font'] = metadata['font_name']

            if 'is_bold' in metadata:
                font_info[f'heading_level_{level}_bold'] = metadata['is_bold']

            # Collect style information (from DOCX)
            if 'style' in metadata:
                style_info[f'heading_level_{level}_style'] = metadata['style']

            # Collect position information
            if 'position' in metadata:
                if isinstance(metadata['position'], dict):
                    # Store the full position dictionary with all details
                    position_info[f'heading_level_{level}_position'] = metadata['position']
                else:
                    # DOCX style position (index) or other scalar value
                    position_info[f'heading_level_{level}_index'] = metadata['position']

            # Remove any lower levels when a new heading is encountered
            for l in list(current_levels.keys()):
                if l > level:
                    del current_levels[l]

        # Build section path from highest to lowest level
        for level in sorted(current_levels.keys()):
            section_path.append(current_levels[level])

        # Calculate text statistics for fine-tuning signals
        text_stats = self._get_text_statistics(chunk_text)

        # Create comprehensive context dict with enhanced metadata
        context = {
            'title': title,
            'section_path': section_path,
            'document_type': document.doc_type,
            'document_metadata': document.metadata.model_dump(exclude_none=True) if document.metadata else {}
        }

        # Ensure document source is just the filename, not full path
        if 'document_metadata' in context and 'source' in context['document_metadata']:
            import os
            context['document_metadata']['source'] = os.path.basename(context['document_metadata']['source'])

        # Add page number information - standardized field name for fine-tuning
        if page_numbers:
            context['page_numbers'] = list(sorted(page_numbers))
            context['page_number'] = min(page_numbers)  # For backward compatibility

        # Add structure information
        if font_info:
            context['font_info'] = font_info

        if style_info:
            context['style_info'] = style_info

        if position_info:
            context['position_info'] = position_info

        # Add text statistics for ML insights
        if text_stats:
            context['text_stats'] = text_stats

        return context

    def _get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate useful text statistics for fine-tuning signals."""
        if not text:
            return {}

        stats = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            # Simple readability estimate (Flesch-Kincaid approximation)
            'avg_word_length': sum(len(word) for word in text.split()) / max(1, len(text.split())),
        }

        # Estimate token count if possible (useful for LLM context)
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoder.encode(text))
            stats['token_count'] = token_count
        except (ImportError, Exception):
            # Unable to count tokens, skip this metric
            pass

        return stats
