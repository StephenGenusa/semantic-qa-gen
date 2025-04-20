"""Fixed-size chunking strategy for SemanticQAGen."""

import uuid
from typing import List, Dict, Any, Optional

from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.document.models import Document, Section, Chunk, SectionType
from semantic_qa_gen.utils.error import ChunkingError


class FixedSizeChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy that chunks documents based on a fixed size.
    
    This is a simple baseline strategy that divides text into chunks
    of approximately equal size.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fixed-size chunking strategy.
        
        Args:
            config: Configuration dictionary for the strategy.
        """
        super().__init__(config)
        self.target_chunk_size = self.config.get('target_chunk_size', 1500)
        self.overlap_size = self.config.get('overlap_size', 150)
        self.preserve_headings = self.config.get('preserve_headings', True)
    
    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """
        Break a document into fixed-size chunks.
        
        Args:
            document: Document to chunk.
            sections: Preprocessed document sections.
            
        Returns:
            List of document chunks.
            
        Raises:
            ChunkingError: If the document cannot be chunked.
        """
        chunks = []
        current_chunk_text = ""
        current_chunk_size = 0
        sequence_num = 0
        preceding_headings = []
        
        for section in sections:
            # Keep track of headings
            if section.section_type == SectionType.HEADING or section.section_type == SectionType.TITLE:
                # Update the preceding headings list
                # Remove any headings of equal or lower level
                preceding_headings = [h for h in preceding_headings if h.level < section.level]
                # Add the new heading
                preceding_headings.append(section)
                
                # If preserving headings and we already have content, start a new chunk
                if self.preserve_headings and current_chunk_size > 0:
                    # Create a chunk with the current content
                    chunk = self._create_chunk(
                        current_chunk_text, document, sequence_num, preceding_headings
                    )
                    chunks.append(chunk)
                    sequence_num += 1
                    
                    # Start a new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk_text)
                    current_chunk_text = overlap_text + section.content + "\n\n"
                    current_chunk_size = len(overlap_text) + len(section.content) + 2
                    continue
            
            # Add section content to the current chunk
            section_text = section.content
            if section.section_type == SectionType.HEADING:
                section_text += "\n\n"
            elif not section_text.endswith("\n"):
                section_text += " "
                
            # Check if adding this section would exceed the target size
            if current_chunk_size + len(section_text) > self.target_chunk_size and current_chunk_size > 0:
                # Create a chunk with the current content
                chunk = self._create_chunk(
                    current_chunk_text, document, sequence_num, preceding_headings
                )
                chunks.append(chunk)
                sequence_num += 1
                
                # Start a new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text
                current_chunk_size = len(overlap_text)
            
            # Add the section to the current chunk
            current_chunk_text += section_text
            current_chunk_size += len(section_text)
        
        # Add the final chunk if there's any content
        if current_chunk_size > 0:
            chunk = self._create_chunk(
                current_chunk_text, document, sequence_num, preceding_headings
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, document: Document, sequence: int, 
                    preceding_headings: List[Section]) -> Chunk:
        """
        Create a chunk from the given text.
        
        Args:
            text: Chunk text.
            document: Source document.
            sequence: Sequence number of the chunk.
            preceding_headings: Headings that precede this chunk.
            
        Returns:
            Chunk object.
        """
        context = self.get_context_for_chunk(text, document, preceding_headings)
        
        return Chunk(
            content=text.strip(),
            id=str(uuid.uuid4()),
            document_id=document.id,
            sequence=sequence,
            context=context,
            preceding_headings=preceding_headings.copy()
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get the text to use as overlap with the next chunk.
        
        Args:
            text: Current chunk text.
            
        Returns:
            Text to use as overlap.
        """
        if not text or len(text) <= self.overlap_size:
            return text
            
        # Try to find a sentence boundary for the overlap
        overlap_candidate = text[-self.overlap_size:]
        
        # Look for a sentence boundary (., !, ? followed by space or newline)
        import re
        sentence_boundaries = list(re.finditer(r'[.!?][\s\n]', overlap_candidate))
        if sentence_boundaries:
            # Use the last sentence boundary as the start of overlap
            last_boundary = sentence_boundaries[-1]
            start_pos = last_boundary.start() + 2  # +2 to include the punctuation and space
            overlap_text = overlap_candidate[start_pos:]
        else:
            # No sentence boundary found, use word boundary
            words = overlap_candidate.split()
            if words:
                # Use the last few words
                overlap_text = ' '.join(words[-3:])  # Use last 3 words
            else:
                # Fallback to character-based overlap
                overlap_text = overlap_candidate[-100:] if len(overlap_candidate) > 100 else overlap_candidate
                
        return overlap_text
