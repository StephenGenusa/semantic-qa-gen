# semantic_qa_gen/chunking/strategies/semantic.py
"""Semantic chunking strategy for SemanticQAGen."""

import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.document.models import Document, Section, Chunk, SectionType
from semantic_qa_gen.utils.error import ChunkingError
from semantic_qa_gen.chunking.strategies.nlp_helpers import find_section_boundaries


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy that chunks documents based on semantic boundaries.

    This strategy attempts to create chunks that maintain semantic coherence
    by respecting document structure and content boundaries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic chunking strategy.

        Args:
            config: Configuration dictionary for the strategy.
        """
        super().__init__(config)
        self.target_chunk_size = self.config.get('target_chunk_size', 1500)
        self.min_chunk_size = self.config.get('min_chunk_size', 500)
        self.max_chunk_size = self.config.get('max_chunk_size', 2500)
        self.overlap_size = self.config.get('overlap_size', 150)
        self.preserve_headings = self.config.get('preserve_headings', True)
        self.logger = logging.getLogger(__name__)

    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """
        Break a document into semantically coherent chunks.

        Args:
            document: Document to chunk.
            sections: Preprocessed document sections.

        Returns:
            List of document chunks.

        Raises:
            ChunkingError: If the document cannot be chunked.
        """
        chunks = []
        current_sections = []
        current_size = 0
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

                # If we have a significant heading and already have content, maybe start a new chunk
                if self.preserve_headings and section.level <= 2 and current_size >= self.min_chunk_size:
                    # Create a chunk with the current content
                    chunk_text = self._combine_sections(current_sections)
                    chunk = self._create_chunk(
                        chunk_text, document, sequence_num, preceding_headings[:-1]  # Exclude current heading
                    )
                    chunks.append(chunk)
                    sequence_num += 1

                    # Start a new chunk
                    current_sections = [section]
                    current_size = len(section.content)
                    continue

            # Calculate the size of this section
            section_size = len(section.content)

            # Check if adding this section would exceed the max size
            if current_size + section_size > self.max_chunk_size and current_size >= self.min_chunk_size:
                # Create a chunk with the current content
                chunk_text = self._combine_sections(current_sections)
                chunk = self._create_chunk(
                    chunk_text, document, sequence_num, preceding_headings
                )
                chunks.append(chunk)
                sequence_num += 1

                # Start a new chunk with the current section
                current_sections = [section]
                current_size = section_size
                continue

            # Add the section to the current chunk
            current_sections.append(section)
            current_size += section_size

            # Check if we've reached the target size and have a good breaking point
            if current_size >= self.target_chunk_size:
                break_point = self._find_semantic_break_point(current_sections)

                if break_point:
                    # Split at the break point
                    chunk_sections = current_sections[:break_point]
                    chunk_text = self._combine_sections(chunk_sections)
                    chunk = self._create_chunk(
                        chunk_text, document, sequence_num, preceding_headings
                    )
                    chunks.append(chunk)
                    sequence_num += 1

                    # Start a new chunk with the remaining sections
                    current_sections = current_sections[break_point:]
                    current_size = sum(len(s.content) for s in current_sections)

        # Add the final chunk if there's any content
        if current_sections:
            chunk_text = self._combine_sections(current_sections)
            chunk = self._create_chunk(
                chunk_text, document, sequence_num, preceding_headings
            )
            chunks.append(chunk)

        return chunks

    def _combine_sections(self, sections: List[Section]) -> str:
        """
        Combine sections into a single text.

        Args:
            sections: List of sections to combine.

        Returns:
            Combined text.
        """
        result = []

        for section in sections:
            if section.section_type == SectionType.HEADING:
                # Make headings stand out
                result.append(f"{section.content}\n")
            elif section.section_type == SectionType.PARAGRAPH:
                result.append(f"{section.content}")
            else:
                result.append(section.content)

            # Add appropriate spacing
            if not result[-1].endswith('\n'):
                result[-1] += '\n'
            if section.section_type == SectionType.HEADING:
                result[-1] += '\n'

        return ''.join(result).strip()

    def _find_semantic_break_point(self, sections: List[Section]) -> Optional[int]:
        """
        Find a semantically appropriate point to break the current sections.

        Args:
            sections: List of sections to analyze.

        Returns:
            Index where the break should occur, or None if no suitable break point found.
        """
        # Prefer breaking after a paragraph ends
        total_sections = len(sections)

        # Start from 2/3 of the way through
        start_idx = max(1, int(total_sections * 2 / 3))

        # Look for paragraph or heading breaks
        for i in range(start_idx, total_sections):
            if sections[i].section_type in [SectionType.HEADING, SectionType.PARAGRAPH]:
                if i < total_sections - 1:
                    return i + 1

        # Fallback: just break at the midpoint if sections list is getting long
        if total_sections > 5:
            return total_sections // 2

        # No good break point found
        return None

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

    def analyze_semantic_boundaries(self, text: str) -> List[Tuple[int, float]]:
        """
        Analyze semantic boundaries within text.

        This method identifies potential break points based on semantic cues.

        Args:
            text: Text to analyze.

        Returns:
            List of (position, strength) tuples representing potential break points.
        """
        boundaries = []

        # Look for paragraph breaks
        paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        for pos in paragraph_breaks:
            boundaries.append((pos, 0.9))  # High strength for paragraph breaks

        # Look for sentence breaks
        sentence_breaks = [m.start() for m in re.finditer(r'[.!?]\s', text)]
        for pos in sentence_breaks:
            # Don't add duplicates near paragraph breaks
            if not any(abs(pos - p) < 5 for p, _ in boundaries):
                boundaries.append((pos, 0.5))  # Medium strength for sentence breaks

        # Look for topic transition words
        transition_patterns = [
            r'\bHowever\b', r'\bMoreover\b', r'\bFurthermore\b', r'\bIn addition\b',
            r'\bConsequently\b', r'\bAs a result\b', r'\bIn contrast\b', r'\bNevertheless\b'
        ]
        for pattern in transition_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                pos = match.start()
                # Find the beginning of the sentence
                sentence_start = text.rfind('.', 0, pos)
                if sentence_start == -1:
                    sentence_start = 0
                else:
                    sentence_start += 1  # Move past the period

                # Don't add duplicates
                if not any(abs(sentence_start - p) < 10 for p, _ in boundaries):
                    boundaries.append((sentence_start, 0.7))  # High-medium strength

        return sorted(boundaries)


    def _find_chunk_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """
        Find semantically coherent chunk boundaries in text.

        Args:
            text: Text to chunk.

        Returns:
            List of (start, end) character indices for chunks.
        """
        # First try using the NLP helpers
        section_boundaries = find_section_boundaries(text)

        # Combine small sections and split large ones to meet target size
        boundaries = []
        current_chunk_start = 0
        current_chunk_size = 0

        for start, end in section_boundaries:
            section_size = self._estimate_tokens(text[start:end])

            if current_chunk_size + section_size <= self.target_chunk_size:
                # Add this section to the current chunk
                current_chunk_size += section_size
            else:
                # Current chunk is full, finalize it
                if current_chunk_size > 0:
                    boundaries.append((current_chunk_start, start))

                # Start a new chunk with this section
                if section_size > self.target_chunk_size:
                    # This section alone is too large, split it
                    split_boundaries = self._split_large_section(text, start, end)
                    boundaries.extend(split_boundaries)
                    current_chunk_start = end
                    current_chunk_size = 0
                else:
                    current_chunk_start = start
                    current_chunk_size = section_size

        # Add the last chunk if there's content remaining
        if current_chunk_size > 0:
            boundaries.append((current_chunk_start, len(text)))

        return boundaries

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Rough approximation: 1 token ≈ 4 characters in English
        return len(text) // 4
