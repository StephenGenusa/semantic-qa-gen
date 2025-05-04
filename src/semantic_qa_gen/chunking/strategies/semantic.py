# chunking/strategies/semantic.py
"""Semantic chunking strategy for SemanticQAGen."""

import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.document.models import Document, Section, Chunk, SectionType
from semantic_qa_gen.utils.error import ChunkingError


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks documents based on semantic boundaries and structure.

    Attempts to create chunks by respecting document sections (headings, paragraphs)
    provided by the DocumentProcessor. It aims to split at logical breakpoints
    while adhering to configured minimum, target, and maximum chunk sizes. Includes
    overlap between chunks to maintain context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic chunking strategy.

        Args:
            config: Configuration dictionary. Expected keys include:
                    'target_chunk_size', 'min_chunk_size', 'max_chunk_size',
                    'overlap_size', 'preserve_headings'.
        """
        super().__init__(config)
        # Set strategy-specific parameters with defaults
        self.target_chunk_size = self.config.get('target_chunk_size', 1500)
        self.min_chunk_size = self.config.get('min_chunk_size', 500)
        self.max_chunk_size = self.config.get('max_chunk_size', 2500)
        self.overlap_size = self.config.get('overlap_size', 150) # Added overlap config
        # `preserve_headings` affects how aggressively we split before major headings
        self.preserve_headings = self.config.get('preserve_headings', True)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"SemanticChunkingStrategy initialized with: target={self.target_chunk_size}, "
                         f"min={self.min_chunk_size}, max={self.max_chunk_size}, "
                         f"overlap={self.overlap_size}, preserve_headings={self.preserve_headings}")


    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """
        Break a document, represented by its sections, into semantically coherent chunks.

        Args:
            document: The source Document object.
            sections: A list of Section objects representing the document structure.

        Returns:
            A list of Chunk objects.

        Raises:
            ChunkingError: If chunking fails unexpectedly.
        """
        if not sections:
            return []

        chunks: List[Chunk] = []
        current_sections: List[Section] = []
        current_size_estimate = 0 # Use character count for estimation
        sequence_num = 0
        preceding_headings: List[Section] = []
        last_chunk_text: Optional[str] = None # Store text of the last created chunk for overlap

        for section in sections:
            # Estimate section size (simple character count for now)
            section_size = len(section.content) if section.content else 0

            # --- Heading Management ---
            # Track heading hierarchy for context and potential splitting
            is_major_heading = False
            if section.section_type in [SectionType.HEADING, SectionType.TITLE]:
                is_major_heading = section.level <= 2 # Consider H1/H2 major

                # Update preceding_headings list: remove lower/equal level headings, add current
                preceding_headings = [h for h in preceding_headings if h.level < section.level]
                preceding_headings.append(section)

                # --- Split Before Major Heading? ---
                # If preserving headings, a major heading is found, and we have decent content
                if self.preserve_headings and is_major_heading and current_size_estimate >= self.min_chunk_size:
                    self.logger.debug(f"Splitting chunk before major heading (L{section.level}): '{section.content[:50]}...'")
                    # Create chunk from current sections *before* this heading
                    chunk_text = self._combine_sections(current_sections)
                    # Don't include the current heading in the *previous* chunk's context
                    context_headings = [h for h in preceding_headings if h is not section]
                    chunk = self._create_chunk(chunk_text, document, sequence_num, context_headings)
                    chunks.append(chunk)
                    last_chunk_text = chunk_text # Store text for overlap
                    sequence_num += 1

                    # Start new chunk *with overlap* and the current heading section
                    overlap_text = self._get_overlap_text(last_chunk_text)
                    current_sections = [section]
                    current_size_estimate = len(overlap_text) + section_size if overlap_text else section_size
                    # Store overlap text to be prepended when the *next* chunk is finalized
                    # Or, modify _create_chunk? Simpler: prepend overlap now
                    # Let's refine this: Apply overlap when chunk is CREATED.
                    # Modify _create_chunk to accept optional previous_chunk_text.
                    # Reset state:
                    current_sections = [section]
                    current_size_estimate = section_size
                    continue # Move to the next section

            # --- Check Max Size ---
            # If adding this section *would* exceed max size, create chunk *before* adding
            if current_size_estimate > 0 and (current_size_estimate + section_size) > self.max_chunk_size:
                 # Only create a chunk if the current accumulation is reasonably sized
                 if current_size_estimate >= self.min_chunk_size:
                    self.logger.debug(f"Max chunk size ({self.max_chunk_size}) reached. Creating chunk.")
                    chunk_text = self._combine_sections(current_sections)
                    # Pass last_chunk_text to _create_chunk for potential overlap calculation
                    chunk = self._create_chunk(chunk_text, document, sequence_num, preceding_headings, last_chunk_text)
                    chunks.append(chunk)
                    last_chunk_text = chunk_text # Update last chunk text
                    sequence_num += 1

                    # Reset state for the new chunk, starting with the current section
                    current_sections = [section]
                    current_size_estimate = section_size
                    continue

            # --- Add Section to Current Chunk ---
            current_sections.append(section)
            current_size_estimate += section_size

            # --- Check Target Size & Find Break Point ---
            # If we've reached the target size, look for a good place to split *within* current sections
            if current_size_estimate >= self.target_chunk_size:
                # Find a semantic break point *within* the accumulated sections
                break_point_index = self._find_semantic_break_point(current_sections)

                if break_point_index is not None and break_point_index > 0:
                    self.logger.debug(f"Target size ({self.target_chunk_size}) reached. Splitting at section index {break_point_index}.")
                    # Sections *before* the break point form the chunk
                    chunk_part = current_sections[:break_point_index]
                    # Sections *after* the break point start the next potential chunk
                    remaining_part = current_sections[break_point_index:]

                    chunk_text = self._combine_sections(chunk_part)
                    chunk = self._create_chunk(chunk_text, document, sequence_num, preceding_headings, last_chunk_text)
                    chunks.append(chunk)
                    last_chunk_text = chunk_text
                    sequence_num += 1

                    # Reset state with the remaining sections
                    current_sections = remaining_part
                    current_size_estimate = sum(len(s.content) for s in current_sections if s.content)

        # --- Final Chunk ---
        # Add any remaining sections as the last chunk
        if current_sections:
            self.logger.debug("Adding final remaining sections as the last chunk.")
            chunk_text = self._combine_sections(current_sections)
            chunk = self._create_chunk(chunk_text, document, sequence_num, preceding_headings, last_chunk_text)
            chunks.append(chunk)

        return chunks

    def _combine_sections(self, sections: List[Section]) -> str:
        """Combine section contents into a single text string."""
        result_parts = []
        for section in sections:
            if not section.content:
                continue
            # Add extra newline after headings for better visual separation
            separator = "\n\n" if section.section_type in [SectionType.HEADING, SectionType.TITLE] else "\n"
            result_parts.append(section.content + separator)

        # Join parts and remove trailing separators/whitespace
        return "".join(result_parts).strip()


    def _find_semantic_break_point(self, sections: List[Section]) -> Optional[int]:
        """
        Find a suitable index within the list of sections to create a chunk break.

        Prioritizes breaking *after* paragraphs or headings, favoring the later
        part of the section list to ensure chunks reach near target size.

        Args:
            sections: The list of Section objects currently accumulated for a potential chunk.

        Returns:
            The index *after* which the split should occur (so the section *at* the returned
            index becomes the start of the *next* chunk), or None if no suitable break found.
        """
        total_sections = len(sections)
        if total_sections < 2: # Cannot split if fewer than 2 sections
            return None

        # Start looking for breaks from about 2/3rds of the way through, backwards
        # We want to break *after* a good point.
        start_search_index = max(1, int(total_sections * 2 / 3))

        # Look backwards from the end for a good split point (after paragraph/heading)
        for i in range(total_sections - 2, start_search_index - 1, -1):
            # Check if section 'i' is a good place to END a chunk
            if sections[i].section_type in [SectionType.HEADING, SectionType.PARAGRAPH]:
                 # Check if next section 'i+1' is a heading (good start for next chunk)
                 if sections[i+1].section_type in [SectionType.HEADING, SectionType.TITLE]:
                       return i + 1 # Split after section i
                 # If next isn't a heading, split anyway if section 'i' was a paragraph/heading end
                 return i + 1 # Split after section i

        # If no ideal break found in the latter part, consider breaking earlier if needed
        # This part might need refinement based on behavior. For now, prefer not splitting
        # if no good semantic break found near target size. Let it potentially exceed target.
        # Fallback: Maybe split near the middle only if the list is getting very long?
        # if total_sections > 10: # Arbitrary threshold
        #    return total_sections // 2

        # No suitable semantic break point found based on current criteria
        self.logger.debug("No suitable semantic break point found in current section batch.")
        return None

    def _get_overlap_text(self, text: Optional[str]) -> str:
        """
        Get the text to use as overlap from the end of the previous chunk.

        Tries to find sentence boundaries to make the overlap more coherent.

        Args:
            text: The full text content of the *previous* chunk.

        Returns:
            Text segment to prepend to the *next* chunk as overlap.
        """
        if not text or self.overlap_size <= 0:
            return ""

        # Ensure overlap size doesn't exceed text length
        effective_overlap = min(len(text), self.overlap_size)

        # Extract the candidate overlap text from the end
        overlap_candidate = text[-effective_overlap:]

        # Try to find the last sentence boundary within the overlap candidate
        # Looking for [.!?] followed by space or end of string
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s*$', overlap_candidate)]
        if sentence_ends:
            # If a sentence ends right at the end, overlap might not be needed? Or take previous sentence end?
            # For simplicity, let's take text *after* the last boundary found *within* the overlap window.
            # This might mean the overlap is shorter than requested.
            last_boundary_pos_in_candidate = max(sentence_ends)
            # Find where this boundary *was* in the original text to get text *after* it
            original_boundary_pos = len(text) - effective_overlap + last_boundary_pos_in_candidate
            # Extract text from *after* the boundary to the end of the original text
            # However, this doesn't work well. Let's rethink.

            # Simpler: Find the *first* sentence start within the overlap window.
            sentence_starts = [m.start() for m in re.finditer(r'\.\s+[A-Z]', overlap_candidate)] # Starts after ". Caps"
            if sentence_starts:
                  first_start_in_candidate = min(sentence_starts)
                  # Overlap starts from this point
                  overlap_text = overlap_candidate[first_start_in_candidate + 2 :] # +2 to skip ". "
                  self.logger.debug(f"Overlap generated from sentence start within window ({len(overlap_text)} chars).")
                  return overlap_text.strip()

        # Fallback: If no clear sentence boundary found, just take the last N characters
        self.logger.debug(f"Overlap generated using fixed character count ({effective_overlap} chars).")
        return overlap_candidate.strip() # Return the full overlap candidate

    def _create_chunk(self, text: str, document: Document, sequence: int,
                      preceding_headings: List[Section],
                      previous_chunk_text: Optional[str] = None) -> Chunk:
        """
        Create a Chunk object, including context and potential overlap.

        Args:
            text: The primary text content for this chunk.
            document: The source Document object.
            sequence: The sequence number of this chunk.
            preceding_headings: List of relevant Section objects (headings).
            previous_chunk_text: Text content of the immediately preceding chunk (for overlap).

        Returns:
            A populated Chunk object.
        """
        # Calculate context based on headings and document metadata
        context = self.get_context_for_chunk(text, document, preceding_headings)

        # Generate overlap text if applicable
        overlap_prefix = self._get_overlap_text(previous_chunk_text) if previous_chunk_text else ""

        # Prepend overlap to the main chunk text
        final_content = (overlap_prefix + "\n\n" + text).strip() if overlap_prefix else text.strip()
        # Add overlap information to context?
        context['overlap_chars_prepended'] = len(overlap_prefix)

        return Chunk(
            content=final_content,
            id=str(uuid.uuid4()),
            document_id=document.id,
            sequence=sequence,
            context=context,
            preceding_headings=preceding_headings.copy() # Ensure a copy is stored
        )

