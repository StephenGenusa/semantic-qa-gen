# chunking/engine.py

"""Chunking engine for SemanticQAGen."""

import logging
import re # Import re for use in _update_context_for_merged_chunk if needed
from typing import List, Optional, Dict, Any, Type

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Section, Chunk
# Removed import of DocumentProcessor as it's no longer needed here
from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from semantic_qa_gen.chunking.strategies.semantic import SemanticChunkingStrategy
from semantic_qa_gen.utils.error import ChunkingError


class ChunkingEngine:
    """
    Engine for managing and applying document chunking strategies.

    This class selects the configured chunking strategy (e.g., 'fixed_size',
    'semantic') and uses it to break down a document, represented by its
    content and extracted structural sections, into smaller, manageable chunks.
    It also applies post-processing steps like optimizing chunk sizes.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the chunking engine.

        Args:
            config_manager: Configuration manager instance containing chunking settings.

        Raises:
            ChunkingError: If the configured strategy is unknown or fails initialization.
        """
        self.config_manager = config_manager
        try:
            # Ensure chunking config exists
            self.config = config_manager.get_section("chunking")
        except (ConfigurationError, AttributeError) as e:
            raise ChunkingError(f"Chunking configuration missing or invalid: {e}")

        self.logger = logging.getLogger(__name__)

        # Initialize available strategies using the chunking config section
        chunking_config_dict = self.config.dict()
        self.strategies: Dict[str, BaseChunkingStrategy] = {
            "fixed_size": FixedSizeChunkingStrategy(chunking_config_dict),
            "semantic": SemanticChunkingStrategy(chunking_config_dict)
            # Add other strategies here if implemented
        }

        # Set the active strategy based on configuration
        strategy_name = getattr(self.config, 'strategy', 'semantic') # Default to semantic if missing
        self.active_strategy = self.strategies.get(strategy_name)
        if not self.active_strategy:
            valid_strategies = list(self.strategies.keys())
            raise ChunkingError(
                f"Unknown chunking strategy '{strategy_name}' configured. "
                f"Valid options are: {valid_strategies}"
            )
        self.logger.info(f"Chunking engine initialized with strategy: '{strategy_name}'")

    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """
        Break a document into chunks using the active strategy.

        Args:
            document: The source Document object (used for metadata and ID).
            sections: The list of Section objects extracted from the document by
                      the DocumentProcessor.

        Returns:
            A list of Chunk objects.

        Raises:
            ChunkingError: If the active chunking strategy fails.
        """
        if not isinstance(sections, list):
             raise ChunkingError("Invalid input: 'sections' must be a list.")

        self.logger.info(
            f"Chunking document '{document.metadata.title or document.id}' "
            f"with '{self.config.strategy}' strategy using {len(sections)} sections."
        )
        if not sections:
            self.logger.warning(f"Document {document.id} has 0 sections. Returning 0 chunks.")
            return []

        try:
            # Delegate chunking to the active strategy
            chunks = self.active_strategy.chunk_document(document, sections)
            self.logger.info(f"Strategy '{self.config.strategy}' generated {len(chunks)} initial chunks.")

            # Optimize chunks (e.g., merge small ones) after initial chunking
            optimized_chunks = self.optimize_chunks(chunks)

            self.logger.info(f"Document chunking complete. Final chunk count: {len(optimized_chunks)}.")
            return optimized_chunks

        except ChunkingError as e:
             # Re-raise specific chunking errors
             self.logger.error(f"Chunking strategy '{self.config.strategy}' failed: {e}")
             raise
        except Exception as e:
            # Wrap unexpected errors from the strategy
            self.logger.critical(f"Unexpected error during chunking with strategy '{self.config.strategy}': {e}", exc_info=True)
            raise ChunkingError(f"Chunking failed due to an unexpected error in the active strategy: {str(e)}")


    def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Optimize chunks, primarily by merging adjacent small chunks.

        Checks if a chunk is below `min_chunk_size` and merges it with the
        *next* chunk if the combined size doesn't exceed a threshold based
        on `max_chunk_size`.

        Args:
            chunks: The initial list of chunks generated by the strategy.

        Returns:
            A list of optimized chunks with potentially fewer, larger chunks.
        """
        # Configurable minimum size from the chunking section
        min_chunk_size = getattr(self.config, 'min_chunk_size', 500)
        # Allow merging slightly over the max size to avoid creating another tiny chunk
        merge_max_size = getattr(self.config, 'max_chunk_size', 2500) * 1.1

        if not chunks or min_chunk_size <= 0: # If no chunks or no merging needed
            return chunks

        optimized: List[Chunk] = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]

            # Check if merge is possible with the *next* chunk
            if len(current_chunk.content) < min_chunk_size and (i + 1) < len(chunks):
                next_chunk = chunks[i+1]
                merged_size = len(current_chunk.content) + len(next_chunk.content) # Approx size

                if merged_size <= merge_max_size:
                    self.logger.debug(f"Merging small chunk {i} (size {len(current_chunk.content)}) "
                                     f"with next chunk {i+1} (size {len(next_chunk.content)}).")
                    # Merge content (simple concatenation, consider smarter joining?)
                    # Adding a newline separator seems reasonable.
                    current_chunk.content += "\n\n" + next_chunk.content

                    # Update context and headings of the current chunk based on the merged one
                    self._update_context_for_merged_chunk(current_chunk, next_chunk)

                    # Add the merged chunk to the result list
                    optimized.append(current_chunk)
                    # Skip the next chunk as it has been merged
                    i += 2
                    continue # Continue loop skipping the increment below

            # If no merge happened, add the current chunk as is
            optimized.append(current_chunk)
            i += 1

        # Update sequence numbers for the final optimized list
        for seq_num, chunk in enumerate(optimized):
            chunk.sequence = seq_num

        if len(optimized) < len(chunks):
             self.logger.info(f"Chunk optimization merged {len(chunks) - len(optimized)} small chunks. "
                             f"Result: {len(optimized)} chunks.")
        else:
             self.logger.debug("Chunk optimization did not merge any chunks.")

        return optimized


    def _update_context_for_merged_chunk(self, target_chunk: Chunk, source_chunk: Chunk) -> None:
        """
        Update context and preceding headings when merging chunks.

        The target chunk's metadata (like context.section_path) might need
        updating to reflect the combined content origins. This implementation
        focuses on merging the `preceding_headings` list intelligently.

        Args:
            target_chunk: The chunk being merged into (its content is already updated).
            source_chunk: The chunk being merged from (its content is added to target).
        """
        # Merge preceding headings: Combine unique headings, maintain relative order
        target_heading_texts = {h.content for h in target_chunk.preceding_headings}
        combined_headings = list(target_chunk.preceding_headings) # Start with target's headings

        # Add headings from the source chunk if they haven't been seen
        for heading in source_chunk.preceding_headings:
            if heading.content not in target_heading_texts:
                combined_headings.append(heading)
                target_heading_texts.add(heading.content) # Add to set to avoid duplicates

        # Sort primarily by level, then potentially by original sequence? Difficult.
        # Simple sort by level is usually sufficient for context path generation.
        combined_headings.sort(key=lambda h: h.level)

        target_chunk.preceding_headings = combined_headings

        # Optionally update context dict (e.g., merge key concepts if present)
        # target_chunk.context['merged_from_chunk_id'] = source_chunk.id # Example meta
        # Potentially re-run `get_context_for_chunk` from base strategy?
        # For now, only headings are merged. Context dict remains from the first chunk.
        # self.logger.debug(f"Updated preceding headings for merged chunk {target_chunk.id}")


    def set_strategy(self, strategy_name: str) -> None:
        """
        Change the active chunking strategy.

        Args:
            strategy_name: The name of the strategy to activate (e.g., 'semantic', 'fixed_size').

        Raises:
            ChunkingError: If the specified strategy name is not registered.
        """
        if strategy_name not in self.strategies:
            valid_strategies = list(self.strategies.keys())
            raise ChunkingError(f"Cannot set unknown chunking strategy: '{strategy_name}'. Valid options are: {valid_strategies}")

        self.active_strategy = self.strategies[strategy_name]
        self.config.strategy = strategy_name # Update config value as well? Maybe not necessary.
        self.logger.info(f"Chunking strategy actively set to: '{strategy_name}'")

    def register_strategy(self, name: str, strategy: BaseChunkingStrategy) -> None:
        """
        Register a new chunking strategy instance.

        Args:
            name: A unique name for the strategy.
            strategy: An instance of a class derived from BaseChunkingStrategy.

        Raises:
             ValueError: If attempting to register with a name that already exists.
        """
        if name in self.strategies:
             raise ValueError(f"Cannot register chunking strategy: Name '{name}' already exists.")
        if not isinstance(strategy, BaseChunkingStrategy):
             raise ValueError("Cannot register strategy: Instance must be derived from BaseChunkingStrategy.")

        self.strategies[name] = strategy
        self.logger.info(f"Registered new chunking strategy: '{name}' ({type(strategy).__name__})")

