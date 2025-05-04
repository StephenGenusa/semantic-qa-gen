# document/loaders/markdown.py

"""Markdown file loader for SemanticQAGen."""

import os
import re
import logging
from typing import Dict, Any, Optional, List, Tuple # Added Tuple

# Safe import for commonmark
try:
    import commonmark
    COMMONMARK_AVAILABLE = True
except ImportError:
    COMMONMARK_AVAILABLE = False

from semantic_qa_gen.document.loaders.base import BaseLoader
# Import Section and SectionType for direct use
from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata, Section, SectionType
from semantic_qa_gen.utils.error import DocumentError


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files (.md, .markdown).

    Extracts text content, attempts to parse YAML front matter for metadata,
    and uses 'commonmark' (if available) or regex fallbacks to extract
    structural sections (headings, paragraphs).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Markdown loader.

        Args:
            config: Configuration dictionary for the loader. Relevant keys:
                    'encoding' (str, default 'utf-8'),
                    'extract_metadata' (bool, default True).
        """
        super().__init__(config)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.logger = logging.getLogger(__name__)

        if not COMMONMARK_AVAILABLE:
            self.logger.warning(
                "The 'commonmark' package is not installed. Markdown parsing will use "
                "a less accurate regex-based fallback. Install with: pip install commonmark"
            )

    def load(self, path: str) -> Document:
        """
        Load content and structure from a Markdown file.

        Args:
            path: Path to the Markdown file.

        Returns:
            A Document object containing the content, metadata, and extracted sections.

        Raises:
            DocumentError: If the file type is unsupported, or if there are issues
                           reading, decoding, or parsing the file.
        """
        if not self.supports_type(path):
            raise DocumentError(f"Unsupported file type for MarkdownLoader: {path}")

        self.logger.info(f"Loading Markdown document: {path}")
        try:
            with open(path, 'r', encoding=self.encoding) as file:
                full_content = file.read()

            # --- Metadata Extraction ---
            metadata = DocumentMetadata(source=path) # Initialize with source path
            raw_content = full_content # Keep original content before stripping front matter

            if self.extract_metadata:
                front_matter_meta = self._extract_front_matter(full_content)
                raw_content = self._strip_front_matter(full_content) # Use content after strip for structure parsing
                # Merge extracted meta, prioritizing front matter but keeping source
                metadata.title = front_matter_meta.title
                metadata.author = front_matter_meta.author
                metadata.date = front_matter_meta.date
                metadata.language = front_matter_meta.language
                metadata.custom = front_matter_meta.custom

            # Fallback title from filename if needed
            if not metadata.title:
                file_name = os.path.basename(path)
                metadata.title = os.path.splitext(file_name)[0]
            self.logger.debug(f"Extracted metadata for {path}: Title='{metadata.title}'")

            # --- Structure (Sections) Extraction ---
            # Use the content *without* front matter for section parsing
            sections = self._parse_structure_to_sections(raw_content)
            self.logger.debug(f"Extracted {len(sections)} sections from {path}.")

            # --- Create Document Object ---
            # The main 'content' can be the raw text or structured text, depending on needs.
            # Let's use the raw_content (without front matter) as the primary content view.
            # The structured sections list is stored separately in document.sections.
            document = Document(
                content=raw_content.strip(), # Use content without front matter
                doc_type=DocumentType.MARKDOWN,
                path=path,
                metadata=metadata,
                sections=sections # Assign the extracted sections directly
            )

            return document

        except UnicodeDecodeError:
            self.logger.error(f"Encoding error reading {path} with {self.encoding}.")
            raise DocumentError(
                f"Failed to decode Markdown file with encoding {self.encoding}: {path}. Try specifying a different encoding in the config."
            )
        except FileNotFoundError:
             # Should be caught by DocumentProcessor, but defensive check
              raise DocumentError(f"Markdown file not found: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load or parse Markdown file {path}: {e}", exc_info=True)
            raise DocumentError(f"Failed to process Markdown file: {str(e)}")

    def supports_type(self, file_path: str) -> bool:
        """Check if the file extension indicates a Markdown file."""
        _, ext = os.path.splitext(file_path.lower())
        return ext in ['.md', '.markdown']

    def _extract_front_matter(self, content: str) -> DocumentMetadata:
        """Extract metadata from YAML front matter."""
        metadata = DocumentMetadata()
        # Regex looking for --- or +++ delimiters at the start of the file
        front_matter_match = re.match(r'^(?:---|\+\+\+)\s*?\n(.*?)\n(?:---|\+\+\+)\s*?\n', content, re.DOTALL)
        if not front_matter_match:
            return metadata

        front_matter_text = front_matter_match.group(1)
        try:
            # Use PyYAML to safely parse the front matter
            import yaml # Local import as PyYAML is a core dep
            front_matter_data = yaml.safe_load(front_matter_text)
            if isinstance(front_matter_data, dict):
                 metadata.title = str(front_matter_data.get('title', '')) or None
                 metadata.author = str(front_matter_data.get('author', '')) or None
                 metadata.date = str(front_matter_data.get('date', '')) or None
                 metadata.language = str(front_matter_data.get('language', '')) or None
                 # Store all other keys in custom metadata
                 metadata.custom = {
                     k: v for k, v in front_matter_data.items()
                     if k not in ['title', 'author', 'date', 'language']
                 }
        except yaml.YAMLError as e:
            self.logger.warning(f"Could not parse YAML front matter: {e}. Falling back to simple key-value parsing.")
            # Fallback to simple line splitting if YAML parsing fails
            for line in front_matter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip().strip("'\"") # Basic cleaning
                    if key == 'title': metadata.title = value
                    elif key == 'author': metadata.author = value
                    elif key == 'date': metadata.date = value
                    elif key == 'language': metadata.language = value
                    else: metadata.custom[key] = value
        except ImportError:
             self.logger.error("PyYAML not found, cannot parse front matter effectively.")
             # Could implement the very basic line splitter here again if needed without PyYAML

        return metadata

    def _strip_front_matter(self, content: str) -> str:
        """Remove YAML front matter from the start of the content."""
        # Uses the same regex as extraction
        return re.sub(r'^(?:---|\+\+\+)\s*?\n(.*?)\n(?:---|\+\+\+)\s*?\n?', '', content, flags=re.DOTALL)

    def _parse_structure_to_sections(self, content: str) -> List[Section]:
        """
        Parse Markdown content into a list of Section objects.

        Uses 'commonmark' if available for accurate parsing, otherwise falls
        back to a regex-based heuristic.

        Args:
            content: The Markdown content (preferably with front matter removed).

        Returns:
            A list of Section objects representing headings, paragraphs, etc.
        """
        if COMMONMARK_AVAILABLE:
            try:
                return self._parse_with_commonmark(content)
            except Exception as e:
                self.logger.warning(f"commonmark parsing failed: {e}. Falling back to regex.", exc_info=True)
                # Fall through to regex fallback if commonmark fails unexpectedly
        # Fallback if commonmark not available or failed
        return self._parse_with_regex(content)

    def _parse_with_commonmark(self, content: str) -> List[Section]:
        """Parse Markdown using the commonmark library AST."""
        sections = []
        parser = commonmark.Parser()
        ast = parser.parse(content)

        current_section_parts = []
        current_heading_level = 0 # Track level for associating paragraphs

        def finalize_paragraph():
             nonlocal current_section_parts, current_heading_level
             if current_section_parts:
                  para_content = "".join(current_section_parts).strip()
                  if para_content:
                       sections.append(Section(
                            content=para_content,
                            section_type=SectionType.PARAGRAPH,
                            level=0,
                            metadata={'heading_level': current_heading_level}
                       ))
                  current_section_parts = []

        for node, entering in ast.walker():
            node_type = node.t

            if entering:
                if node_type == 'heading':
                    finalize_paragraph() # Finalize previous paragraph before starting heading
                    heading_text = "".join(c.literal for c in node.walker() if c[0].t == 'text').strip()
                    current_heading_level = node.level
                    if heading_text:
                         sections.append(Section(
                              content=heading_text,
                              section_type=SectionType.HEADING,
                              level=current_heading_level
                         ))
                elif node_type == 'paragraph':
                    finalize_paragraph() # Start of a new paragraph block
                elif node_type == 'text':
                    if node.literal:
                         current_section_parts.append(node.literal)
                elif node_type == 'softbreak':
                     # Treat soft breaks as spaces within a paragraph
                     if current_section_parts and not current_section_parts[-1].endswith(' '):
                          current_section_parts.append(' ')
                elif node_type == 'linebreak':
                     # Treat hard breaks as newlines within a paragraph (or potentially list item)
                     current_section_parts.append('\n')
                elif node_type == 'code_block':
                      finalize_paragraph()
                      if node.literal:
                           sections.append(Section(
                                content=node.literal.strip('\n'),
                                section_type=SectionType.CODE_BLOCK,
                                level=0,
                                metadata={'language': node.info or None, 'heading_level': current_heading_level}
                           ))
                elif node_type == 'list':
                      finalize_paragraph() # Lists break paragraphs
                      # List parsing can be complex; basic approach: treat items as paragraphs for now
                      # A more sophisticated approach would create SectionType.LIST_ITEM
                      pass # Handled by list_item processing below
                elif node_type == 'item': # List item
                      # Let text nodes inside the item be collected
                      pass
                elif node_type == 'block_quote':
                      finalize_paragraph()
                      # Could potentially create QUOTE sections if needed
                      pass # Treat contained paragraphs normally for now

            else: # Exiting node
                 if node_type == 'paragraph':
                      finalize_paragraph() # Finalize paragraph content when leaving paragraph node
                 elif node_type == 'list':
                      finalize_paragraph() # Finalize list item content if buffer has anything
                 elif node_type == 'block_quote':
                      finalize_paragraph() # Finalize quote content


        finalize_paragraph() # Add any trailing paragraph content
        return sections


    def _parse_with_regex(self, content: str) -> List[Section]:
        """Fallback parser using regex for headings and paragraphs."""
        sections: List[Section] = []
        current_paragraph_lines: List[str] = []

        # Simpler heading regex focused on line start
        heading_pattern = re.compile(r'^(#{1,6})\s+(.*)')

        def finalize_paragraph_regex():
            nonlocal current_paragraph_lines
            if current_paragraph_lines:
                para_content = "\n".join(current_paragraph_lines).strip()
                if para_content:
                    sections.append(Section(content=para_content, section_type=SectionType.PARAGRAPH, level=0))
                current_paragraph_lines = []

        for line in content.splitlines():
            heading_match = heading_pattern.match(line)
            if heading_match:
                finalize_paragraph_regex() # Finalize previous paragraph first
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                if heading_text:
                    sections.append(Section(
                        content=heading_text,
                        section_type=SectionType.HEADING,
                        level=level
                    ))
            elif line.strip(): # Non-empty line, not a heading
                current_paragraph_lines.append(line)
            else: # Empty line signifies paragraph break
                finalize_paragraph_regex()

        finalize_paragraph_regex() # Add any trailing paragraph
        return sections

