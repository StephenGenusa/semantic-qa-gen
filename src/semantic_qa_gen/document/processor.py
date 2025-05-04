# document/processor.py

"""Document processing for SemanticQAGen."""

import os
import logging
import re # Import re for enhanced fallback heuristic
from typing import Dict, List, Optional, Any, Type

# Safe import for type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    # Define fallback type if needed elsewhere, but not strictly necessary here
    class Magic: # type: ignore
         def __init__(self, mime=True): pass
         def from_file(self, path): return None

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Section, SectionType, DocumentType # Updated imports
from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.loaders.text import TextLoader
from semantic_qa_gen.document.loaders.markdown import MarkdownLoader
# Import optional loaders safely for type checking later
try:
    from semantic_qa_gen.document.loaders.pdf import PDFLoader
except ImportError:
    PDFLoader = None # Assign None if not available
try:
    from semantic_qa_gen.document.loaders.docx import DocxLoader
except ImportError:
    DocxLoader = None # Assign None if not available

from semantic_qa_gen.utils.error import DocumentError, with_error_handling


class DocumentProcessor:
    """
    Processes documents for question generation.

    Responsible for selecting the appropriate loader based on file type,
    loading the document content and metadata, performing preprocessing steps
    (like whitespace normalization), and extracting a structured representation
    (list of Sections) for downstream processing like chunking.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the document processor.

        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.config = config_manager.get_section("document")
        self.logger = logging.getLogger(__name__)

        # Register document loaders based on configuration
        self.loaders: List[BaseLoader] = []
        self._initialize_loaders()
        if not self.loaders:
             self.logger.warning("No document loaders were successfully initialized. Document loading capability will be limited.")

    def _initialize_loaders(self) -> None:
        """Initialize document loaders based on configuration settings."""
        self.logger.debug("Initializing document loaders...")
        loader_configs = self.config.loaders

        # Text loader (always available)
        if loader_configs.text.enabled:
            self.loaders.append(TextLoader(loader_configs.text.dict()))
            self.logger.debug("Registered Text loader.")

        # PDF loader (optional dependency)
        if loader_configs.pdf.enabled:
            if PDFLoader:
                try:
                    self.loaders.append(PDFLoader(loader_configs.pdf.dict()))
                    self.logger.debug("Registered PDF loader.")
                except Exception as e:
                     # Catch potential init errors within the loader itself
                     self.logger.error(f"Error initializing PDFLoader (dependency likely installed but failed init): {e}", exc_info=True)
            else:
                self.logger.warning("PDF loader enabled in config, but 'pymupdf' dependency not found. PDF loading disabled. Install with: pip install semantic-qa-gen[pdf]")

        # Markdown loader (optional dependency)
        if loader_configs.markdown.enabled:
            # MarkdownLoader has internal check for commonmark
            try:
                self.loaders.append(MarkdownLoader(loader_configs.markdown.dict()))
                self.logger.debug("Registered Markdown loader.")
            except Exception as e:
                 self.logger.error(f"Error initializing MarkdownLoader: {e}", exc_info=True)


        # DOCX loader (optional dependency)
        if hasattr(loader_configs, 'docx') and loader_configs.docx.enabled:
            if DocxLoader:
                 try:
                    self.loaders.append(DocxLoader(loader_configs.docx.dict()))
                    self.logger.debug("Registered DOCX loader.")
                 except Exception as e:
                     self.logger.error(f"Error initializing DocxLoader (dependency likely installed but failed init): {e}", exc_info=True)
            else:
                self.logger.warning("DOCX loader enabled in config, but 'python-docx' dependency not found. DOCX loading disabled. Install with: pip install semantic-qa-gen[docx]")

        self.logger.info(f"Initialized {len(self.loaders)} document loader(s).")


    @with_error_handling(error_types=(DocumentError, IOError), max_retries=1)
    def load_document(self, path: str) -> Document:
        """
        Load and preprocess a document from a file path.

        Selects the appropriate loader, loads content and metadata,
        applies preprocessing, and attempts to extract structural sections.

        Args:
            path: Path to the document file.

        Returns:
            Loaded and preprocessed Document object.

        Raises:
            FileNotFoundError: If the path does not exist.
            DocumentError: If the path is not a file, no suitable loader is found,
                           the document is empty, or loading/preprocessing fails.
        """
        if not os.path.exists(path):
            # Use FileNotFoundError for consistency with standard Python behavior
            raise FileNotFoundError(f"Document file not found: {path}")
        if not os.path.isfile(path):
             raise DocumentError(f"Specified path is not a file: {path}")

        # Find an appropriate loader for this file type
        loader = self._get_loader_for_file(path)
        if not loader:
            # Attempt to determine type for better error message
            detected_type = "unknown"
            if MAGIC_AVAILABLE:
                try:
                     mime = magic.Magic(mime=True)
                     detected_type = mime.from_file(path) or detected_type
                except Exception:
                     pass
            raise DocumentError(f"No configured or compatible loader available for file type '{detected_type}' at: {path}")

        self.logger.info(f"Loading document using {type(loader).__name__}: {path}")
        try:
            document = loader.load(path)
        except DocumentError as e:
             self.logger.error(f"Loader {type(loader).__name__} failed for {path}: {e}")
             raise # Re-raise specific DocumentError
        except Exception as e:
             self.logger.error(f"Unexpected error during loading with {type(loader).__name__} for {path}: {e}", exc_info=True)
             raise DocumentError(f"Failed to load document '{path}' using {type(loader).__name__}: {str(e)}")


        # Validate loaded content
        if not document.content or len(document.content.strip()) == 0:
            self.logger.warning(f"Document loaded from {path} appears to be empty.")
            # Decide whether to raise error or allow empty doc processing
            raise DocumentError(f"Document content is empty after loading: {path}")

        # Preprocess the document content
        self.logger.debug(f"Preprocessing document content for: {path}")
        document = self.preprocess_document(document)

        # Attempt to extract structural sections (might be done by loader or here)
        # This populates document.sections if not already done by the loader.
        if not document.sections: # Only run if loader didn't provide sections
             self.logger.debug(f"Loader did not provide sections, attempting extraction for: {path}")
             document.sections = self.extract_sections(document)
             self.logger.info(f"Extracted {len(document.sections)} sections using processor logic for: {document.metadata.title or path}")
        else:
             self.logger.info(f"Using {len(document.sections)} sections provided by the loader for: {document.metadata.title or path}")

        return document

    def _get_loader_for_file(self, path: str) -> Optional[BaseLoader]:
        """
        Select the most appropriate loader based on file extension and MIME type.

        Args:
            path: Path to the document file.

        Returns:
            An instance of a suitable BaseLoader, or None if no match found.
        """
        self.logger.debug(f"Attempting to find loader for: {path}")
        # 1. Try matching by file extension (case-insensitive)
        _, extension = os.path.splitext(path.lower())
        for loader in self.loaders:
            # hasattr check ensures the method exists on the loader instance
            if hasattr(loader, 'supports_type') and loader.supports_type(path):
                self.logger.debug(f"Found loader by extension '{extension}': {type(loader).__name__}")
                return loader

        self.logger.debug(f"No loader found by extension '{extension}'. Trying MIME type detection.")

        # 2. Try matching by MIME type using python-magic (if available)
        if MAGIC_AVAILABLE:
            try:
                mime = magic.Magic(mime=True)
                file_mime_type = mime.from_file(path)
                self.logger.debug(f"Detected MIME type for {path}: {file_mime_type}")

                # Map MIME types to loader classes more dynamically
                mime_map = {
                    'text/plain': TextLoader,
                    'text/markdown': MarkdownLoader,
                    'application/pdf': PDFLoader,
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocxLoader,
                    # Add more mappings as needed
                }

                loader_class = mime_map.get(file_mime_type)
                if loader_class:
                    # Find the *instance* of this class in our initialized loaders
                    for loader in self.loaders:
                        if isinstance(loader, loader_class):
                            self.logger.debug(f"Found loader by MIME type '{file_mime_type}': {type(loader).__name__}")
                            return loader
            except Exception as e:
                # Log error but don't fail, proceed to fallback
                self.logger.warning(f"MIME type detection failed for {path}: {str(e)}. Proceeding without MIME check.")
        else:
             self.logger.debug("python-magic not installed, skipping MIME type detection.")


        # 3. Fallback: Check if it's likely a text file if TextLoader is available
        self.logger.debug(f"No specific loader found. Checking if file is likely text for TextLoader fallback.")
        text_loader_instance = next((ldr for ldr in self.loaders if isinstance(ldr, TextLoader)), None)
        if text_loader_instance:
            try:
                with open(path, 'rb') as f:
                    # Read a sample to check for text-like content
                    sample = f.read(1024)
                    # Basic check: If it decodes as UTF-8 without errors, treat as text
                    sample.decode('utf-8')
                    self.logger.debug(f"File {path} seems readable as text. Using TextLoader as fallback.")
                    return text_loader_instance
            except UnicodeDecodeError:
                self.logger.debug(f"File {path} could not be decoded as UTF-8 text.")
            except Exception as e:
                self.logger.warning(f"Error during text fallback check for {path}: {str(e)}")

        # 4. No suitable loader found
        self.logger.warning(f"Could not find a suitable loader for file: {path}")
        return None

    def preprocess_document(self, document: Document) -> Document:
        """
        Apply configured preprocessing steps to the document content.

        Args:
            document: The Document object to preprocess.

        Returns:
            The Document object with preprocessed content.
        """
        content = document.content
        if self.config.normalize_whitespace:
            content = self._normalize_whitespace(content)
        if self.config.fix_encoding_issues:
            content = self._fix_encoding_issues(content)
        # Add more preprocessors as needed
        content = self._fix_formatting_issues(content) # Apply basic formatting fixes

        document.content = content
        return document

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize excessive whitespace and line breaks."""
        if not content: return ""
        # Replace multiple spaces/tabs with a single space
        content = re.sub(r'[ \t]+', ' ', content)
        # Replace multiple newlines with a maximum of two (for paragraph breaks)
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove leading/trailing whitespace from each line
        content = "\n".join(line.strip() for line in content.splitlines())
        # Ensure content ends with a single newline
        return content.strip() + '\n'

    def _fix_encoding_issues(self, content: str) -> str:
        """
        Attempt to fix common Mojibake and incorrect characters.

        Uses ftfy if available for comprehensive text encoding fixes, or falls back
        to a manual replacement dictionary for common issues.
        """
        if not content: return ""

        # Try to use ftfy first if available
        try:
            import ftfy
            self.logger.debug("Using ftfy for comprehensive text encoding fixes")

            # Apply ftfy's comprehensive encoding fixes
            # By default, ftfy.fix_text handles:
            # - Mojibake (incorrectly decoded text)
            # - Unicode normalization
            # - Character width issues
            # - Line breaks and more
            fixed_content = ftfy.fix_text(content)

            return fixed_content
        except ImportError:
            self.logger.debug("ftfy library not found, using basic replacement dictionary")
        except Exception as e:
            # Handle any unexpected errors from ftfy
            self.logger.warning(f"Error using ftfy for encoding fixes: {e}. Falling back to basic replacements.")

        # Fallback to simple replacements for common issues
        replacements = {
            '\u00c2\u00a0': ' ',  # Non-breaking space encoded incorrectly
            '\u00e2\u0080\u0099': "'",  # Right single quote
            '\u00e2\u0080\u0098': "'",  # Left single quote
            '\u00e2\u0080\u009c': '"',  # Left double quote
            '\u00e2\u0080\u009d': '"',  # Right double quote
            '\u00e2\u0080\u0093': '-',  # En dash
            '\u00e2\u0080\u0094': '--',  # Em dash
            '\u00e2\u0080\u00a6': '...',  # Ellipsis
            '\ufeff': '',  # BOM
            # Add more based on observed issues
        }
        for char, replacement in replacements.items():
            content = content.replace(char, replacement)
        return content

    def _fix_formatting_issues(self, content: str) -> str:
        """Apply fixes for common text formatting inconsistencies."""
        if not content: return ""
        # Standardize bullet points (more robustly)
        # Matches lines starting with space(s) then bullet/dash/star etc, captures the content
        def replace_bullet(match):
            spaces = match.group(1) or ""
            content_after_bullet = match.group(3).strip()
            # Use a standard bullet, maintain indentation roughly
            return f"{spaces}* {content_after_bullet}"

        # Regex to find various bullet types at the start of lines, handling optional indentation
        bullet_pattern = r'(?m)^([ \t]*)(?:[•\*\-\+◦○●■□▪▫]|(?:\d+\.))([ \t]+)(.*)'
        content = re.sub(bullet_pattern, replace_bullet, content)

        # Collapse lines that likely belong to the same paragraph (unless they look like list items)
        lines = content.split('\n')
        processed_lines = []
        buffer = ""
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line: # End of potential paragraph block
                 if buffer:
                      processed_lines.append(buffer.strip())
                 processed_lines.append("") # Keep paragraph breaks
                 buffer = ""
                 continue

            # Heuristic: Does the next line start capitalized or is the current line short/ends with punctuation?
            ends_like_sentence = stripped_line.endswith(('.', '!', '?', '"', "'", ">"))
            next_line_starts_cap = (i + 1 < len(lines) and lines[i+1].strip() and lines[i+1].strip()[0].isupper())
            is_list_item = stripped_line.startswith('* ') # Assuming standardized bullets

            if ends_like_sentence or next_line_starts_cap or is_list_item:
                 # Likely end of a sentence or start of new item, treat line separately
                 if buffer:
                      processed_lines.append(buffer.strip() + " " + stripped_line)
                 else:
                      processed_lines.append(stripped_line)
                 buffer = ""
            else:
                 # Likely continuation of a paragraph, append to buffer
                 buffer = buffer.strip() + " " + stripped_line if buffer else stripped_line

        if buffer: # Add any remaining buffer content
             processed_lines.append(buffer.strip())

        content = "\n".join(processed_lines)

        return content


    def extract_sections(self, document: Document) -> List[Section]:
        """
        Extract or infer structural sections from a document's content.

        This method first checks if the loader already populated `document.sections`.
        If not (e.g., for plain text), it applies a heuristic-based approach
        to identify headings and paragraphs.

        Args:
            document: The Document object (potentially with pre-populated sections).

        Returns:
            A list of Section objects representing the document structure.
        """
        # --- Priority 1: Use sections provided by the loader ---
        # Loaders like DocxLoader or potentially PDFLoader should populate this.
        if document.sections is not None and len(document.sections) > 0:
            self.logger.debug(f"Using {len(document.sections)} sections provided by the document loader.")
            return document.sections

        # --- Priority 2: Use Markdown parser if applicable ---
        # MarkdownLoader might provide structure via its own methods,
        # but if extract_sections is called on a markdown doc without pre-parsed sections.
        if document.doc_type == DocumentType.MARKDOWN:
            self.logger.debug("Attempting to parse sections from Markdown content.")
            # Find the MarkdownLoader instance to use its parser
            loader_instance = next((ldr for ldr in self.loaders if isinstance(ldr, MarkdownLoader)), None)
            if loader_instance and hasattr(loader_instance, 'parse_document_structure'):
                 try:
                    structure = loader_instance.parse_document_structure(document.content)
                    markdown_sections = []
                    for section_data in structure.get('sections', []):
                         heading_text = section_data.get('heading', '').strip()
                         content_text = section_data.get('content', '').strip()
                         level = section_data.get('level', 0)

                         if heading_text:
                              markdown_sections.append(Section(
                                   content=heading_text,
                                   section_type=SectionType.HEADING,
                                   level=level if level > 0 else 1 # Ensure level is at least 1 for headings
                              ))
                         if content_text:
                              # Add content as paragraphs (or potentially lists/tables if parser was more complex)
                              # Simple approach: treat all as paragraph for now
                              markdown_sections.append(Section(
                                   content=content_text,
                                   section_type=SectionType.PARAGRAPH,
                                   level=0,
                                   metadata={'heading_level': level}
                              ))
                    if markdown_sections:
                         return markdown_sections
                    else:
                         self.logger.warning("Markdown parser returned no structured sections.")
                 except Exception as e:
                      self.logger.warning(f"Error parsing Markdown structure: {e}. Falling back to heuristic.")
            else:
                 self.logger.warning("Markdown document type detected, but MarkdownLoader or its parser unavailable. Falling back to heuristic.")


        # --- Priority 3: Fallback Heuristic for Plain Text (or others) ---
        self.logger.debug("Applying fallback heuristic to extract sections from plain text content.")
        sections: List[Section] = []
        current_paragraph_lines: List[str] = []

        # Regex patterns for headings (adapt as needed)
        patterns = [
            (re.compile(r'^#{1,6}\s+(.*)'), SectionType.HEADING), # Markdown Headers
            (re.compile(r'^(Chapter|Section|Part|Appendix)\s+([IVXLCDM\d\.\:]+)([:.\s].*)?$', re.IGNORECASE), SectionType.HEADING), # Chapter/Section Headers
            (re.compile(r'^([A-Z][A-Za-z\s]*)$'), SectionType.HEADING), # Simple Title Case Heuristic (potentially noisy)
            (re.compile(r'^([A-Z\s\d]+)$'), SectionType.HEADING), # ALL CAPS Heuristic
        ]
        # Max length for heuristic title/heading detection
        MAX_HEADING_LEN = 150

        def finalize_paragraph():
            nonlocal current_paragraph_lines
            if current_paragraph_lines:
                # Join lines, attempting to fix unnecessary line breaks within paragraphs
                para_content = " ".join(current_paragraph_lines).strip()
                # Additional check: split buffer by double newlines if they survived preprocessing
                paras = re.split(r'\n{2,}', para_content)
                for p in paras:
                    p_strip = p.strip()
                    if p_strip:
                        sections.append(Section(content=p_strip, section_type=SectionType.PARAGRAPH, level=0))
                current_paragraph_lines = []

        lines = document.content.splitlines()
        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if not stripped_line:
                finalize_paragraph()
                continue

            matched_heading = False
            potential_heading_level = 0

            # 1. Check explicit patterns
            for pattern, sec_type in patterns:
                match = pattern.match(stripped_line)
                if match:
                    finalize_paragraph() # Finalize previous paragraph before adding heading
                    matched_heading = True
                    heading_content = stripped_line # Default to full line
                    if pattern.pattern.startswith('^#{1,6}'): # Markdown
                        potential_heading_level = stripped_line.find(' ')
                        heading_content = match.group(1).strip()
                    elif pattern.pattern.startswith('^(Chapter|Section'): # Chapter/Section
                         potential_heading_level = 1 if 'Chapter' in match.group(1) else 2
                         # Try to extract just the core title part if possible
                         heading_parts = [p for p in match.groups() if p is not None]
                         heading_content = " ".join(heading_parts).strip()
                    elif pattern.pattern.startswith('^([A-Z\\s\d]+)$'): # ALL CAPS
                         potential_heading_level = 1
                    elif pattern.pattern.startswith('^([A-Z][A-Za-z\\s]*)$'): # Title Case
                          # Consider Title Case a heading only if short and doesn't end like a sentence
                          if len(stripped_line) < MAX_HEADING_LEN and not stripped_line.endswith(('.', '!', '?')):
                               potential_heading_level = 2
                          else:
                               matched_heading = False # Revert match if it looks like a sentence

                    if matched_heading:
                         sections.append(Section(
                              content=heading_content,
                              section_type=sec_type,
                              level=max(1, potential_heading_level) # Ensure level >= 1 for headings
                         ))
                         break # Stop checking patterns once matched

            # 2. If not matched by patterns, treat as paragraph
            if not matched_heading:
                 current_paragraph_lines.append(stripped_line)


        # Finalize any remaining paragraph content after the loop
        finalize_paragraph()

        if not sections and document.content.strip():
             # If absolutely no structure found, treat the whole content as one paragraph
              self.logger.warning("Heuristic section extraction found no structure; treating entire document as one paragraph.")
              sections.append(Section(content=document.content.strip(), section_type=SectionType.PARAGRAPH, level=0))

        return sections

