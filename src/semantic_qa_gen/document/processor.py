
"""Document processing for SemanticQAGen."""

import os
import logging
import magic
from typing import Dict, List, Optional, Any, Type

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Section, SectionType
from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.loaders.text import TextLoader
from semantic_qa_gen.document.loaders.markdown import MarkdownLoader
from semantic_qa_gen.utils.error import DocumentError, with_error_handling


class DocumentProcessor:
    """
    Processes documents for question generation.
    
    This class is responsible for loading and preprocessing documents
    before they are chunked and analyzed.
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
        
        # Register document loaders
        self.loaders: List[BaseLoader] = []
        self._initialize_loaders()
    
    def _initialize_loaders(self) -> None:
        """Initialize document loaders based on configuration."""
        # Text loader
        if self.config.loaders.text.enabled:
            self.loaders.append(TextLoader(self.config.loaders.text.dict()))
            self.logger.debug("Registered Text loader")
            
        # PDF loader
        if self.config.loaders.pdf.enabled:
            try:
                from semantic_qa_gen.document.loaders.pdf import PDFLoader
                self.loaders.append(PDFLoader(self.config.loaders.pdf.dict()))
                self.logger.debug("Registered PDF loader")
            except ImportError as e:
                self.logger.warning(f"Failed to load PDF loader: {str(e)}")
                
        # Markdown loader
        if self.config.loaders.markdown.enabled:
            try:
                self.loaders.append(MarkdownLoader(self.config.loaders.markdown.dict()))
                self.logger.debug("Registered Markdown loader")
            except ImportError as e:
                self.logger.warning(f"Failed to load Markdown loader: {str(e)}")
                
        # DOCX loader
        if hasattr(self.config.loaders, 'docx') and self.config.loaders.docx.enabled:
            try:
                from semantic_qa_gen.document.loaders.docx import DocxLoader
                self.loaders.append(DocxLoader(self.config.loaders.docx.dict()))
                self.logger.debug("Registered DOCX loader")
            except ImportError as e:
                self.logger.warning(f"Failed to load DOCX loader: {str(e)}")
    
    @with_error_handling(error_types=Exception, max_retries=1)
    def load_document(self, path: str) -> Document:
        """
        Load a document from a file.
        
        Args:
            path: Path to the document file.
            
        Returns:
            Loaded Document object.
            
        Raises:
            DocumentError: If the document cannot be loaded.
        """
        if not os.path.exists(path):
            raise DocumentError(f"Document file not found: {path}")
        
        # Find an appropriate loader for this file
        loader = self._get_loader_for_file(path)
        if not loader:
            raise DocumentError(f"No loader available for file: {path}")
        
        self.logger.info(f"Loading document: {path}")
        document = loader.load(path)
        
        # Check if we have content
        if not document.content or len(document.content.strip()) == 0:
            raise DocumentError(f"Document is empty: {path}")
            
        # Preprocess the document
        document = self.preprocess_document(document)
        
        self.logger.info(f"Document loaded: {document.metadata.title or path}")
        return document

    def _get_loader_for_file(self, path: str) -> Optional[BaseLoader]:
        """
        Find an appropriate loader for the given file.

        Args:
            path: Path to the document file.

        Returns:
            Appropriate BaseLoader instance, or None if no loader is found.
        """
        # First try to identify by file extension
        for loader in self.loaders:
            if loader.supports_type(path):
                return loader

        # If no loader found by extension, try to identify by MIME type
        try:
            # Import magic safely
            try:
                import magic
                has_magic = True
            except ImportError:
                has_magic = False
                self.logger.debug("python-magic not available, skipping MIME type detection")

            if has_magic:
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(path)

                # Map MIME types to loaders
                mime_map = {
                    'text/plain': TextLoader,
                    'text/markdown': MarkdownLoader,
                    'application/pdf': None,  # Will be replaced with PDFLoader if available
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': None  # DocxLoader
                }

                # Import specialized loaders if needed
                if file_type == 'application/pdf':
                    try:
                        from semantic_qa_gen.document.loaders.pdf import PDFLoader
                        mime_map['application/pdf'] = PDFLoader
                    except ImportError:
                        pass

                if file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    try:
                        from semantic_qa_gen.document.loaders.docx import DocxLoader
                        mime_map['application/vnd.openxmlformats-officedocument.wordprocessingml.document'] = DocxLoader
                    except ImportError:
                        pass

                # Find loader by MIME type
                loader_class = mime_map.get(file_type)
                if loader_class:
                    for loader in self.loaders:
                        if isinstance(loader, loader_class):
                            self.logger.info(f"Selected loader for {path} based on MIME type: {file_type}")
                            return loader
        except Exception as e:
            self.logger.debug(f"Failed to determine MIME type: {str(e)}")

        # If still no loader found, try using text loader as fallback for text-based files
        try:
            # Try to read first few bytes to see if it's text
            with open(path, 'rb') as f:
                content = f.read(1024)
                try:
                    content.decode('utf-8')
                    # If we get here, it's likely a text file
                    for loader in self.loaders:
                        if isinstance(loader, TextLoader):
                            self.logger.info(f"Using TextLoader as fallback for {path}")
                            return loader
                except UnicodeDecodeError:
                    # Not a text file
                    pass
        except Exception as e:
            self.logger.debug(f"Failed to check if file is text: {str(e)}")

        return None

    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a document for better analysis.
        
        Args:
            document: Document to preprocess.
            
        Returns:
            Preprocessed document.
        """
        # Normalize whitespace
        document.content = self._normalize_whitespace(document.content)
        
        # Fix encoding issues
        document.content = self._fix_encoding_issues(document.content)
        
        # Fix bullet points and other common formatting issues
        document.content = self._fix_formatting_issues(document.content)
        
        return document
    
    def _normalize_whitespace(self, content: str) -> str:
        """
        Normalize whitespace in content.
        
        Args:
            content: Text content to normalize.
            
        Returns:
            Normalized text content.
        """
        # Replace multiple newlines with double newlines
        import re
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Replace tabs with spaces
        content = content.replace('\t', '    ')
        
        # Normalize line endings
        content = content.replace('\r\n', '\n')
        
        # Ensure content ends with newline
        if not content.endswith('\n'):
            content += '\n'
            
        return content
    
    def _fix_encoding_issues(self, content: str) -> str:
        """
        Fix common encoding issues.
        
        Args:
            content: Text content to fix.
            
        Returns:
            Fixed text content.
        """
        # Replace common problematic characters
        replacements = {
            '\ufeff': '',       # Zero-width no-break space/BOM
            '\u2019': "'",      # Right single quotation mark
            '\u2018': "'",      # Left single quotation mark
            '\u201c': '"',      # Left double quotation mark
            '\u201d': '"',      # Right double quotation mark
            '\u2013': '-',      # En dash
            '\u2014': '--',     # Em dash
            '\u00a0': ' ',      # Non-breaking space
            '\u2026': '...',    # Ellipsis
        }
        
        for char, replacement in replacements.items():
            content = content.replace(char, replacement)
            
        return content
    
    def _fix_formatting_issues(self, content: str) -> str:
        """
        Fix common formatting issues.
        
        Args:
            content: Text content to fix.
            
        Returns:
            Fixed text content.
        """
        # Standardize bullet points
        import re
        bullet_pattern = r'(?m)^[\s]*[•\*\-\+◦○●■□▪▫][\s]+'
        content = re.sub(bullet_pattern, '• ', content)
        
        return content
    
    def extract_sections(self, document: Document) -> List[Section]:
        """
        Extract sections from a document.
        
        This is used for structural analysis and semantic chunking.
        
        Args:
            document: Document to extract sections from.
            
        Returns:
            List of Section objects representing the document structure.
        """
        # If document has predefined sections, use them
        if hasattr(document, 'sections') and document.sections:
            return document.sections
            
        sections = []
        
        if document.doc_type == "markdown":
            # For markdown, use the specialized parser
            loader = next((l for l in self.loaders if isinstance(l, MarkdownLoader)), None)
            if loader:
                structure = loader.parse_document_structure(document.content)
                
                # Process each section
                for section_data in structure['sections']:
                    heading = Section(
                        content=section_data['heading'],
                        section_type=SectionType.HEADING,
                        level=section_data['level'],
                    )
                    sections.append(heading)
                    
                    # Add the content as a paragraph section
                    if section_data['content'].strip():
                        paragraph = Section(
                            content=section_data['content'],
                            section_type=SectionType.PARAGRAPH,
                            level=0,
                            metadata={'heading_level': section_data['level']}
                        )
                        sections.append(paragraph)
        else:
            # For plain text, use a simple line-based approach
            import re
            
            # Identify potential headings (lines that end with newlines and don't have punctuation at the end)
            lines = document.content.split('\n')
            current_section_type = SectionType.PARAGRAPH
            current_section_content = []
            current_level = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Empty line - if we have content, add it as a section
                    if current_section_content:
                        section_text = ' '.join(current_section_content)
                        sections.append(Section(
                            content=section_text,
                            section_type=current_section_type,
                            level=current_level
                        ))
                        current_section_content = []
                    continue
                
                # Check if this looks like a heading
                # Criteria: short line without punctuation at end, or all caps, or starts with hash
                is_heading = (len(line) < 100 and not re.search(r'[.,:;?!]$', line)) or \
                             line.isupper() or \
                             re.match(r'^#{1,6}\s', line) is not None
                
                if is_heading:
                    # If we have content, add it as a section
                    if current_section_content:
                        section_text = ' '.join(current_section_content)
                        sections.append(Section(
                            content=section_text,
                            section_type=current_section_type,
                            level=current_level
                        ))
                        current_section_content = []
                    
                    # Remove markdown heading markers if present
                    heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
                    if heading_match:
                        level = len(heading_match.group(1))
                        line = heading_match.group(2)
                    else:
                        # Determine heading level based on characteristics
                        level = 1
                        if line.isupper():
                            level = 1  # ALL CAPS titles are likely top-level
                        elif line[0].isupper():
                            level = 2  # Title Case headings are likely second-level
                            
                    sections.append(Section(
                        content=line,
                        section_type=SectionType.HEADING,
                        level=level
                    ))
                    
                    # Reset for next section
                    current_section_type = SectionType.PARAGRAPH
                    current_level = 0
                    current_section_content = []
                else:
                    # Regular paragraph content
                    current_section_content.append(line)
            
            # Add any remaining content
            if current_section_content:
                section_text = ' '.join(current_section_content)
                sections.append(Section(
                    content=section_text,
                    section_type=current_section_type,
                    level=current_level
                ))
        
        return sections
