"""PDF file loader for SemanticQAGen."""

import os
import re
import fitz  # PyMuPDF
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple, Set

from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata, Section, SectionType
from semantic_qa_gen.utils.error import DocumentError, with_error_handling


class PDFLoader(BaseLoader):
    """
    Loader for PDF files with advanced text extraction.
    
    This loader handles extracting text from PDF files while preserving:
    - Document metadata
    - Page numbers for each text block
    - Title detection based on font size
    - Cross-page sentence handling
    - Reading order reconstruction
    - Automatic header/footer detection and removal
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF loader.
        
        Args:
            config: Configuration dictionary for the loader.
        """
        super().__init__(config)
        self.extract_images = self.config.get('extract_images', False)
        self.ocr_enabled = self.config.get('ocr_enabled', False)
        self.min_heading_ratio = self.config.get('min_heading_ratio', 1.2)
        self.header_footer_threshold = self.config.get('header_footer_threshold', 0.75)
        self.detect_headers_footers = self.config.get('detect_headers_footers', True)
        self.fix_cross_page_sentences = self.config.get('fix_cross_page_sentences', True)
        self.preserve_page_numbers = self.config.get('preserve_page_numbers', True)
    
    @with_error_handling(error_types=Exception, max_retries=1)
    def load(self, path: str) -> Document:
        """
        Load a document from a PDF file.
        
        Args:
            path: Path to the PDF file.
            
        Returns:
            Loaded Document object.
            
        Raises:
            DocumentError: If the PDF file cannot be loaded.
        """
        if not self.supports_type(path):
            raise DocumentError(f"Unsupported file type for PDFLoader: {path}")
        
        try:
            # Open the PDF document
            pdf_document = fitz.open(path)
            
            # Extract metadata
            metadata = self._extract_metadata(pdf_document, path)
            
            # Extract content with structure preservation
            content, sections = self._extract_content_with_structure(pdf_document)
            
            # Create the document object
            document = Document(
                content=content,
                doc_type=DocumentType.PDF,
                path=path,
                metadata=metadata
            )
            
            # Store sections as a custom attribute for later use in chunking
            document.sections = sections
            
            return document
            
        except Exception as e:
            raise DocumentError(f"Failed to load PDF file {path}: {str(e)}")
        finally:
            if 'pdf_document' in locals():
                pdf_document.close()
    
    def supports_type(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file type.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this is a PDF file, False otherwise.
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext == '.pdf'
    
    def _extract_metadata(self, pdf_document, path: str) -> DocumentMetadata:
        """
        Extract metadata from a PDF document.
        
        Args:
            pdf_document: Open PDF document.
            path: Path to the PDF file.
            
        Returns:
            DocumentMetadata object.
        """
        # Extract built-in PDF metadata
        pdf_meta = pdf_document.metadata
        
        # Get the title, falling back to filename if not available
        title = pdf_meta.get('title')
        if not title or title.strip() == "":
            # Fallback to filename
            file_name = os.path.basename(path)
            title, _ = os.path.splitext(file_name)
            title = title.replace('_', ' ').replace('-', ' ').strip()
            
            # Try to properly capitalize title
            if title.isupper() or title.islower():
                title = ' '.join(word.capitalize() for word in title.split())
        
        # Format creation date if present
        creation_date = pdf_meta.get('creationDate')
        if creation_date and creation_date.startswith('D:'):
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            try:
                date_str = creation_date[2:14]  # Extract YYYYMMDDHHMM
                formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
                creation_date = formatted_date
            except:
                # If parsing fails, use the original value
                pass
        
        metadata = DocumentMetadata(
            title=title,
            author=pdf_meta.get('author'),
            date=creation_date,
            source=path,
            language=pdf_meta.get('language'),
            custom={
                'page_count': len(pdf_document),
                'producer': pdf_meta.get('producer'),
                'creator': pdf_meta.get('creator'),
                'keywords': pdf_meta.get('keywords'),
            }
        )
        
        return metadata
    
    def _extract_content_with_structure(self, pdf_document) -> Tuple[str, List[Section]]:
        """
        Extract content from PDF with structure preservation.
        
        Args:
            pdf_document: Open PDF document.
            
        Returns:
            Tuple of (full_text, sections_list)
        """
        # First, detect headers and footers if enabled
        headers, footers = None, None
        if self.detect_headers_footers:
            headers, footers = self._detect_headers_and_footers(pdf_document)
        
        # Extract text by page with structure information
        page_texts = []
        page_sections = []
        all_sections = []
        
        # First pass: extract text and identify headings by page
        for page_num, page in enumerate(pdf_document):
            page_content, sections = self._extract_page_content(
                page, 
                page_num,
                headers, 
                footers
            )
            
            if page_content:
                page_texts.append(page_content)
                page_sections.append(sections)
                all_sections.extend(sections)
        
        # Second pass: handle cross-page sentences if enabled
        processed_text = page_texts
        if self.fix_cross_page_sentences:
            processed_text = self._handle_cross_page_sentences(page_texts)
        
        # Combine all text
        full_text = "\n\n".join(processed_text)
        
        return full_text, all_sections
    
    def _extract_page_content(self, page, page_num: int, 
                             headers: Optional[List[str]], 
                             footers: Optional[List[str]]) -> Tuple[str, List[Section]]:
        """
        Extract content from a single page with structural information.
        
        Args:
            page: PDF page object.
            page_num: Page number (0-indexed).
            headers: List of detected header patterns to skip.
            footers: List of detected footer patterns to skip.
            
        Returns:
            Tuple of (page_text, page_sections)
        """
        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Extract text blocks with position and font information
        blocks_dict = page.get_text("dict")
        blocks = blocks_dict.get("blocks", [])
        
        # Find the most common font size on this page (body text)
        font_sizes = []
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span.get("size", 0))
        
        body_font_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 10
        
        # Process blocks in reading order (top to bottom)
        page_content = []
        page_sections = []
        
        # Sort blocks by vertical position for reading order
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        
        for block in sorted_blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue
            
            block_text = ""
            max_font_size = 0
            font_name = None
            is_bold = False
            
            # Extract text from spans while tracking formatting
            for line in block["lines"]:
                line_text = ""
                
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    if not span_text:
                        continue
                    
                    span_font = span.get("font", "")
                    is_bold = is_bold or "bold" in span_font.lower()
                    
                    line_text += span_text + " "
                    max_font_size = max(max_font_size, span["size"])
                    font_name = span.get("font")
                
                if line_text:
                    block_text += line_text.strip() + "\n"
            
            block_text = block_text.strip()
            if not block_text:
                continue
            
            # Check if this block is a header/footer to skip
            if headers and any(self._similar_text(block_text, h) for h in headers):
                continue
            if footers and any(self._similar_text(block_text, f) for f in footers):
                continue
            
            # Determine if this is a heading based on font size and formatting
            is_heading = max_font_size > body_font_size * self.min_heading_ratio or is_bold
            
            if is_heading:
                # Determine heading level based on size difference and position
                heading_level = 1  # Default to top level
                
                if max_font_size < body_font_size * 1.5:
                    heading_level = 2
                elif max_font_size < body_font_size * 1.3:
                    heading_level = 3
                
                # Special case: If at top of page and significantly larger, likely a title
                is_at_top = block["bbox"][1] < page_height * 0.2
                if is_at_top and max_font_size > body_font_size * 1.5:
                    section_type = SectionType.TITLE
                    heading_level = 1
                else:
                    section_type = SectionType.HEADING
                
                section = Section(
                    content=block_text,
                    section_type=section_type,
                    level=heading_level,
                    metadata={
                        'page': page_num + 1,
                        'font_size': max_font_size,
                        'font_name': font_name,
                        'is_bold': is_bold,
                        'position': {
                            'x': block["bbox"][0],
                            'y': block["bbox"][1]
                        }
                    }
                )
                
                page_sections.append(section)
                
                # Add the heading to the page content with page number if enabled
                if self.preserve_page_numbers:
                    page_content.append(f"{block_text} [Page {page_num + 1}]")
                else:
                    page_content.append(block_text)
                
            else:
                # Regular paragraph
                section = Section(
                    content=block_text,
                    section_type=SectionType.PARAGRAPH,
                    level=0,
                    metadata={
                        'page': page_num + 1,
                        'font_size': max_font_size,
                        'font_name': font_name,
                        'position': {
                            'x': block["bbox"][0],
                            'y': block["bbox"][1]
                        }
                    }
                )
                
                page_sections.append(section)
                page_content.append(block_text)
        
        # Join the page content into a single string
        return "\n".join(page_content), page_sections
    
    def _detect_headers_and_footers(self, pdf_document) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Detect repeated headers and footers in the PDF.
        
        Args:
            pdf_document: Open PDF document.
            
        Returns:
            Tuple of (headers, footers) lists.
        """
        page_count = len(pdf_document)
        if page_count < 3:  # Need multiple pages to detect patterns
            return None, None
        
        # Extract top and bottom text blocks from each page
        top_blocks = []
        bottom_blocks = []
        
        for page_num, page in enumerate(pdf_document):
            blocks_dict = page.get_text("dict")
            blocks = blocks_dict.get("blocks", [])
            if not blocks:
                continue
                
            # Sort blocks by y-position
            sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
            
            if sorted_blocks:
                # Top block
                top_block = sorted_blocks[0]
                top_text = self._extract_block_text(top_block)
                if top_text and len(top_text) < 200:  # Reasonable header size
                    top_blocks.append((page_num, top_text))
                
                # Bottom block
                bottom_block = sorted_blocks[-1]
                bottom_text = self._extract_block_text(bottom_block)
                if bottom_text and len(bottom_text) < 200:  # Reasonable footer size
                    bottom_blocks.append((page_num, bottom_text))
        
        # Find repeated patterns
        header_candidates = self._find_repeated_text(top_blocks)
        footer_candidates = self._find_repeated_text(bottom_blocks)
        
        return header_candidates, footer_candidates
    
    def _extract_block_text(self, block) -> str:
        """
        Extract text from a block.
        
        Args:
            block: PDF text block.
            
        Returns:
            Extracted text string.
        """
        if block.get("type") != 0:  # Not text block
            return ""
            
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "") + " "
        return text.strip()
    
    def _find_repeated_text(self, text_blocks, min_repetitions=3) -> List[str]:
        """
        Find text that appears multiple times across pages.
        
        Args:
            text_blocks: List of (page_num, text) tuples.
            min_repetitions: Minimum number of repetitions required.
            
        Returns:
            List of repeated text patterns.
        """
        if len(text_blocks) < min_repetitions:
            return []
            
        # Group by similar text
        groups = {}
        
        for page_num, text in text_blocks:
            added = False
            for key in groups:
                if SequenceMatcher(None, text, key).ratio() > self.header_footer_threshold:
                    groups[key].append((page_num, text))
                    added = True
                    break
            
            if not added:
                groups[text] = [(page_num, text)]
        
        # Return text that appears frequently
        repeated = []
        for key, items in groups.items():
            if len(items) >= min_repetitions:
                repeated.append(key)
                
        return repeated
    
    def _handle_cross_page_sentences(self, pages_text: List[str]) -> List[str]:
        """
        Detect and fix sentences broken across page boundaries.
        
        Args:
            pages_text: List of text content for each page.
            
        Returns:
            List of processed text with fixed sentences.
        """
        if not pages_text:
            return []
            
        result = []
        pending_text = ""
        
        for i, page_text in enumerate(pages_text):
            if not page_text.strip():
                continue
                
            # If there's pending text from the previous page
            if pending_text:
                # Check if this page starts with lowercase or punctuation that would
                # indicate continuation of a previous sentence
                first_char = page_text.lstrip()[:1]
                if first_char and (first_char.islower() or first_char in ',;:)]}>'):
                    # This page likely continues the previous sentence
                    page_text = pending_text + " " + page_text.lstrip()
                else:
                    # Doesn't seem to be a continuation, add pending text separately
                    result.append(pending_text)
                
                pending_text = ""
            
            # Check if this page ends with an incomplete sentence
            last_sentence_end = max(
                page_text.rfind('.'), page_text.rfind('!'),
                page_text.rfind('?'), page_text.rfind('."'),
                page_text.rfind('!"'), page_text.rfind('?"')
            )
            
            # If no proper sentence ending is found near the end of the page
            # or if the page ends with a hyphen (indicating a broken word)
            if (last_sentence_end == -1 or last_sentence_end < len(page_text) - 20) or \
               page_text.rstrip().endswith('-'):
                # This might be a sentence cut by page break
                pending_text = page_text
            else:
                # There's a proper sentence end
                result.append(page_text)
        
        # Add any remaining text
        if pending_text:
            result.append(pending_text)
            
        return result

    def _similar_text(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar using sequence matcher.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            True if texts are similar, False otherwise.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio() > self.header_footer_threshold
