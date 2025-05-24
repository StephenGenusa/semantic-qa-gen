"""PDF file loader for SemanticQAGen with advanced reading order and OCR support."""

import os
import re
import numpy as np
import fitz  # PyMuPDF
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple, Set

# Check if tesseract/pytesseract are available
try:
    import pytesseract
    from PIL import Image
    import io

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

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
    - Reading order reconstruction (simple or advanced algorithm)
    - Automatic header/footer detection and removal
    - OCR support for scanned documents
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
        self.ocr_language = self.config.get('ocr_language', 'eng')
        self.ocr_dpi = self.config.get('ocr_dpi', 300)
        self.ocr_timeout = self.config.get('ocr_timeout', 30)
        self.min_heading_ratio = self.config.get('min_heading_ratio', 1.2)
        self.header_footer_threshold = self.config.get('header_footer_threshold', 0.75)
        self.detect_headers_footers = self.config.get('detect_headers_footers', True)
        self.fix_cross_page_sentences = self.config.get('fix_cross_page_sentences', True)
        self.preserve_page_numbers = self.config.get('preserve_page_numbers', True)
        self.use_advanced_reading_order = self.config.get('use_advanced_reading_order', False)

        # Check if OCR is enabled but dependencies are missing
        if self.ocr_enabled and not TESSERACT_AVAILABLE:
            self.logger.warning(
                "OCR is enabled in configuration, but pytesseract or Pillow is not installed. "
                "OCR functionality will be disabled. Install with: pip install pytesseract pillow"
            )
            self.ocr_enabled = False

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

        # Check if any pages need OCR
        pages_needing_ocr = []
        if self.ocr_enabled:
            pages_needing_ocr = self._identify_pages_needing_ocr(pdf_document)
            if pages_needing_ocr:
                self.logger.info(f"Identified {len(pages_needing_ocr)} pages that may need OCR processing")

        # First pass: extract text and identify headings by page
        for page_num, page in enumerate(pdf_document):
            # Check if this page needs OCR
            needs_ocr = page_num in pages_needing_ocr

            # Extract content with either simple or advanced reading order algorithm
            page_content, sections = self._extract_page_content(
                page,
                page_num,
                headers,
                footers,
                needs_ocr
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
                              footers: Optional[List[str]],
                              needs_ocr: bool = False) -> Tuple[str, List[Section]]:
        """
        Extract content from a single page, selecting the appropriate algorithm.

        Args:
            page: PDF page object.
            page_num: Page number (0-indexed).
            headers: List of detected header patterns to skip.
            footers: List of detected footer patterns to skip.
            needs_ocr: Whether this page needs OCR processing.

        Returns:
            Tuple of (page_text, page_sections)
        """
        # If OCR is needed and enabled, process with OCR
        if needs_ocr and self.ocr_enabled and TESSERACT_AVAILABLE:
            # Process page with OCR
            return self._extract_page_content_with_ocr(page, page_num, headers, footers)

        # Otherwise use the configured reading order algorithm
        if self.use_advanced_reading_order:
            return self._extract_page_content_advanced(page, page_num, headers, footers)
        else:
            return self._extract_page_content_simple(page, page_num, headers, footers)

    def _extract_page_content_simple(self, page, page_num: int,
                                     headers: Optional[List[str]],
                                     footers: Optional[List[str]]) -> Tuple[str, List[Section]]:
        """
        Extract content from a single page using the simple vertical sorting algorithm.

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

    def _extract_page_content_advanced(self, page, page_num: int,
                                       headers: Optional[List[str]],
                                       footers: Optional[List[str]]) -> Tuple[str, List[Section]]:
        """
        Extract content from a single page using advanced reading order algorithm.

        This implementation detects columns and follows natural reading order.

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

        # Filter out header/footer blocks
        text_blocks = []
        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks
                continue

            block_text = self._extract_block_text(block)
            if not block_text:
                continue

            # Check if this block is a header/footer to skip
            if headers and any(self._similar_text(block_text, h) for h in headers):
                continue
            if footers and any(self._similar_text(block_text, f) for f in footers):
                continue

            # Store the block with its extracted text
            block['extracted_text'] = block_text
            text_blocks.append(block)

        # Detect columns using X-coordinates
        columns = self._detect_columns(text_blocks, page_width)

        # Process blocks in proper reading order
        processed_blocks = self._order_blocks_by_reading_flow(text_blocks, columns)

        # Now create sections and extract content
        page_content = []
        page_sections = []

        for block in processed_blocks:
            block_text = block['extracted_text']

            # Extract formatting information
            max_font_size = 0
            font_name = None
            is_bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    max_font_size = max(max_font_size, span["size"])
                    font_name = span.get("font")
                    is_bold = is_bold or "bold" in (span.get("font", "").lower())

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
                            'y': block["bbox"][1],
                            'width': block["bbox"][2] - block["bbox"][0],
                            'height': block["bbox"][3] - block["bbox"][1],
                            'page_width': page_width,
                            'page_height': page_height,
                            'normalized_x': block["bbox"][0] / page_width,
                            'normalized_y': block["bbox"][1] / page_height
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
                            'y': block["bbox"][1],
                            'width': block["bbox"][2] - block["bbox"][0],
                            'height': block["bbox"][3] - block["bbox"][1],
                            'page_width': page_width,
                            'page_height': page_height,
                            'normalized_x': block["bbox"][0] / page_width,
                            'normalized_y': block["bbox"][1] / page_height
                        }
                    }
                )

                page_sections.append(section)
                page_content.append(block_text)

        # Join the page content into a single string
        return "\n".join(page_content), page_sections

    def _extract_page_content_with_ocr(self, page, page_num: int,
                                       headers: Optional[List[str]],
                                       footers: Optional[List[str]]) -> Tuple[str, List[Section]]:
        """
        Extract content from a page using OCR for scanned documents.

        Args:
            page: PDF page object.
            page_num: Page number (0-indexed).
            headers: List of detected header patterns to skip.
            footers: List of detected footer patterns to skip.

        Returns:
            Tuple of (page_text, page_sections)
        """
        if not TESSERACT_AVAILABLE:
            self.logger.warning(
                "OCR requested but pytesseract is not available. Falling back to native text extraction.")
            if self.use_advanced_reading_order:
                return self._extract_page_content_advanced(page, page_num, headers, footers)
            else:
                return self._extract_page_content_simple(page, page_num, headers, footers)

        try:
            # Render the page to an image
            pix = page.get_pixmap(dpi=self.ocr_dpi)
            img_data = pix.tobytes()

            # Create PIL Image from raw data
            img = Image.open(io.BytesIO(img_data))

            # Run OCR on the image
            ocr_text = pytesseract.image_to_string(
                img,
                lang=self.ocr_language,
                timeout=self.ocr_timeout,
                config='--psm 6'  # Assume a single block of text
            )

            # Check if OCR produced any text
            if not ocr_text.strip():
                self.logger.warning(f"OCR produced no text for page {page_num + 1}. Falling back to native extraction.")
                if self.use_advanced_reading_order:
                    return self._extract_page_content_advanced(page, page_num, headers, footers)
                else:
                    return self._extract_page_content_simple(page, page_num, headers, footers)

            # Basic processing to clean up OCR text
            ocr_text = self._clean_ocr_text(ocr_text)

            # Extract paragraphs (simple heuristic: double newlines)
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', ocr_text) if p.strip()]

            # Create a section for each paragraph
            page_content = []
            page_sections = []

            for para in paragraphs:
                # TODO: Add more sophisticated heading detection for OCR text
                section = Section(
                    content=para,
                    section_type=SectionType.PARAGRAPH,
                    level=0,
                    metadata={
                        'page': page_num + 1,
                        'ocr': True,
                        'ocr_lang': self.ocr_language,
                        'ocr_dpi': self.ocr_dpi
                    }
                )

                page_sections.append(section)

                # Add paragraph to page content
                if self.preserve_page_numbers:
                    page_content.append(f"{para} [Page {page_num + 1}]")
                else:
                    page_content.append(para)

            return "\n".join(page_content), page_sections

        except Exception as e:
            self.logger.error(
                f"OCR processing failed for page {page_num + 1}: {str(e)}. Falling back to native extraction.")
            if self.use_advanced_reading_order:
                return self._extract_page_content_advanced(page, page_num, headers, footers)
            else:
                return self._extract_page_content_simple(page, page_num, headers, footers)

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR output text by removing common artifacts."""
        if not text:
            return ""

        # Remove very short lines (likely noise)
        lines = [line for line in text.splitlines() if len(line.strip()) > 3]

        # Fix common OCR issues
        cleaned_lines = []
        for line in lines:
            # Replace common OCR errors
            line = line.replace('|', 'I')
            line = line.replace('¢', 'c')
            line = line.replace('—', '-')
            line = line.replace('–', '-')
            line = re.sub(r'(\w)\.(\w)', r'\1\2', line)  # Fix words split by periods

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _detect_columns(self, blocks: List[Dict[str, Any]], page_width: float) -> List[List[Dict[str, Any]]]:
        """
        Detect columns on a page based on X-coordinates of blocks.

        Args:
            blocks: List of text blocks.
            page_width: Width of the page.

        Returns:
            List of column definitions, where each column is a list of blocks.
        """
        if not blocks:
            return []

        # Extract x-center coordinates of blocks
        x_centers = []
        for block in blocks:
            x1, _, x2, _ = block["bbox"]
            x_center = (x1 + x2) / 2
            x_centers.append(x_center)

        # If there are very few blocks, assume a single column
        if len(x_centers) < 3:
            return [blocks]

        # Use clustering to identify column centers
        try:
            # Try using numpy for clustering if available
            import numpy as np
            from sklearn.cluster import KMeans

            # Reshape for KMeans
            X = np.array(x_centers).reshape(-1, 1)

            # Estimate number of columns (1-3)
            # Use silhouette score to determine optimal number
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_n_clusters = 1

            # Try 1-3 columns and pick the best
            for n_clusters in range(1, min(4, len(X))):
                if n_clusters == 1:
                    best_n_clusters = 1
                    break

                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                labels = kmeans.labels_

                # Calculate silhouette score if there are enough samples
                if len(set(labels)) > 1 and len(X) > 2:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters

            # Perform clustering with the optimal number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=0).fit(X)
            labels = kmeans.labels_

            # Create columns based on cluster labels
            columns = [[] for _ in range(best_n_clusters)]
            for i, block in enumerate(blocks):
                columns[labels[i]].append(block)

            # Sort columns from left to right
            column_centers = []
            for col in columns:
                x_sum = sum((block["bbox"][0] + block["bbox"][2]) / 2 for block in col)
                x_avg = x_sum / len(col) if col else 0
                column_centers.append((x_avg, col))

            columns = [col for _, col in sorted(column_centers)]

            return columns

        except (ImportError, Exception) as e:
            # Fallback to simple column detection
            self.logger.debug(f"KMeans clustering failed for column detection: {e}. Using simple approach.")

            # Simple approach: divide page into equal columns based on block distribution
            x_centers_sorted = sorted(x_centers)

            # Check for likely multi-column layout
            histogram = {}
            for x in x_centers_sorted:
                bin_idx = int(x / (page_width / 10))
                histogram[bin_idx] = histogram.get(bin_idx, 0) + 1

            # Check if there are distinct peaks in the histogram
            peaks = []
            for i in range(1, 9):
                if histogram.get(i, 0) > histogram.get(i - 1, 0) and histogram.get(i, 0) > histogram.get(i + 1, 0):
                    peaks.append(i)

            # If we found clear column peaks, use them
            num_columns = len(peaks) + 1 if peaks else 1

            if num_columns == 1:
                return [blocks]

            # Divide blocks into columns based on x-center
            column_width = page_width / num_columns
            columns = [[] for _ in range(num_columns)]

            for block in blocks:
                x1, _, x2, _ = block["bbox"]
                x_center = (x1 + x2) / 2
                column_idx = min(num_columns - 1, int(x_center / column_width))
                columns[column_idx].append(block)

            return columns

    def _order_blocks_by_reading_flow(self, blocks: List[Dict[str, Any]], columns: List[List[Dict[str, Any]]]) -> List[
        Dict[str, Any]]:
        """
        Order blocks following natural reading flow in columns.

        Args:
            blocks: All text blocks on the page.
            columns: Blocks grouped by column.

        Returns:
            List of blocks in proper reading order.
        """
        if not columns:
            # If column detection failed or returned empty, sort by y-position
            return sorted(blocks, key=lambda b: b["bbox"][1])

        # Process each column in left-to-right order
        ordered_blocks = []
        for column in columns:
            # Sort blocks within column by y-position (top to bottom)
            column_blocks = sorted(column, key=lambda b: b["bbox"][1])

            # Detect and handle special table structures
            column_blocks = self._handle_special_structures(column_blocks)

            ordered_blocks.extend(column_blocks)

        return ordered_blocks

    def _handle_special_structures(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle special structures like tables, maintaining their integrity.

        Args:
            blocks: Text blocks within a column.

        Returns:
            Blocks properly ordered with special structures preserved.
        """
        # Simple implementation, could be expanded to detect and handle tables
        # For now, just maintain the vertical order
        return blocks

    def _identify_pages_needing_ocr(self, pdf_document) -> List[int]:
        """
        Identify pages that likely need OCR due to low text content.

        Args:
            pdf_document: Open PDF document.

        Returns:
            List of page indices (0-based) that need OCR.
        """
        if not self.ocr_enabled or not TESSERACT_AVAILABLE:
            return []

        pages_needing_ocr = []

        for page_idx, page in enumerate(pdf_document):
            # Get text content using PyMuPDF
            text = page.get_text()

            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height

            # Calculate text density
            text_length = len(text.strip())
            chars_per_area = text_length / page_area if page_area > 0 else 0

            # Check for images
            has_images = page.get_images()

            # Determine if OCR is needed
            if text_length < 50 and has_images:
                # Very little text but has images - likely needs OCR
                pages_needing_ocr.append(page_idx)
            elif chars_per_area < 0.0005 and has_images:
                # Very low text density but has images - might need OCR
                pages_needing_ocr.append(page_idx)

        return pages_needing_ocr

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
