"""Markdown file loader for SemanticQAGen."""

import os
import re
import logging
from typing import Dict, Any, Optional
# Need to handle case where commonmark is not available
try:
    import commonmark
    COMMONMARK_AVAILABLE = True
except ImportError:
    COMMONMARK_AVAILABLE = False

from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata
from semantic_qa_gen.utils.error import DocumentError


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Markdown loader.

        Args:
            config: Configuration dictionary for the loader.
        """
        super().__init__(config)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.logger = logging.getLogger(__name__)  # Initialize logger for all cases

        if not COMMONMARK_AVAILABLE:
            self.logger.warning("commonmark package not installed. Markdown parsing will be limited.")

    def load(self, path: str) -> Document:
        """
        Load a document from a Markdown file.
        
        Args:
            path: Path to the Markdown file.
            
        Returns:
            Loaded Document object.
            
        Raises:
            DocumentError: If the Markdown file cannot be loaded.
        """
        if not self.supports_type(path):
            raise DocumentError(f"Unsupported file type for MarkdownLoader: {path}")
        
        try:
            with open(path, 'r', encoding=self.encoding) as file:
                content = file.read()
            
            metadata = self.extract_front_matter(content) if self.extract_metadata else DocumentMetadata()
            
            # Add file information to metadata if not present
            if not metadata.title:
                file_name = os.path.basename(path)
                title, _ = os.path.splitext(file_name)
                metadata.title = title
                
            metadata.source = path
            
            # Remove front matter from content if present
            content = self.strip_front_matter(content)
            
            return Document(
                content=content,
                doc_type=DocumentType.MARKDOWN,
                path=path,
                metadata=metadata
            )
            
        except UnicodeDecodeError:
            raise DocumentError(
                f"Failed to decode Markdown file with encoding {self.encoding}: {path}"
            )
        except Exception as e:
            raise DocumentError(f"Failed to load Markdown file: {str(e)}")
    
    def supports_type(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file type.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this is a Markdown file, False otherwise.
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in ['.md', '.markdown']
    
    def extract_front_matter(self, content: str) -> DocumentMetadata:
        """
        Extract front matter metadata from Markdown content.
        
        Args:
            content: Markdown content.
            
        Returns:
            DocumentMetadata object.
        """
        metadata = DocumentMetadata()
        
        # Look for YAML front matter enclosed in --- or +++
        front_matter_match = re.match(r'^(---|\+\+\+)\n(.*?)\n(---|\+\+\+)', content, re.DOTALL)
        if not front_matter_match:
            return metadata
            
        front_matter_text = front_matter_match.group(2)
        
        # Simple parsing of key: value pairs
        for line in front_matter_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                    
                if key == 'title':
                    metadata.title = value
                elif key == 'author':
                    metadata.author = value
                elif key == 'date':
                    metadata.date = value
                elif key == 'language':
                    metadata.language = value
                else:
                    # Store other metadata as custom
                    if metadata.custom is None:
                        metadata.custom = {}
                    metadata.custom[key] = value
                
        return metadata
    
    def strip_front_matter(self, content: str) -> str:
        """
        Remove front matter from Markdown content.
        
        Args:
            content: Markdown content.
            
        Returns:
            Content with front matter removed.
        """
        return re.sub(r'^(---|\+\+\+)\n(.*?)\n(---|\+\+\+)\n?', '', content, flags=re.DOTALL)

    def parse_document_structure(self, content: str) -> Dict[str, Any]:
        """
        Parse the structure of a Markdown document.

        This is a helper method that extracts headings and their hierarchy
        from the document.

        Args:
            content: Markdown content.

        Returns:
            Dictionary containing document structure information.
        """
        if not COMMONMARK_AVAILABLE:
            # Fall back to simpler regex-based parsing
            return self._parse_document_structure_fallback(content)

        parser = commonmark.Parser()
        ast = parser.parse(content)

        structure = {
            'headings': [],
            'sections': []
        }

        current_heading = None
        current_level = 0
        current_section = []

        # Fixed walker implementation - needs to check node[0] and node[1]
        for node, entering in ast.walker():
            if node.t == 'heading' and entering:
                # Save previous section when we find a new heading
                if current_heading is not None:
                    structure['sections'].append({
                        'heading': current_heading,
                        'level': current_level,
                        'content': ''.join(current_section)
                    })
                    current_section = []

                # Extract heading text by collecting all text nodes within the heading
                heading_text = ''
                for child in node.walker():
                    node_obj, child_entering = child
                    if node_obj.t == 'text' and child_entering:
                        heading_text += node_obj.literal

                current_heading = heading_text
                current_level = node.level
                structure['headings'].append({
                    'text': current_heading,
                    'level': node.level
                })
            elif node.t == 'text' and entering and node.literal:
                if current_section is not None:
                    current_section.append(node.literal)

        # Add the last section
        if current_heading is not None:
            structure['sections'].append({
                'heading': current_heading,
                'level': current_level,
                'content': ''.join(current_section)
            })

        return structure

    def _parse_document_structure_fallback(self, content: str) -> Dict[str, Any]:
        """
        Simple fallback parser for when commonmark is not available.

        Args:
            content: Markdown content.

        Returns:
            Dictionary containing document structure information.
        """
        structure = {
            'headings': [],
            'sections': []
        }

        lines = content.split('\n')
        current_heading = None
        current_level = 0
        current_section = []

        for line in lines:
            # Check if this is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                if current_heading is not None:
                    # Store the previous section
                    structure['sections'].append({
                        'heading': current_heading,
                        'level': current_level,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                current_heading = heading_text
                current_level = level
                structure['headings'].append({
                    'text': heading_text,
                    'level': level
                })
            else:
                if current_heading is not None:  # Only add content if we have a heading
                    current_section.append(line)

        # Add the last section
        if current_heading is not None:
            structure['sections'].append({
                'heading': current_heading,
                'level': current_level,
                'content': '\n'.join(current_section)
            })

        return structure