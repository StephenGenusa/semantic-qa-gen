"""Text file loader for SemanticQAGen."""

import os
from typing import Dict, Any, Optional

from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata
from semantic_qa_gen.utils.error import DocumentError


class TextLoader(BaseLoader):
    """
    Loader for plain text files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text loader.
        
        Args:
            config: Configuration dictionary for the loader.
        """
        super().__init__(config)
        self.encoding = self.config.get('encoding', 'utf-8')
    
    def load(self, path: str) -> Document:
        """
        Load a document from a text file.
        
        Args:
            path: Path to the text file.
            
        Returns:
            Loaded Document object.
            
        Raises:
            DocumentError: If the text file cannot be loaded.
        """
        if not self.supports_type(path):
            raise DocumentError(f"Unsupported file type for TextLoader: {path}")
        
        try:
            with open(path, 'r', encoding=self.encoding) as file:
                content = file.read()
            
            metadata = self.extract_metadata(path)
            
            return Document(
                content=content,
                doc_type=DocumentType.TEXT,
                path=path,
                metadata=metadata
            )
            
        except UnicodeDecodeError:
            raise DocumentError(
                f"Failed to decode text file with encoding {self.encoding}: {path}"
            )
        except Exception as e:
            raise DocumentError(f"Failed to load text file: {str(e)}")
    
    def supports_type(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file type.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this is a text file, False otherwise.
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in ['.txt', '.text']
    
    def extract_metadata(self, path: str) -> DocumentMetadata:
        """
        Extract metadata from a text file.
        
        Args:
            path: Path to the text file.
            
        Returns:
            DocumentMetadata object.
        """
        file_name = os.path.basename(path)
        title, _ = os.path.splitext(file_name)
        
        metadata = DocumentMetadata(
            title=title,
            source=path,
        )
        
        # Try to detect language and other metadata
        # This could be enhanced with language detection libraries
        
        return metadata
