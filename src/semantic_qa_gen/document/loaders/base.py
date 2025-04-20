"""Base document loader interface for SemanticQAGen."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata
from semantic_qa_gen.utils.error import DocumentError


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    
    This class defines the interface that all document loaders must implement.
    Document loaders are responsible for reading files and converting them
    to Document objects for processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document loader.
        
        Args:
            config: Configuration dictionary for the loader.
        """
        self.config = config or {}
    
    @abstractmethod
    def load(self, path: str) -> Document:
        """
        Load a document from a file.
        
        Args:
            path: Path to the document file.
            
        Returns:
            Loaded Document object.
            
        Raises:
            DocumentError: If the document cannot be loaded.
        """
        pass
    
    @abstractmethod
    def supports_type(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file type.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this loader supports the file type, False otherwise.
        """
        pass
    
    def extract_metadata(self, path: str) -> DocumentMetadata:
        """
        Extract metadata from a document file.
        
        Args:
            path: Path to the document file.
            
        Returns:
            DocumentMetadata object.
        """
        # Default implementation returns empty metadata
        return DocumentMetadata()
