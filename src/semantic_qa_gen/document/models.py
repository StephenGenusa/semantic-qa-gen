"""Document models for SemanticQAGen."""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import uuid


class DocumentType(str, Enum):
    """Types of supported documents."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None
    custom: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize custom metadata if not provided."""
        if self.custom is None:
            self.custom = {}


@dataclass
class Document:
    """Represents a document to be processed."""
    content: str
    doc_type: DocumentType
    path: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None
    id: str = None
    
    def __post_init__(self):
        """Initialize document ID if not provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())
            
        if self.metadata is None:
            self.metadata = DocumentMetadata()


class SectionType(str, Enum):
    """Types of document sections."""
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    IMAGE = "image"
    OTHER = "other"


@dataclass
class Section:
    """Represents a section of a document."""
    content: str
    section_type: SectionType
    level: int = 0  # Heading level (0 for non-headings)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """Represents a semantically coherent chunk of text."""
    content: str
    id: str
    document_id: str
    sequence: int
    context: Dict[str, Any]
    preceding_headings: List[Section] = None
    
    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.preceding_headings is None:
            self.preceding_headings = []


@dataclass
class AnalysisResult:
    """Result of semantic analysis on a chunk."""
    chunk_id: str
    information_density: float  # 0.0 to 1.0
    topic_coherence: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    estimated_question_yield: Dict[str, int]  # category -> count
    key_concepts: List[str]
    notes: Optional[str] = None


@dataclass
class Question:
    """Represents a generated question."""
    id: str
    text: str
    answer: str
    chunk_id: str
    category: str  # factual, inferential, conceptual
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Result of validating a question-answer pair."""
    question_id: str
    is_valid: bool
    scores: Dict[str, float]
    reasons: List[str]
    suggested_improvements: Optional[str] = None

