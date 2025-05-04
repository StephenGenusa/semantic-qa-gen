# document/models.py

"""Data models for representing documents, chunks, questions, and related structures."""

from dataclasses import dataclass, field # Import field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

# --- Enums ---

class DocumentType(str, Enum):
    """Enumeration of supported source document types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    UNKNOWN = "unknown" # Added for robustness

    def __str__(self):
        return self.value


class SectionType(str, Enum):
    """Enumeration of logical section types within a document."""
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"  # Potentially useful distinction
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item" # Changed LIST to LIST_ITEM for clarity
    TABLE = "table"
    CODE_BLOCK = "code_block" # Changed CODE to CODE_BLOCK
    QUOTE = "quote"
    IMAGE = "image"
    FOOTNOTE = "footnote"
    HEADER = "header" # For detected headers
    FOOTER = "footer" # For detected footers
    OTHER = "other"

    def __str__(self):
        return self.value

# --- Dataclasses ---

@dataclass
class DocumentMetadata:
    """Container for metadata associated with a Document."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None # Consider using datetime objects if parsing is reliable
    source: Optional[str] = None # Typically the file path or URL
    language: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict) # Initialize custom metadata

    def __post_init__(self):
        """Ensure custom dictionary exists."""
        if self.custom is None: # Should be handled by default_factory, but defensive check
            self.custom = {}


@dataclass
class Document:
    """Represents the content and metadata of a loaded document."""
    content: str # The full text content of the document
    doc_type: DocumentType # The detected type of the document
    path: Optional[str] = None # Original path of the document file
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata) # Document metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique ID for the document instance
    # Added sections field to formally hold structural elements extracted by loaders/processors
    # repr=False keeps the representation clean when printing Document objects.
    sections: Optional[List['Section']] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure metadata is initialized if default_factory wasn't triggered."""
        # This __post_init__ might be redundant now with field(default_factory=...)
        # Keeping id generation here if needed for complex scenarios later.
        # if self.id is None: self.id = str(uuid.uuid4()) # Handled by default_factory
        if self.metadata is None: self.metadata = DocumentMetadata()


@dataclass
class Section:
    """Represents a structural section identified within a document."""
    content: str # Text content of the section
    section_type: SectionType # The type of section (e.g., HEADING, PARAGRAPH)
    level: int = 0 # Heading level (1-6 for headings, 0 for others) or list indentation level
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional metadata (e.g., page number, style)

    def __post_init__(self):
        """Ensure metadata dictionary exists."""
        if self.metadata is None: # Should be handled by default_factory
            self.metadata = {}


@dataclass
class Chunk:
    """Represents a semantically coherent chunk of text derived from a Document."""
    content: str # Text content of the chunk
    id: str # Unique ID for the chunk
    document_id: str # ID of the source Document
    sequence: int # Order of the chunk within the document
    context: Dict[str, Any] = field(default_factory=dict) # Contextual info (title, section path, etc.)
    # Store preceding headings relevant to this chunk's context
    preceding_headings: List[Section] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default values if necessary."""
        if self.context is None: self.context = {}
        if self.preceding_headings is None: self.preceding_headings = []


@dataclass
class AnalysisResult:
    """Stores the results of semantic analysis performed on a Chunk."""
    chunk_id: str
    information_density: float = 0.5 # Normalized score (0.0 to 1.0)
    topic_coherence: float = 0.5 # Normalized score (0.0 to 1.0)
    complexity: float = 0.5 # Normalized score (0.0 to 1.0)
    # Estimated number of questions possible per category
    estimated_question_yield: Dict[str, int] = field(default_factory=lambda: {"factual": 0, "inferential": 0, "conceptual": 0})
    key_concepts: List[str] = field(default_factory=list) # List of key terms/concepts found
    notes: Optional[str] = None # Any additional notes from the analysis LLM

    def __post_init__(self):
        """Ensure default structures are initialized."""
        if self.estimated_question_yield is None:
            self.estimated_question_yield = {"factual": 0, "inferential": 0, "conceptual": 0}
        if self.key_concepts is None:
            self.key_concepts = []


@dataclass
class Question:
    """Represents a generated question-answer pair."""
    id: str # Unique ID for the question
    text: str # The question text
    answer: str # The generated answer text
    chunk_id: str # ID of the source Chunk
    category: str # Cognitive level ('factual', 'inferential', 'conceptual')
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional metadata (e.g., generation order, scores)

    def __post_init__(self):
        """Ensure metadata dictionary exists."""
        if self.metadata is None: self.metadata = {}


@dataclass
class ValidationResult:
    """Stores the outcome of validating a single Question."""
    question_id: str # ID of the question being validated
    is_valid: bool # Overall validity based on configured validators and thresholds
    # Dictionary of scores from individual validation checks (e.g., {'factual_accuracy': 0.9})
    scores: Dict[str, float] = field(default_factory=dict)
    # List of reasons provided by validators (especially for failure)
    reasons: List[str] = field(default_factory=list)
    # Optional textual suggestions for improving the question/answer
    suggested_improvements: Optional[str] = None

    def __post_init__(self):
        """Initialize default lists/dicts."""
        if self.scores is None: self.scores = {}
        if self.reasons is None: self.reasons = []

    def __bool__(self) -> bool:
        """Allows treating the result directly as a boolean (True if valid)."""
        return self.is_valid

    def __str__(self) -> str:
        """Provides a concise string representation."""
        status = "Valid" if self.is_valid else "Invalid"
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        reason_str = f": {'; '.join(self.reasons)}" if self.reasons else ""
        return f"Q:{self.question_id} -> {status} (Scores: [{score_str}]{reason_str})"

