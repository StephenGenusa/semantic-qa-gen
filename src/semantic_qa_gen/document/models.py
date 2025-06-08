# filename: semantic_qa_gen/document/models.py

"""Data models for representing documents, chunks, questions, and related structures using Pydantic V2."""

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

# --- Use Pydantic V2 components ---
from pydantic import BaseModel, Field, field_validator, AliasChoices

# --- Enums (Remain the same) ---

class DocumentType(str, Enum):
    """Enumeration of supported source document types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

class SectionType(str, Enum):
    """Enumeration of logical section types within a document."""
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    IMAGE = "image"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"

    def __str__(self):
        return self.value

# --- Pydantic Models ---

class DocumentMetadata(BaseModel):
    """Container for metadata associated with a Document."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None
    custom: Dict[str, Any] = Field(default_factory=dict)

class Section(BaseModel):
    """Represents a structural section identified within a document."""
    content: str
    section_type: SectionType
    level: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """Represents the content and metadata of a loaded document."""
    content: str
    doc_type: DocumentType
    path: Optional[str] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Use List['Section'] for forward reference, handled automatically by Pydantic V2
    sections: Optional[List[Section]] = Field(default=None)

class Chunk(BaseModel):
    """Represents a semantically coherent chunk of text derived from a Document."""
    content: str
    id: str # Should generally be unique, consider default_factory if not always provided
    document_id: str
    sequence: int
    context: Dict[str, Any] = Field(default_factory=dict)
    # Ensure Section is defined above or handle forward reference appropriately
    preceding_headings: List[Section] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    """Stores the results of semantic analysis performed on a Chunk."""
    chunk_id: str
    # Add validation constraints using Field
    information_density: float = Field(default=0.5, ge=0.0, le=1.0)
    topic_coherence: float = Field(default=0.5, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_question_yield: Dict[str, int] = Field(
        default_factory=lambda: {"factual": 0, "inferential": 0, "conceptual": 0}
    )
    key_concepts: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # Optional: Add validator ensure the dictionary has the correct keys and non-negative values
    @field_validator('estimated_question_yield')
    def check_yield_format(cls, v):
        if not isinstance(v, dict):
            # Log warning or error - returning default for robustness
            print(f"WARNING: estimated_question_yield was not a dict, resetting. Value: {v}")
            return {"factual": 0, "inferential": 0, "conceptual": 0}

        expected_keys = {"factual", "inferential", "conceptual"}
        # Ensure all expected keys are present and values are non-negative ints
        validated_yield = {}
        valid = True
        for key in expected_keys:
            val = v.get(key)
            try:
                 num_val = int(val) if val is not None else 0
                 validated_yield[key] = max(0, num_val)
            except (ValueError, TypeError):
                 validated_yield[key] = 0 # Default to 0 if invalid value
                 valid = False

        # Check for extra keys (optional, depends on desired strictness)
        if set(v.keys()) != expected_keys or not valid:
             # Log warning if structure was corrected
             print(f"WARNING: Corrected estimated_question_yield structure/values. Original: {v}, Corrected: {validated_yield}")
             return validated_yield # Return the corrected/default version

        return v # Return the original if it was already valid


class Question(BaseModel):
    """Represents a generated question-answer pair."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(validation_alias=AliasChoices('text', 'question'))
    answer: str
    chunk_id: str
    context: str = Field(...,
                         description="The original chunk of text from which the question and answer were generated.")
    # Consider using Literal['factual', 'inferential', 'conceptual'] or an Enum for category
    category: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidationResult(BaseModel):
    """Stores the outcome of validating a single Question."""
    question_id: str
    is_valid: bool
    scores: Dict[str, float] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    suggested_improvements: Optional[str] = None

    # Note: Pydantic V2 models don't typically implement __bool__.
    # Check validity directly using result.is_valid.

    # __str__ method should work fine
    def __str__(self) -> str:
        """Provides a concise string representation."""
        status = "Valid" if self.is_valid else "Invalid"
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        reason_str = f": {'; '.join(self.reasons)}" if self.reasons else ""
        return f"Q:{self.question_id} -> {status} (Scores: [{score_str}]{reason_str})"

# Optional: If you encounter issues with forward references ('Section' in Document/Chunk),
# uncomment these lines after all models are defined. Usually not needed in Pydantic V2
# for simple string type hints.
# Document.model_rebuild()
# Chunk.model_rebuild()

