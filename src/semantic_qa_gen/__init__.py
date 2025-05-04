"""
SemanticQAGen: A high-quality question-answer generation library

This library processes various document formats (PDF, DOCX, Markdown, TXT)
and generates semantic question-answer pairs using language models with a
focus on quality, accuracy, and educational value.

Features:
- Multi-format document processing (PDF, DOCX, Markdown, TXT)
- Intelligent document chunking and analysis
- Advanced PDF processing with column detection and natural reading order
- OCR support for scanned documents
- Multiple LLM provider support
- Customizable prompts and generation strategies

Basic Usage:
    from semantic_qa_gen import SemanticQAGen
    from semantic_qa_gen.config.schema import ProjectConfig

    # Create a project with default settings
    project = SemanticQAGen()

    # Or with custom configuration
    config = ProjectConfig(
        llm_service={
            "provider": "openai",
            "model": "gpt-4"
        },
        processing={
            "chunk_size": 2000,
            "chunk_overlap": 200
        }
    )
    project = SemanticQAGen(config)

    # Process a document
    results = project.process_document("path/to/document.pdf")

    # Save the results
    results.save("output.json")

    # Advanced PDF processing with OCR and column detection
    pdf_config = {
        "ocr_enabled": True,
        "use_advanced_reading_order": True
    }
    results = project.process_document("path/to/scanned_document.pdf", loader_config=pdf_config)
"""


__version__ = "0.1.0"
__author__ = "Stephen Genusa"
__license__ = "MIT"


import logging
from semantic_qa_gen.version import __version__
from semantic_qa_gen.semantic_qa_gen import SemanticQAGen
from semantic_qa_gen.config.schema import (
    SemanticQAGenConfig,
    DocumentConfig,
    ChunkingConfig,
    LLMServiceConfig,
    QuestionGenerationConfig,
    ValidationConfig,
    OutputConfig
)
from semantic_qa_gen.document.models import Document
from semantic_qa_gen.pipeline.semantic import SemanticPipeline
from semantic_qa_gen.utils.error import SemanticQAGenError
from semantic_qa_gen.utils.project import ProjectManager

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'SemanticQAGen',
    'SemanticQAGenConfig',
    'DocumentConfig',
    'ChunkingConfig',
    'LLMServiceConfig',
    'QuestionGenerationConfig',
    'ValidationConfig',
    'OutputConfig',
    'Document',
    'SemanticPipeline',
    'SemanticQAGenError',
    'ProjectManager',
    '__version__'
]
