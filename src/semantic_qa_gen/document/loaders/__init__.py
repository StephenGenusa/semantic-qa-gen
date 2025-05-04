"""Document loaders for SemanticQAGen."""

from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.loaders.text import TextLoader
from semantic_qa_gen.document.loaders.pdf import PDFLoader
from semantic_qa_gen.document.loaders.markdown import MarkdownLoader
from semantic_qa_gen.document.loaders.docx import DocxLoader

__all__ = [
    'BaseLoader',
    'TextLoader',
    'PDFLoader',
    'MarkdownLoader',
    'DocxLoader'
]