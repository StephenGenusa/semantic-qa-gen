# semantic_qa_gen/semantic_qa_gen.py
"""
SemanticQAGen - A Python library for generating high-quality question-answer pairs from text content.

This library processes text content with enhanced semantic understanding, analyzes information density,
and generates diverse questions across multiple cognitive levels for AI model training.

Key Features:
- Semantic chunking of documents with context preservation
- Information density analysis
- Multi-level question generation (factual, inferential, conceptual)
- Flexible LLM provider integration (OpenAI, local models like Ollama)
- Comprehensive validation system
- Multiple output formats

Built upon concepts from the augmentoolkit project, with significant enhancements
in architecture, configurability, and extensibility.
"""


import asyncio
import logging
import json
import os
from typing import Optional, Dict, Any, List, Union

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.processor import DocumentProcessor
from semantic_qa_gen.document.models import Document, Chunk, Question
from semantic_qa_gen.chunking.engine import ChunkingEngine
from semantic_qa_gen.utils.logging import setup_logger
from semantic_qa_gen.utils.error import SemanticQAGenError
from semantic_qa_gen.utils.progress import ProgressReporter, ProcessingStage
from semantic_qa_gen.utils.checkpoint import CheckpointManager
from semantic_qa_gen.pipeline.semantic import SemanticPipeline


class SemanticQAGen:
    """
    Main interface for the SemanticQAGen library.

    This class provides a high-level API for generating question-answer pairs
    from text documents. It handles configuration management, document processing,
    and output formatting.
    """

    def __init__(self, config_path: Optional[str] = None,
                config_dict: Optional[Dict[str, Any]] = None,
                verbose: bool = False):
        """
        Initialize SemanticQAGen.

        Args:
            config_path: Optional path to configuration file.
            config_dict: Optional configuration dictionary.
            verbose: Whether to enable verbose logging.

        Raises:
            SemanticQAGenError: If configuration fails.
        """
        # Set up logging
        log_level = "DEBUG" if verbose else "INFO"
        self.logger = setup_logger(level=log_level)
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path, config_dict)
            self.config = self.config_manager.config
            
            # Update log level from config if not explicitly set to verbose
            if not verbose and hasattr(self.config, 'processing'):
                log_level_name = self.config.processing.log_level
                numeric_level = getattr(logging, log_level_name.upper(), None)
                if isinstance(numeric_level, int):
                    logging.getLogger().setLevel(numeric_level)
            
            # Initialize pipeline
            self.pipeline = SemanticPipeline(self.config_manager)
            
            self.logger.info(f"SemanticQAGen {__version__} initialized")
            
        except Exception as e:
            raise SemanticQAGenError(f"Failed to initialize SemanticQAGen: {str(e)}")
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document to generate question-answer pairs.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dictionary containing questions, statistics, and other data.
            
        Raises:
            SemanticQAGenError: If processing fails.
            FileNotFoundError: If the document does not exist.
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
            
        self.logger.info(f"Processing document: {document_path}")
        
        # Process document using asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            return loop.run_until_complete(self.pipeline.process_document(document_path))
        except Exception as e:
            # Enhanced error reporting
            if isinstance(e, SemanticQAGenError):
                raise e
            else:
                raise SemanticQAGenError(f"Failed to process document: {str(e)}")
    
    def save_questions(self, result: Dict[str, Any], 
                      output_path: str,
                      format_name: Optional[str] = None) -> str:
        """
        Save generated questions to a file.
        
        Args:
            result: Results from process_document.
            output_path: Path where to save the output.
            format_name: Format to save in (json, csv, etc.). Defaults to config value.
            
        Returns:
            Path to the saved file.
            
        Raises:
            SemanticQAGenError: If saving fails.
        """
        try:
            # Ensure we have the output formatter
            output_formatter = getattr(self.pipeline, 'output_formatter', None)
            if not output_formatter:
                from semantic_qa_gen.output.formatter import OutputFormatter
                output_formatter = OutputFormatter(self.config_manager)
            
            # If format isn't specified in arguments, use the one from configuration
            if not format_name:
                format_name = self.config.output.format
            
            # Format and save
            questions = result.get('questions', [])
            document = result.get('document', {})
            statistics = result.get('statistics', {})
            
            # If we already have formatted output, use it directly
            if 'formatted_output' in result:
                formatted_data = result['formatted_output']
                saved_path = output_formatter.save_to_file(
                    formatted_data,
                    output_path,
                    format_name
                )
            else:
                # Otherwise format and save
                saved_path = output_formatter.format_and_save(
                    questions=questions,
                    document_info=document,
                    statistics=statistics,
                    output_path=output_path,
                    format_name=format_name
                )
                
            self.logger.info(f"Saved questions to {saved_path}")
            return saved_path
            
        except Exception as e:
            raise SemanticQAGenError(f"Failed to save questions: {str(e)}")
    
        def create_qa_retriever(self, result: Dict[str, Any], api_key: Optional[str] = None) -> Any:
        """
        Create a LlamaIndex retriever from generated QA pairs.

        Args:
            result: Results from process_document.
            api_key: Optional OpenAI API key.

        Returns:
            LlamaIndex retriever object for RAG applications.

        Raises:
            SemanticQAGenError: If retriever creation fails.
            ImportError: If LlamaIndex is not available.
        """
        try:
            from semantic_qa_gen.utils.rag_tools import QARetrieverBuilder

            questions = []
            for q_dict in result.get('questions', []):
                # Convert dict representation back to Question objects
                questions.append(Question(
                    id=q_dict.get('id', ''),
                    text=q_dict.get('text', q_dict.get('question', '')),
                    answer=q_dict.get('answer', ''),
                    chunk_id=q_dict.get('chunk_id', ''),
                    category=q_dict.get('category', 'factual'),
                    metadata=q_dict.get('metadata', {})
                ))

            retriever = QARetrieverBuilder.build_retriever(questions, api_key)
            self.logger.info(f"Created RAG retriever from {len(questions)} QA pairs")
            return retriever

        except ImportError as e:
            raise ImportError(
                "LlamaIndex is required for creating a retriever. "
                "Install with: pip install llama-index"
            )
        except Exception as e:
            raise SemanticQAGenError(f"Failed to create retriever: {str(e)}")


    
    def create_default_config_file(self, output_path: str) -> None:
        """
        Create a default configuration file.

        Args:
            output_path: Path where to save the configuration file.

        Raises:
            SemanticQAGenError: If configuration cannot be saved.
        """
        try:
            # Create a default configuration
            default_config = ConfigManager()

            # Save to file
            default_config.save_config(output_path)
            self.logger.info(f"Default configuration saved to {output_path}")

        except Exception as e:
            raise SemanticQAGenError(f"Failed to create default configuration: {str(e)}")
