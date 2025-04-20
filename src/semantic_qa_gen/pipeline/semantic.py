# semantic_qa_gen/pipeline/semantic.py

"""Semantic processing pipeline with optimized concurrency."""

import os
import logging
import asyncio
import time
import traceback
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Document, Chunk, AnalysisResult, Question
from semantic_qa_gen.document.processor import DocumentProcessor
from semantic_qa_gen.chunking.engine import ChunkingEngine
from semantic_qa_gen.chunking.analyzer import SemanticAnalyzer
from semantic_qa_gen.llm.router import TaskRouter
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.question.processor import QuestionProcessor
from semantic_qa_gen.utils.error import SemanticQAGenError
from semantic_qa_gen.utils.progress import ProgressReporter, ProcessingStage
from semantic_qa_gen.utils.checkpoint import CheckpointManager
from semantic_qa_gen.output.formatter import OutputFormatter


class SemanticPipeline:
    """
    Main processing pipeline for SemanticQAGen.
    
    This class orchestrates the entire document processing workflow,
    from loading documents to generating and validating questions.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the semantic pipeline with optimized components.
        
        Args:
            config_manager: Configuration manager.
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = logging.getLogger(__name__)
        
        # Set up components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {
            "start_time": 0,
            "total_time": 0,
            "stage_times": {
                "loading": {},
                "chunking": {},
                "analysis": {},
                "question_generation": {},
                "validation": {},
                "output": {}
            }
        }
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components with optimal settings."""
        # Document processing
        self.document_processor = DocumentProcessor(self.config_manager)
        self.chunking_engine = ChunkingEngine(self.config_manager)
        
        # LLM services
        self.prompt_manager = PromptManager()
        self.task_router = TaskRouter(self.config_manager)
        
        # Analysis
        self.semantic_analyzer = SemanticAnalyzer(self.task_router, self.prompt_manager)
        
        # Question generation
        self.question_generator = QuestionGenerator(
            self.config_manager, 
            self.task_router,
            self.prompt_manager
        )
        
        # Validation
        self.validation_engine = ValidationEngine(
            self.config_manager,
            self.task_router,
            self.prompt_manager
        )
        
        # Question processing
        self.question_processor = QuestionProcessor(
            self.config_manager,
            self.question_generator,
            self.validation_engine
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Progress reporting
        self.progress_reporter = ProgressReporter(
            show_progress_bar=True
        )
        
        # Output formatter
        self.output_formatter = OutputFormatter(self.config_manager)
    
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document to generate question-answer pairs with optimized performance.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Processing results including questions and statistics.
            
        Raises:
            SemanticQAGenError: If processing fails.
        """
        self._start_timer("total")
        
        try:
            # Stage 1: Load document
            self._start_timer("loading")
            self.progress_reporter.update_stage(ProcessingStage.LOADING)
            document = self.document_processor.load_document(document_path)
            
            # Check for existing checkpoint
            checkpoint = None
            if self.config.processing.enable_checkpoints:
                checkpoint = self.checkpoint_manager.load_checkpoint(document)
            
            self._end_timer("loading")
            
            # Stage 2: Chunk document
            self._start_timer("chunking")
            self.progress_reporter.update_stage(ProcessingStage.CHUNKING)
            chunks = self.chunking_engine.chunk_document(document, self.document_processor)
            self.progress_reporter.update_progress(len(chunks), len(chunks))
            self._end_timer("chunking")
            
            # Apply checkpoint if available
            if checkpoint:
                self.logger.info(f"Resuming from checkpoint: {checkpoint}")
                # Extract checkpoint data
                completed_chunks = set(checkpoint.get('completed_chunks', []))
                current_chunk_idx = checkpoint.get('current_chunk_idx', 0)

                # Filter chunks to only process incomplete ones
                chunks_to_process = [c for c in chunks if c.id not in completed_chunks]

                if chunks_to_process:
                    self.logger.info(f"Resuming processing from chunk {current_chunk_idx}, "
                                    f"{len(chunks) - len(chunks_to_process)} chunks already completed")
                else:
                    self.logger.info("All chunks already processed in checkpoint")
            else:
                chunks_to_process = chunks

            # Stage 3: Analyze chunks
            self._start_timer("analysis")
            self.progress_reporter.update_stage(ProcessingStage.ANALYSIS)

            # Process chunks with optimal concurrency
            analyses = await self._process_chunks_with_batching(chunks_to_process)

            self._end_timer("analysis")

            # Stage 4: Generate questions
            self._start_timer("question_generation")
            self.progress_reporter.update_stage(ProcessingStage.QUESTION_GENERATION)
            all_questions, generation_stats = await self.question_processor.process_chunks(
                chunks_to_process,
                analyses,
                self.progress_reporter
            )
            self._end_timer("question_generation")

            # Stage 5: Finalize results
            self._start_timer("output")
            self.progress_reporter.update_stage(ProcessingStage.OUTPUT)

            # Create the final result
            questions_list = []
            for chunk_id, questions in all_questions.items():
                questions_list.extend(questions)

            # Convert questions to dictionaries for output formatting
            questions_dicts = [
                {
                    "id": q.id,
                    "text": q.text,
                    "answer": q.answer,
                    "category": q.category,
                    "metadata": q.metadata,
                    "chunk_id": q.chunk_id
                } for q in questions_list
            ]

            # Add document information
            document_info = {
                "id": document.id,
                "title": document.metadata.title if document.metadata else None,
                "path": document_path,
                "author": document.metadata.author if document.metadata else None,
                "date": document.metadata.date if document.metadata else None
            }

            # Add performance metrics to statistics
            generation_stats["performance"] = self.performance_metrics["stage_times"]

            # Generate output with formatter
            formatted_output = self.output_formatter.format_questions(
                questions=questions_dicts,
                document_info=document_info,
                statistics=generation_stats
            )

            self._end_timer("output")
            self._end_timer("total")

            # Add overall time metrics
            generation_stats["performance"]["total_seconds"] = self.performance_metrics["total_time"]

            # Complete
            self.progress_reporter.update_stage(ProcessingStage.COMPLETE)
            self.progress_reporter.complete({
                "total_questions": len(questions_list),
                "chunks_processed": len(chunks),
                "processing_time": f"{self.performance_metrics['total_time']:.1f} seconds"
            })

            # If checkpointing is enabled, save a final checkpoint
            if self.config.processing.enable_checkpoints:
                self.save_checkpoint(document, chunks, len(chunks), generation_stats)
            
            return {
                "formatted_output": formatted_output,
                "questions": questions_dicts,
                "document": document_info,
                "statistics": generation_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}", exc_info=True)
            
            if self.config.processing.debug_mode:
                # Print detailed traceback in debug mode
                traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                self.logger.debug(f"Detailed error traceback:\n{traceback_str}")
                
            raise SemanticQAGenError(f"Failed to process document: {str(e)}")

    async def _process_chunks_with_batching(self, chunks: List[Chunk]) -> Dict[str, AnalysisResult]:
        """
        Process chunks in batches for better performance and resource management.

        Args:
            chunks: List of document chunks to analyze.

        Returns:
            Dictionary mapping chunk IDs to analysis results.
        """
        results = {}
        concurrency = self.config.processing.concurrency
        semaphore = asyncio.Semaphore(concurrency)
        total_chunks = len(chunks)

        async def process_chunk(chunk: Chunk, index: int) -> Tuple[str, AnalysisResult]:
            """Process a single chunk with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    self.logger.debug(f"Analyzing chunk {index + 1}/{total_chunks}: {chunk.id}")
                    result = await self.semantic_analyzer.analyze_chunk(chunk)

                    # Update progress
                    self.progress_reporter.update_progress(index + 1, total_chunks)

                    return chunk.id, result
                except Exception as e:
                    self.logger.error(f"Failed to analyze chunk {chunk.id}: {str(e)}")
                    # Create a minimal fallback result
                    return chunk.id, AnalysisResult(
                        chunk_id=chunk.id,
                        information_density=0.5,
                        topic_coherence=0.5,
                        complexity=0.5,
                        estimated_question_yield={
                            "factual": 2,
                            "inferential": 1,
                            "conceptual": 1
                        },
                        key_concepts=[],
                        notes="Analysis failed, using default values"
                    )

        # Create tasks for all chunks
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]

        # Process all chunks concurrently with bounded concurrency
        chunk_results = await asyncio.gather(*tasks)

        # Convert results to dictionary
        for chunk_id, result in chunk_results:
            results[chunk_id] = result

        return results

    def _start_timer(self, stage_name: str) -> None:
        """Start timing a processing stage."""
        if stage_name == "total":
            self.performance_metrics["start_time"] = time.time()
        else:
            self.performance_metrics["stage_times"][stage_name] = {
                "start": time.time()
            }
    
    def _end_timer(self, stage_name: str) -> None:
        """End timing a processing stage and record duration."""
        now = time.time()
        if stage_name == "total":
            self.performance_metrics["total_time"] = now - self.performance_metrics["start_time"]
        elif stage_name in self.performance_metrics["stage_times"]:
            stage = self.performance_metrics["stage_times"][stage_name]
            stage["end"] = now
            stage["duration"] = now - stage["start"]
            self.logger.debug(f"Stage '{stage_name}' completed in {stage['duration']:.2f}s")
    
    def save_checkpoint(self, document: Document, chunks: List[Chunk], 
                      current_chunk_idx: int, stats: Dict[str, Any]) -> str:
        """
        Save a processing checkpoint.
        
        Args:
            document: The document being processed.
            chunks: List of document chunks.
            current_chunk_idx: Index of the current chunk being processed.
            stats: Processing statistics.
            
        Returns:
            Path to the saved checkpoint.
        """
        return self.checkpoint_manager.save_checkpoint(
            document=document,
            processed_chunks=chunks,
            current_chunk_idx=current_chunk_idx,
            stats=stats
        )
