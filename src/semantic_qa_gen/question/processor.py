"""Memory-optimized question processor."""

import logging
import asyncio
import gc
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.utils.error import ValidationError
from semantic_qa_gen.utils.progress import ProgressReporter


class QuestionProcessor:
    """
    Memory-optimized processor for generating and validating questions.
    """
    
    def __init__(self, 
                config_manager: ConfigManager,
                question_generator: QuestionGenerator,
                validation_engine: ValidationEngine):
        """Initialize the question processor."""
        self.config_manager = config_manager
        self.config = config_manager.get_section("question_generation")
        self.question_generator = question_generator
        self.validation_engine = validation_engine
        self.logger = logging.getLogger(__name__)
        
        # Cache to store chunks by ID for efficient retrieval
        self._chunk_cache = {}
    
    async def process_chunk(self, 
                          chunk: Chunk, 
                          analysis: AnalysisResult,
                          progress_reporter: Optional[ProgressReporter] = None) -> Tuple[List[Question], Dict[str, Any]]:
        """Process a document chunk to generate and validate questions."""
        stats = {
            "chunk_id": chunk.id,
            "generated_questions": 0,
            "valid_questions": 0,
            "rejected_questions": 0,
            "categories": {
                "factual": 0,
                "inferential": 0,
                "conceptual": 0
            }
        }
        
        try:
            # Cache the chunk for efficient access
            self._chunk_cache[chunk.id] = chunk
            
            # Update progress
            if progress_reporter:
                progress_reporter.update_progress(0, 2, {
                    "stage": "question_generation",
                    "chunk_id": chunk.id
                })
            
            # Generate questions
            generated_questions = await self.question_generator.generate_questions(
                chunk=chunk, 
                analysis=analysis
            )
            
            # Handle case where no questions were generated
            if not generated_questions:
                self.logger.warning(f"No questions generated for chunk {chunk.id}")
                return [], stats

            stats["generated_questions"] = len(generated_questions)

            # Update progress
            if progress_reporter:
                progress_reporter.update_progress(1, 2, {
                    "stage": "question_validation",
                    "generated": len(generated_questions)
                })

            # Validate questions - process in smaller batches to optimize memory
            validation_results = {}
            batch_size = 5  # Process 5 questions at a time

            for i in range(0, len(generated_questions), batch_size):
                batch = generated_questions[i:i+batch_size]
                batch_results = await self.validation_engine.validate_questions(
                    questions=batch,
                    chunk=chunk
                )
                validation_results.update(batch_results)

                # Allow garbage collection between batches
                await asyncio.sleep(0)

            # Filter valid questions
            valid_questions = self.validation_engine.get_valid_questions(
                generated_questions,
                validation_results
            )

            # Update statistics
            stats["valid_questions"] = len(valid_questions)
            stats["rejected_questions"] = len(generated_questions) - len(valid_questions)

            # Count by category
            for question in valid_questions:
                category = question.category
                if category in stats["categories"]:
                    stats["categories"][category] += 1
                else:
                    stats["categories"][category] = 1

            # Update progress
            if progress_reporter:
                progress_reporter.update_progress(2, 2, {
                    "stage": "complete",
                    "valid": len(valid_questions),
                    "rejected": stats["rejected_questions"]
                })

            self.logger.info(
                f"Processed chunk {chunk.id}: "
                f"generated={stats['generated_questions']}, "
                f"valid={stats['valid_questions']}, "
                f"rejected={stats['rejected_questions']}"
            )

            # Clear the chunk from cache if not needed anymore
            if chunk.id in self._chunk_cache and len(self._chunk_cache) > 10:
                del self._chunk_cache[chunk.id]

            return valid_questions, stats

        except Exception as e:
            raise ValidationError(f"Failed to process chunk {chunk.id}: {str(e)}")

    
    async def process_chunks(self, 
                           chunks: List[Chunk],
                           analyses: Dict[str, AnalysisResult],
                           progress_reporter: Optional[ProgressReporter] = None) -> Tuple[Dict[str, List[Question]], Dict[str, Any]]:
        """
        Process multiple chunks with optimized memory usage.
        """
        all_questions: Dict[str, List[Question]] = {}
        overall_stats = {
            "total_chunks": len(chunks),
            "total_generated_questions": 0,
            "total_valid_questions": 0,
            "total_rejected_questions": 0,
            "categories": {
                "factual": 0,
                "inferential": 0,
                "conceptual": 0
            },
            "chunk_stats": {}
        }
        
        # Update progress
        if progress_reporter:
            progress_reporter.update_progress(0, len(chunks))
        
        # Process chunks sequentially to maintain validators context
        for i, chunk in enumerate(chunks):
            analysis = analyses.get(chunk.id)
            if not analysis:
                self.logger.warning(f"No analysis found for chunk {chunk.id}, skipping")
                continue
                
            try:
                questions, stats = await self.process_chunk(chunk, analysis, progress_reporter)
                all_questions[chunk.id] = questions
                
                # Update overall statistics
                overall_stats["total_generated_questions"] += stats["generated_questions"]
                overall_stats["total_valid_questions"] += stats["valid_questions"]
                overall_stats["total_rejected_questions"] += stats["rejected_questions"]
                
                for category, count in stats["categories"].items():
                    overall_stats["categories"][category] = overall_stats["categories"].get(category, 0) + count
                    
                overall_stats["chunk_stats"][chunk.id] = stats
                
                # Update progress
                if progress_reporter:
                    progress_reporter.update_progress(i + 1, len(chunks), {
                        "valid_questions": overall_stats["total_valid_questions"],
                        "generated_questions": overall_stats["total_generated_questions"]
                    })
                
                # Force garbage collection periodically to free memory
                if i % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Failed to process chunk {chunk.id}: {str(e)}")
        
        # Final cleanup
        self._chunk_cache.clear()
        gc.collect()
        
        return all_questions, overall_stats
