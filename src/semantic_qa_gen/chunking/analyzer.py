
"""Semantic analyzer for document chunks."""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import time

from semantic_qa_gen.document.models import Chunk, AnalysisResult
from semantic_qa_gen.llm.router import TaskRouter
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ChunkingError, async_with_error_handling
from semantic_qa_gen.utils.progress import ProgressReporter


class SemanticAnalyzer:
    """
    Analyzes document chunks for information density and question potential.
    
    This class is responsible for using LLMs to assess the content of chunks
    and determine their suitability for question generation.
    """
    
    def __init__(self, task_router: TaskRouter, prompt_manager: Optional[PromptManager] = None):
        """
        Initialize the semantic analyzer.
        
        Args:
            task_router: Task router for LLM services.
            prompt_manager: Optional prompt manager.
        """
        self.task_router = task_router
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)
        
        # Cache for analysis results to avoid reprocessing
        self.analysis_cache: Dict[str, Tuple[AnalysisResult, float]] = {}
        self.cache_ttl = 3600  # Cache results for 1 hour
    
    @async_with_error_handling(error_types=Exception, max_retries=2)
    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """
        Analyze a single document chunk.
        
        Args:
            chunk: Document chunk to analyze.
            
        Returns:
            Analysis result for the chunk.
            
        Raises:
            ChunkingError: If analysis fails.
        """
        # Check cache first
        if chunk.id in self.analysis_cache:
            result, timestamp = self.analysis_cache[chunk.id]
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Using cached analysis for chunk {chunk.id}")
                return result
        
        try:
            # Get the LLM service for analysis
            result = await self.task_router.analyze_chunk(chunk)
            
            self.logger.info(
                f"Analyzed chunk {chunk.id}: "
                f"density={result.information_density:.2f}, "
                f"coherence={result.topic_coherence:.2f}, "
                f"complexity={result.complexity:.2f}, "
                f"yield={sum(result.estimated_question_yield.values())} questions"
            )
            
            # Cache the result
            self.analysis_cache[chunk.id] = (result, time.time())
            
            return result
            
        except Exception as e:
            raise ChunkingError(f"Failed to analyze chunk {chunk.id}: {str(e)}")
    
    async def analyze_chunks(self, chunks: List[Chunk], 
                          progress_reporter: Optional[ProgressReporter] = None) -> Dict[str, AnalysisResult]:
        """
        Analyze multiple document chunks with optimized concurrency.
        
        Args:
            chunks: List of document chunks to analyze.
            progress_reporter: Optional progress reporter.
            
        Returns:
            Dictionary mapping chunk IDs to analysis results.
            
        Raises:
            ChunkingError: If analysis fails.
        """
        results = {}
        
        # Set up progress reporting
        if progress_reporter:
            progress_reporter.update_progress(0, len(chunks))
        
        # Process chunks in efficient batches
        concurrency = min(3, len(chunks))  # Limit concurrency to avoid overwhelming LLM APIs
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(chunk):
            async with semaphore:
                return await self.analyze_chunk(chunk)
        
        # Create tasks for all chunks
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(process_with_semaphore(chunk))
            tasks.append((chunk.id, task))
        
        # Process results as they complete
        for i, (chunk_id, task) in enumerate(tasks):
            try:
                result = await task
                results[chunk_id] = result
                
                # Update progress
                if progress_reporter:
                    progress_reporter.update_progress(i + 1, len(chunks), {
                        "analyzed": i + 1,
                        "avg_density": self._calculate_avg_density(results)
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze chunk {chunk_id}: {str(e)}")
                
                # Create a fallback result with neutral values
                results[chunk_id] = AnalysisResult(
                    chunk_id=chunk_id,
                    information_density=0.5,
                    topic_coherence=0.5,
                    complexity=0.5,
                    estimated_question_yield={
                        "factual": 3,
                        "inferential": 2,
                        "conceptual": 1
                    },
                    key_concepts=[],
                    notes="Analysis failed, using default values"
                )
        
        return results
    
    def _calculate_avg_density(self, results: Dict[str, AnalysisResult]) -> float:
        """Calculate the average information density from current results."""
        if not results:
            return 0.0
        return sum(result.information_density for result in results.values()) / len(results)
    
    def estimate_question_yield(self, chunks: List[Chunk], 
                              analyses: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """
        Estimate the total question yield for a set of chunks.
        
        Args:
            chunks: List of document chunks.
            analyses: Analysis results for the chunks.
            
        Returns:
            Statistics about estimated question yield.
        """
        total_yield = {
            "factual": 0,
            "inferential": 0,
            "conceptual": 0
        }
        
        for chunk in chunks:
            analysis = analyses.get(chunk.id)
            if analysis:
                for category, count in analysis.estimated_question_yield.items():
                    if category in total_yield:
                        total_yield[category] += count
        
        total_yield["total"] = sum(total_yield.values())
        
        # Calculate averages for better insight
        avg_values = {}
        if analyses:
            avg_values = {
                "avg_information_density": sum(a.information_density for a in analyses.values()) / len(analyses),
                "avg_topic_coherence": sum(a.topic_coherence for a in analyses.values()) / len(analyses),
                "avg_complexity": sum(a.complexity for a in analyses.values()) / len(analyses),
                "std_dev_density": self._calculate_std_dev([a.information_density for a in analyses.values()])
            }
        
        return {
            "estimated_questions": total_yield,
            "chunks_analyzed": len(analyses),
            **avg_values
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return (variance ** 0.5)

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        self.logger.debug("Analysis cache cleared")
