"""Question generation based on document chunks and analysis."""

import logging
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.llm.router import TaskRouter
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import ValidationError


class QuestionGenerator:
    """
    Generator for creating questions based on document chunks.
    
    This class is responsible for generating diverse questions
    at different cognitive levels based on content analysis.
    """
    
    def __init__(self, config_manager: ConfigManager, 
                task_router: TaskRouter,
                prompt_manager: Optional[PromptManager] = None):
        """
        Initialize the question generator.
        
        Args:
            config_manager: Configuration manager.
            task_router: Task router for LLM calls.
            prompt_manager: Optional prompt manager.
        """
        self.config_manager = config_manager
        self.config = config_manager.get_section("question_generation")
        self.task_router = task_router
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)
    
    async def generate_questions(self, 
                               chunk: Chunk, 
                               analysis: AnalysisResult) -> List[Question]:
        """
        Generate questions for a document chunk based on analysis.
        
        Args:
            chunk: Document chunk to generate questions for.
            analysis: Analysis of the chunk.
            
        Returns:
            List of generated questions.
            
        Raises:
            ValidationError: If question generation fails.
        """
        try:
            # Determine how many questions to generate based on analysis
            category_counts = self._calculate_question_counts(analysis)
            
            self.logger.info(
                f"Generating questions for chunk {chunk.id} with categories: "
                f"{', '.join(f'{k}:{v}' for k, v in category_counts.items())}"
            )
            
            # Generate questions
            questions = await self.task_router.generate_questions(
                chunk=chunk,
                analysis=analysis,
                categories=category_counts
            )
            
            self.logger.info(f"Generated {len(questions)} questions for chunk {chunk.id}")
            
            return questions
            
        except Exception as e:
            raise ValidationError(f"Failed to generate questions for chunk {chunk.id}: {str(e)}")
    
    async def categorize_question(self, question: Question, chunk: Chunk) -> str:
        """
        Determine the cognitive level of a question.
        
        Args:
            question: The question to categorize.
            chunk: The source chunk.
            
        Returns:
            Category name (factual, inferential, or conceptual).
            
        Raises:
            ValidationError: If categorization fails.
        """
        # For now, we'll trust the category provided by the LLM
        # This could be enhanced with a dedicated categorization step
        return question.category
    
    def _calculate_question_counts(self, analysis: AnalysisResult) -> Dict[str, int]:
        """
        Calculate how many questions to generate for each category.
        
        Args:
            analysis: Analysis of the document chunk.
            
        Returns:
            Dictionary mapping categories to question counts.
        """
        # Get configuration for question categories
        category_config = self.config.categories
        max_questions = self.config.max_questions_per_chunk
        
        # Start with the analysis estimates
        counts = analysis.estimated_question_yield.copy()
        
        # Ensure minimum questions per category
        for category, settings in category_config.items():
            min_questions = settings.min_questions
            if category in counts:
                counts[category] = max(counts[category], min_questions)
            else:
                counts[category] = min_questions
        
        # Apply adaptive generation if enabled
        if self.config.adaptive_generation:
            # Scale question counts based on information density
            density_factor = analysis.information_density
            for category in counts:
                scaled_count = int(counts[category] * density_factor * 1.5)
                counts[category] = max(1, scaled_count)
        
        # Ensure we don't exceed the maximum
        total_count = sum(counts.values())
        if total_count > max_questions:
            # Scale down proportionally
            scale_factor = max_questions / total_count
            for category in counts:
                counts[category] = max(1, int(counts[category] * scale_factor))
        
        return counts
