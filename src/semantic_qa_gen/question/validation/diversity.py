"""Diversity validator for preventing duplicate or similar questions."""

import re
from typing import Dict, Any, Optional, List, Set, Tuple
from difflib import SequenceMatcher
from collections import defaultdict

from semantic_qa_gen.document.models import Question, Chunk
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.utils.error import ValidationError


class DiversityValidator(BaseValidator):
    """
    Validator that checks for diversity among generated questions.
    
    This validator ensures that questions are not too similar to each other,
    promoting a diverse set of questions that cover different aspects
    of the source material.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the diversity validator.
        
        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.min_similarity_threshold = self.config.get('min_similarity_threshold', 0.75)
        self.existing_questions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Map of chunk_id -> [(question_id, normalized_question_text)]
    
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """
        Validate that the question is sufficiently different from existing questions.
        
        Args:
            question: The question to validate.
            chunk: The source chunk.
            
        Returns:
            ValidationResult indicating question diversity.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Normalize the question text for comparison
            normalized_text = self._normalize_text(question.text)
            
            # Check similarity against existing questions for this chunk
            most_similar_question = None
            highest_similarity = 0.0
            
            for existing_id, existing_text in self.existing_questions[chunk.id]:
                similarity = self._calculate_similarity(normalized_text, existing_text)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_question = existing_id
            
            # Determine if the question is valid (sufficiently different)
            is_valid = highest_similarity < self.min_similarity_threshold
            
            # If valid, add to the list of existing questions
            if is_valid:
                self.existing_questions[chunk.id].append((question.id, normalized_text))
                
                diversity_score = 1.0 - highest_similarity if most_similar_question else 1.0
                
                validation_result = ValidationResult(
                    question_id=question.id,
                    is_valid=True,
                    scores={"diversity": diversity_score},
                    reasons=[f"Question is sufficiently different (similarity: {highest_similarity:.2f})"]
                )
            else:
                validation_result = ValidationResult(
                    question_id=question.id,
                    is_valid=False,
                    scores={"diversity": 1.0 - highest_similarity},
                    reasons=[f"Question too similar to existing question {most_similar_question} "
                             f"(similarity: {highest_similarity:.2f})"],
                    suggested_improvements="Create a more distinct question focusing on different aspects of the content."
                )
            
            self.logger.debug(
                f"Diversity validation for q:{question.id}: "
                f"similarity={highest_similarity:.2f}, valid={is_valid}"
            )
            
            return validation_result
            
        except Exception as e:
            raise ValidationError(f"Diversity validation failed: {str(e)}")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing punctuation and stopwords.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Simple stopword removal (this could be expanded)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 
                    'were', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 
                    'of', 'from', 'as', 'what', 'which', 'who', 'whom', 'whose', 
                    'how', 'when', 'where', 'why'}
        
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        
        return ' '.join(filtered_words)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    def reset_for_chunk(self, chunk_id: str) -> None:
        """
        Reset the existing questions for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to reset.
        """
        if chunk_id in self.existing_questions:
            del self.existing_questions[chunk_id]
    
    def reset_all(self) -> None:
        """Reset all stored questions."""
        self.existing_questions.clear()
