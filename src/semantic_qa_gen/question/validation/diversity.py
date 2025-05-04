# filename: semantic_qa_gen/question/validation/diversity.py

"""Diversity validator for preventing duplicate or similar questions."""

import re
from typing import Dict, Any, Optional, List, Set, Tuple
from difflib import SequenceMatcher
from collections import defaultdict
import logging

from semantic_qa_gen.document.models import Question, Chunk
# Import new base validator and result model
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.utils.error import ValidationError

# Optional: Use NLP library for better normalization if available
try:
    import nltk
    from nltk.corpus import stopwords
    # Ensure required NLTK data is downloaded (run once)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords', quiet=True)
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    NLTK_AVAILABLE = False
    # Basic stop words fallback
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was',
                  'were', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
                  'of', 'from', 'as', 'what', 'which', 'who', 'whom', 'whose',
                  'how', 'when', 'where', 'why', 'it', 'i', 'you', 'he', 'she', 'we', 'they'}


class DiversityValidator(BaseValidator):
    """
    Validator checking question diversity within a chunk.

    Ensures questions are not too textually similar to previously validated,
    *valid* questions for the same chunk using SequenceMatcher.
    This is a stateful validator, reset per chunk by the ValidationEngine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the diversity validator.

        Args:
            config: Config dict, expected key 'threshold' corresponds to
                    similarity_rejection_threshold in the main schema.
        """
        super().__init__(config)
        # Threshold above which questions are considered TOO similar and thus INVALID
        # Maps to validation.diversity.threshold in schema
        self.similarity_rejection_threshold = self.config.get('threshold', 0.85)
        # State: Stores normalized text of VALID questions per chunk
        # Key: chunk_id, Value: List of tuples (question_id, normalized_text)
        self.valid_questions_cache: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.logger = logging.getLogger(__name__) # Ensure logger capture

    async def validate(self,
                       question: Question,
                       chunk: Chunk,
                       llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate question diversity against others in the same chunk.

        Args:
            question: The Question object to validate.
            chunk: The source Chunk context.
            llm_validation_data: Ignored by this validator.

        Returns:
            A ValidationResult indicating diversity outcome.
        """
        validator_name = self.name
        try:
            normalized_text = self._normalize_text(question.text)
            # Ensure chunk ID exists in cache
            if chunk.id not in self.valid_questions_cache:
                 self.valid_questions_cache[chunk.id] = []

            highest_similarity = 0.0
            most_similar_id = None

            # Compare against already accepted questions for this chunk
            for existing_id, existing_text in self.valid_questions_cache[chunk.id]:
                similarity = SequenceMatcher(None, normalized_text, existing_text).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_id = existing_id

            # Determine validity based on the threshold
            # Question is valid (diverse enough) if similarity is BELOW the rejection threshold
            is_valid = highest_similarity < self.similarity_rejection_threshold

            diversity_score = 1.0 - highest_similarity # Higher score means more diverse

            reasons = []
            suggestions = None
            if is_valid:
                # Add the currently validated question to the cache *for this chunk*
                # so subsequent questions in the same batch can be compared against it.
                self.valid_questions_cache[chunk.id].append((question.id, normalized_text))
                reasons.append(f"Question is sufficiently diverse (max similarity to previous valid questions: {highest_similarity:.2f}).")
            else:
                reasons.append(
                    f"Question is too similar to previous valid question '{most_similar_id}' "
                    f"(similarity: {highest_similarity:.2f} >= threshold: {self.similarity_rejection_threshold})."
                )
                suggestions = "Rephrase the question to focus on a different aspect or use different wording."

            self.logger.debug(
                f"{validator_name} check for Q:{question.id} in chunk {chunk.id}: "
                f"max_similarity={highest_similarity:.2f}, threshold={self.similarity_rejection_threshold}, valid={is_valid}"
            )

            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=is_valid,
                scores={"diversity_score": diversity_score}, # Score representing uniqueness
                reasons=reasons,
                suggested_improvements=suggestions
            )

        except Exception as e:
            self.logger.exception(f"{validator_name}: Error during internal logic for Q:{question.id}: {e}", exc_info=True)
            # Return invalid result indicating internal error
            return ValidationResult(
                question_id=question.id,
                validator_name=validator_name,
                is_valid=False,
                scores={},
                reasons=[f"Internal Validator Error: {str(e)}"]
            )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        text = text.lower()
        # Keep alphanumeric and spaces only
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize and remove stopwords
        if NLTK_AVAILABLE:
            try:
                 words = nltk.word_tokenize(text)
                 words = [w for w in words if w not in STOP_WORDS]
            except Exception as e:
                 self.logger.warning(f"NLTK normalization failed: {e}. Using basic split.")
                 words = [w for w in text.split() if w not in STOP_WORDS] # Fallback split
        else:
            words = [w for w in text.split() if w not in STOP_WORDS]

        return ' '.join(words)

    def reset_for_chunk(self, chunk_id: str) -> None:
        """Reset the cache for a specific chunk_id."""
        if chunk_id in self.valid_questions_cache:
            self.logger.debug(f"Resetting diversity cache for chunk {chunk_id}.")
            del self.valid_questions_cache[chunk_id]

    def reset_all(self) -> None:
        """Reset the cache for all chunks."""
        self.logger.debug("Resetting diversity cache for all chunks.")
        self.valid_questions_cache.clear()

