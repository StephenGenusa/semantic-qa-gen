# filename: semantic_qa_gen/question/processor.py

"""Processor for generating and validating questions for a single chunk."""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple

# Project imports
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError
# Removed ProgressReporter import, belongs in pipeline orchestrator


class QuestionProcessor:
    """
    Processes a single document chunk to generate and validate questions.

    This class coordinates the generation of questions using QuestionGenerator
    and their subsequent validation using ValidationEngine for a specific chunk.
    Error handling is included to allow the main pipeline to continue processing
    other chunks even if one fails. Memory optimizations like caching or
    explicit GC are removed in favor of letting the pipeline manage iteration
    and aggregation.
    """

    def __init__(self,
                 config_manager: ConfigManager,
                 question_generator: QuestionGenerator,
                 validation_engine: ValidationEngine):
        """
        Initialize the question processor.

        Args:
            config_manager: Configuration manager instance.
            question_generator: Instance of QuestionGenerator.
            validation_engine: Instance of ValidationEngine.
        """
        self.config_manager = config_manager
        # Config sections accessed by generator/validator internally as needed
        self.question_generator = question_generator
        self.validation_engine = validation_engine
        self.logger = logging.getLogger(__name__)
        # Removed chunk cache and GC calls

    async def process_chunk(self,
                            chunk: Chunk,
                            analysis: AnalysisResult) -> Tuple[List[Question], Dict[str, Any]]:
        """
        Process a single document chunk to generate and validate questions.

        Handles the flow: Generate Questions -> Validate Questions -> Calculate Stats.
        Catches errors during generation or validation for robustness.

        Args:
            chunk: The document chunk to process.
            analysis: The pre-computed analysis result for the chunk.

        Returns:
            A tuple containing:
            - list[Question]: A list of validated questions generated for this chunk.
            - dict[str, Any]: Statistics specific to the processing of this chunk.
                              (e.g., generated, valid, rejected counts).
        """
        chunk_stats = {
            "chunk_id": chunk.id,
            "generated_questions": 0,
            "validated_questions": 0,  # Renamed from valid_questions for clarity
            "valid_questions_final": 0,  # Final count after filtering
            "rejected_questions": 0,
            "errors": [],
            "categories": {  # Count valid questions by category
                "factual": 0,
                "inferential": 0,
                "conceptual": 0
            }
            # Add status: 'success', 'generation_failed', 'validation_failed'?
        }
        generated_questions: List[Question] = []
        valid_questions: List[Question] = []

        try:
            # === Step 1: Generate Questions ===
            self.logger.debug(f"Starting question generation for chunk {chunk.id}")
            try:
                # Passing the chunk and analysis directly to enable metadata enhancement
                generated_questions = await self.question_generator.generate_questions(
                    chunk=chunk,
                    analysis=analysis
                )
                chunk_stats["generated_questions"] = len(generated_questions)
                self.logger.info(f"Generated {len(generated_questions)} questions for chunk {chunk.id}.")
            except (ValidationError, LLMServiceError) as e:
                self.logger.error(f"Question generation failed for chunk {chunk.id}: {e}")
                chunk_stats["errors"].append(f"Generation Error: {str(e)}")
                # Return empty list and stats indicating failure at this stage
                return [], chunk_stats  # Stop processing this chunk if generation fails critically
            except Exception as e:
                self.logger.exception(f"Unexpected error during question generation for chunk {chunk.id}",
                                      exc_info=True)
                chunk_stats["errors"].append(f"Unexpected Generation Error: {str(e)}")
                return [], chunk_stats  # Stop processing

            # === Step 2: Validate Questions (if any were generated) ===
            if generated_questions:
                self.logger.debug(f"Starting validation for {len(generated_questions)} questions (Chunk {chunk.id}).")
                validation_results: Dict[str, Dict[str, Any]] = {}
                try:
                    # Use the efficient batch validation method from ValidationEngine
                    validation_results = await self.validation_engine.validate_questions(
                        questions=generated_questions,
                        chunk=chunk
                    )
                    chunk_stats["validated_questions"] = len(
                        generated_questions)  # Count how many went through validation

                    # Filter valid questions based on the results
                    # This helper method in ValidationEngine checks the aggregated 'is_valid' flag
                    valid_questions = self.validation_engine.get_valid_questions(
                        generated_questions,
                        validation_results
                    )
                    chunk_stats["valid_questions_final"] = len(valid_questions)
                    chunk_stats["rejected_questions"] = chunk_stats["generated_questions"] - chunk_stats[
                        "valid_questions_final"]

                    self.logger.info(
                        f"Validation complete for chunk {chunk.id}: "
                        f"{chunk_stats['valid_questions_final']} valid, "
                        f"{chunk_stats['rejected_questions']} rejected."
                    )

                except (ValidationError, LLMServiceError) as e:
                    # Errors during the validation *process* (e.g., LLM call failure)
                    self.logger.error(f"Question validation failed for chunk {chunk.id}: {e}")
                    chunk_stats["errors"].append(f"Validation Error: {str(e)}")
                    # Decide if we return the originally generated (but unvalidated) questions or none
                    # Safer to return none if validation process fails.
                    valid_questions = []
                    chunk_stats["validated_questions"] = 0
                    chunk_stats["valid_questions_final"] = 0
                    chunk_stats["rejected_questions"] = chunk_stats[
                        "generated_questions"]  # All rejected due to validation failure
                except Exception as e:
                    self.logger.exception(f"Unexpected error during question validation for chunk {chunk.id}",
                                          exc_info=True)
                    chunk_stats["errors"].append(f"Unexpected Validation Error: {str(e)}")
                    valid_questions = []  # Safer default
                    chunk_stats["validated_questions"] = 0
                    chunk_stats["valid_questions_final"] = 0
                    chunk_stats["rejected_questions"] = chunk_stats["generated_questions"]

            # === Step 3: Update Category Stats for Valid Questions ===
            if valid_questions:
                for question in valid_questions:
                    category = question.category
                    if category in chunk_stats["categories"]:
                        chunk_stats["categories"][category] += 1
                    else:
                        # Log if unexpected category encountered
                        self.logger.warning(
                            f"Encountered unexpected question category '{category}' in chunk {chunk.id}. Adding to stats.")
                        chunk_stats["categories"][category] = 1

            return valid_questions, chunk_stats

        except Exception as e:
            # Catch-all for any truly unexpected error within this method's orchestration
            self.logger.exception(f"Critical unexpected error processing chunk {chunk.id}", exc_info=True)
            chunk_stats["errors"].append(f"Critical Processor Error: {str(e)}")
            return [], chunk_stats  # Return empty results and error state