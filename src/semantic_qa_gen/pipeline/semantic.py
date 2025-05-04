"""Semantic processing pipeline with optimized concurrency."""

import os
import logging
import asyncio
import time
from enum import Enum
import traceback
from typing import Dict, Any, Optional, List, Tuple

from pydantic import BaseModel

# Local project imports
from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.document.models import Document, Chunk, AnalysisResult, Question, Section, SectionType # Added Section/SectionType
from semantic_qa_gen.document.processor import DocumentProcessor
from semantic_qa_gen.chunking.engine import ChunkingEngine
from semantic_qa_gen.chunking.analyzer import SemanticAnalyzer
from semantic_qa_gen.llm.router import TaskRouter
# PromptManager used by components, not directly by pipeline usually
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.question.processor import QuestionProcessor
from semantic_qa_gen.output.formatter import OutputFormatter
from semantic_qa_gen.utils.checkpoint import CheckpointManager, CheckpointError
from semantic_qa_gen.utils.error import (
    SemanticQAGenError, ConfigurationError, DocumentError, AnalyzerError, # Added AnalyzerError
    ChunkingError, LLMServiceError, ValidationError, OutputError
)
from semantic_qa_gen.utils.progress import ProgressReporter, ProcessingStage


class SemanticPipeline:
    """Orchestrates the end-to-end process of generating semantic Q&A pairs."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize the semantic pipeline and its components."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.performance_metrics: Dict[str, Any] = {"start_time": 0.0, "end_time": 0.0, "total_time": 0.0, "stage_times": {}}
        self.config: SemanticQAGenConfig

        try:
            # Access the validated config object (ensures Pydantic validation ran)
            self.config = config_manager.config
        except AttributeError as e:
            self.logger.critical(f"Failed to access validated configuration from manager: {e}", exc_info=True)
            raise ConfigurationError(f"Configuration object not found in ConfigManager: {e}")
        except Exception as e: # Catch other potential errors during config access
            self.logger.critical(f"Unexpected error accessing configuration: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to access configuration: {e}")

        # Initialize components after confirming config is loaded
        try:
            self._initialize_components()
            self.logger.info("SemanticPipeline components initialized successfully.")
        except (ConfigurationError, ImportError) as e: # Catch expected init errors
            self.logger.critical(f"Failed to initialize key pipeline components: {e}", exc_info=True)
            raise # Re-raise critical initialization errors
        except Exception as e: # Catch unexpected init errors
            self.logger.critical(f"Unexpected error initializing pipeline components: {e}", exc_info=True)
            raise SemanticQAGenError(f"Pipeline initialization failed unexpectedly: {e}")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components based on the configuration."""
        self.logger.debug("Initializing pipeline components...")
        self.document_processor = DocumentProcessor(self.config_manager)
        self.chunking_engine = ChunkingEngine(self.config_manager)
        # Create PromptManager instance once - shared if needed
        self.prompt_manager = PromptManager() # TODO: Allow passing prompts_dir from config?
        # Pass PromptManager to TaskRouter
        self.task_router = TaskRouter(self.config_manager, self.prompt_manager)
        # Initialize the SemanticAnalyzer, passing router and prompt manager
        self.semantic_analyzer = SemanticAnalyzer(self.task_router, self.prompt_manager)
        # Initialize generators and validators, passing necessary dependencies
        self.question_generator = QuestionGenerator(self.config_manager, self.task_router, self.prompt_manager)
        self.validation_engine = ValidationEngine(self.config_manager, self.task_router, self.prompt_manager)
        # Initialize the QuestionProcessor with generator and validator engine
        self.question_processor = QuestionProcessor(self.config_manager, self.question_generator, self.validation_engine)
        # CheckpointManager needs the full validated config object
        self.checkpoint_manager = CheckpointManager(self.config)
        # Get progress bar setting from config
        show_progress = getattr(self.config.processing, 'log_level', 'INFO') in ['INFO', 'DEBUG']
        self.progress_reporter = ProgressReporter(show_progress_bar=show_progress)
        self.output_formatter = OutputFormatter(self.config_manager)
        self.logger.debug("Pipeline components initialization complete.")


    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """Execute the full processing pipeline for a single document."""
        self._start_timer("total")
        all_processed_questions: List[Question] = []
        # Initialize stats consistent with _aggregate_stats structure
        overall_stats: Dict[str, Any] = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "failed_analysis_chunks": 0,
            "failed_qa_chunks": 0,
            "total_generated_questions": 0,
            "total_validated_questions": 0,
            "total_valid_questions_final": 0,
            "total_rejected_questions": 0,
            "categories": {}, # Aggregated counts for valid questions
            "chunk_details": {}, # Optional: store per-chunk results if needed later
        }
        document_info: Dict[str, Any] = {}
        checkpoint: Optional[Dict[str, Any]] = None
        start_chunk_index = 0
        checkpoint_interval = self.config.processing.checkpoint_interval
        debug_mode = self.config.processing.debug_mode # Cache for logging

        try:
            # === Stage 1: Load Document & Extract Sections ===
            self._start_timer(ProcessingStage.LOADING.name)
            self.progress_reporter.update_stage(ProcessingStage.LOADING)
            self.logger.info(f"Loading document: {document_path}")
            document = self.document_processor.load_document(document_path)
            document_info = self._get_document_info(document, document_path)
            # Document processor now ensures sections list exists
            sections = document.sections if document.sections else []
            if not sections and document.content.strip():
                 # Handle case where processor/heuristics yield nothing despite content
                 sections = [Section(content=document.content.strip(), section_type=SectionType.PARAGRAPH, level=0)]
                 self.logger.warning(f"Document processor yielded no sections for {document_path}, but content exists. Treating as one section.")
            self.logger.info(f"Document '{document_info.get('title', document.id)}' loaded with {len(sections)} sections.")
            self._end_timer(ProcessingStage.LOADING.name)
            self.progress_reporter.update_progress(1, 1) # Loading is one unit of work

            # === Checkpoint Loading ===
            if self.config.processing.enable_checkpoints:
                try:
                    checkpoint = self.checkpoint_manager.load_checkpoint(document)
                    if checkpoint:
                        # Use count representing chunks *already completed*
                        start_chunk_index = checkpoint.get('processed_chunk_count', 0)
                        # Safely update overall stats, merging category counts
                        loaded_stats = checkpoint.get('statistics', {})
                        overall_stats.update({k:v for k,v in loaded_stats.items() if k != 'categories'}) # Update non-category stats
                        if 'categories' in loaded_stats:
                            for cat, count in loaded_stats['categories'].items():
                                overall_stats['categories'][cat] = overall_stats['categories'].get(cat, 0) + count

                        questions_data = checkpoint.get('processed_qa_pairs_data', [])
                        if questions_data:
                             loaded_qs = self.checkpoint_manager.load_questions_from_data(questions_data)
                             all_processed_questions = loaded_qs
                             self.logger.info(f"Loaded {len(all_processed_questions)} questions from checkpoint.")
                        self.logger.info(f"Resuming from checkpoint. Starting processing at chunk index {start_chunk_index}.")
                    else:
                         self.logger.info("No valid checkpoint found. Processing from beginning.")
                except CheckpointError as e:
                    self.logger.warning(f"Could not load checkpoint: {e}. Processing from beginning.")
            else:
                 self.logger.info("Checkpoints disabled. Processing from beginning.")


            # === Stage 2: Chunk Document ===
            self._start_timer(ProcessingStage.CHUNKING.name)
            self.progress_reporter.update_stage(ProcessingStage.CHUNKING)
            self.logger.info("Chunking document...")
            chunks = self.chunking_engine.chunk_document(document, sections)
            total_chunks = len(chunks)
            overall_stats["total_chunks"] = total_chunks
            self.progress_reporter.update_progress(1, 1) # Chunking is usually fast
            self._end_timer(ProcessingStage.CHUNKING.name)

            if not chunks:
                 self.logger.warning(f"Document {document_path} resulted in 0 chunks.")
                 return self._complete_processing(document_info, all_processed_questions, overall_stats)

            # Filter chunks based on checkpoint if resuming
            chunks_to_process = chunks[start_chunk_index:]
            processed_count_in_run = 0 # How many chunks processed *in this run*

            if not chunks_to_process:
                self.logger.info("All chunks were already processed according to checkpoint.")
                return self._complete_processing(document_info, all_processed_questions, overall_stats)

            num_to_process = len(chunks_to_process)
            self.logger.info(f"Created {total_chunks} total chunks. Processing {num_to_process} chunks (starting from index {start_chunk_index}).")

            # === Stage 3: Analyze Chunks ===
            self._start_timer(ProcessingStage.ANALYSIS.name)
            self.progress_reporter.update_stage(ProcessingStage.ANALYSIS)
            self.logger.info(f"Analyzing {num_to_process} chunks...")
            # Updated to use SemanticAnalyzer instance directly
            analyses = await self._run_analysis_in_batches(chunks_to_process)
             # Check for failed analyses (indicated by default/fallback results)
            failed_analysis_count = sum(1 for res in analyses.values() if res.notes and "failed" in res.notes.lower())
            overall_stats["failed_analysis_chunks"] = failed_analysis_count
            if failed_analysis_count > 0:
                 self.logger.warning(f"{failed_analysis_count} chunk(s) failed during analysis phase.")
            # Progress update now happens inside _run_analysis_in_batches per chunk
            self._end_timer(ProcessingStage.ANALYSIS.name)


            # === Stage 4: Generate & Validate Questions (Chunk by Chunk) ===
            self._start_timer(ProcessingStage.QUESTION_GENERATION.name) # Combined timer for Gen+Val
            self.progress_reporter.update_stage(ProcessingStage.QUESTION_GENERATION)
            self.logger.info(f"Generating and validating questions for {num_to_process} chunks...")

            # Loop through chunks to process one by one
            current_chunk_abs_index = start_chunk_index
            for i, chunk in enumerate(chunks_to_process):
                chunk_id = chunk.id
                self.logger.debug(f"Processing chunk {i+1}/{num_to_process} (Abs Index: {current_chunk_abs_index}, ID: {chunk_id}) for Q&A")

                # Get analysis result for this chunk
                analysis = analyses.get(chunk_id)
                if not analysis or (analysis.notes and "failed" in analysis.notes.lower()):
                     self.logger.warning(f"Analysis result missing or indicates failure for chunk {chunk_id}. Skipping Q&A generation.")
                     processed_count_in_run += 1
                     overall_stats["processed_chunks"] = start_chunk_index + processed_count_in_run
                     overall_stats["failed_qa_chunks"] += 1 # Count failure here too
                     self.progress_reporter.update_progress(processed_count_in_run, num_to_process, {"stage": "qa_skipped", "valid_qs": len(all_processed_questions)})
                     current_chunk_abs_index += 1
                     continue

                # Process using the refactored QuestionProcessor
                try:
                    # QuestionProcessor handles generate -> validate internally for the chunk
                    valid_questions_for_chunk, chunk_stats = await self.question_processor.process_chunk(
                        chunk=chunk,
                        analysis=analysis
                    )
                    # Aggregate results
                    all_processed_questions.extend(valid_questions_for_chunk)
                    self._aggregate_stats(overall_stats, chunk_stats) # Merge chunk stats into overall
                    overall_stats["chunk_details"][chunk_id] = chunk_stats # Store individual chunk results

                    # Log chunk result and handle errors reported by processor
                    if chunk_stats.get("errors"):
                        self.logger.warning(f"Chunk {chunk_id} processed with errors: {chunk_stats['errors']}")
                        overall_stats["failed_qa_chunks"] += 1 # Count as failed if errors occurred

                except Exception as e:
                    # Catch unexpected errors specifically from QuestionProcessor.process_chunk call
                    self.logger.exception(f"Unexpected error from QuestionProcessor for chunk {chunk_id}: {e}", exc_info=debug_mode)
                    overall_stats["failed_qa_chunks"] += 1
                    # Aggregate basic stats even on failure if possible
                    self._aggregate_stats(overall_stats, {"chunk_id": chunk_id, "errors": [f"Processor Crash: {str(e)}"]})
                    # Continue to next chunk

                # Update progress after each chunk
                processed_count_in_run += 1
                overall_stats["processed_chunks"] = start_chunk_index + processed_count_in_run
                self.progress_reporter.update_progress(processed_count_in_run, num_to_process, {
                   "stage": "qa_processed",
                   "valid_qs_total": len(all_processed_questions),
                   "failed_qa_chunks": overall_stats.get("failed_qa_chunks", 0),
                })

                # --- Periodic Checkpoint Saving ---
                total_processed_chunk_count = start_chunk_index + processed_count_in_run
                if self.config.processing.enable_checkpoints and \
                   checkpoint_interval > 0 and \
                   (processed_count_in_run % checkpoint_interval == 0):
                    self.logger.info(f"Checkpoint interval reached ({checkpoint_interval} chunks processed this run). Saving progress...")
                    # Create stats snapshot *before* potentially modifying questions list again
                    current_save_stats = overall_stats.copy()
                    try:
                        # Short pause before writing to potentially avoid race conditions if filesystem is slow
                        await asyncio.sleep(0.05)
                        self.checkpoint_manager.save_checkpoint(
                            document=document,
                            processed_chunk_count=total_processed_chunk_count,
                            stats=current_save_stats,
                            all_questions_so_far=all_processed_questions # Pass the current list
                        )
                    except CheckpointError as cp_err:
                         # Non-fatal, allow pipeline to continue
                         self.logger.error(f"Failed to save intermediate checkpoint: {cp_err}")

                current_chunk_abs_index += 1


            self._end_timer(ProcessingStage.QUESTION_GENERATION.name)

            # === Stage 5: Finalize & Report ===
            if self.config.processing.enable_checkpoints:
                 self.logger.info(f"Saving final checkpoint after processing run...")
                 final_save_stats = overall_stats.copy()
                 try:
                      await asyncio.sleep(0.05) # Small delay before final write
                      self.checkpoint_manager.save_checkpoint(
                           document=document,
                           processed_chunk_count=total_chunks, # Mark all as processed eventually
                           stats=final_save_stats,
                           all_questions_so_far=all_processed_questions
                      )
                 except CheckpointError as cp_err:
                      self.logger.error(f"Failed to save final checkpoint: {cp_err}")

            # --- Close Adapter Connections优雅地 ---
            await self.task_router.close_adapters()

            return self._complete_processing(document_info, all_processed_questions, overall_stats)

        # --- Exception Handling for the Pipeline ---
        except (FileNotFoundError, DocumentError, ChunkingError, AnalyzerError, LLMServiceError, ValidationError, OutputError, CheckpointError, ConfigurationError) as e:
             stage = self.progress_reporter.current_stage if self.progress_reporter else ProcessingStage.LOADING
             self.logger.critical(f"Pipeline failed at stage {stage.name}: {e}", exc_info=debug_mode)
             # Ensure timers are stopped on handled failure
             self._ensure_timers_stopped(stage.name)
             # Close adapters on failure
             await self.task_router.close_adapters()
             raise # Re-raise specific SemanticQAGen related errors
        except Exception as e:
            stage = self.progress_reporter.current_stage if self.progress_reporter else ProcessingStage.LOADING
            self.logger.critical(f"Unexpected critical error in pipeline at stage {stage.name}: {e}", exc_info=True)
            self._ensure_timers_stopped(stage.name)
            await self.task_router.close_adapters()
            raise SemanticQAGenError(f"Pipeline failed unexpectedly: {str(e)}") from e


    async def _run_analysis_in_batches(self, chunks: List[Chunk]) -> Dict[str, AnalysisResult]:
        """Process chunks for analysis in batches using SemanticAnalyzer."""
        if not chunks: return {}

        results: Dict[str, AnalysisResult] = {}
        concurrency = self.config.processing.concurrency
        semaphore = asyncio.Semaphore(concurrency)
        total_chunks_to_analyze = len(chunks)
        analyzed_count = 0
        debug_mode = self.config.processing.debug_mode

        async def analyze_single_chunk_wrapper(chunk: Chunk, index: int) -> Optional[Tuple[str, AnalysisResult]]:
            nonlocal analyzed_count
            try:
                async with semaphore:
                    self.logger.debug(f"Analyzing chunk {index + 1}/{total_chunks_to_analyze} (ID: {chunk.id}) via SemanticAnalyzer")
                    # Call the SemanticAnalyzer instance directly
                    result = await self.semantic_analyzer.analyze_chunk(chunk)
                    # Update progress immediately after successful analysis
                    analyzed_count += 1
                    self.progress_reporter.update_progress(analyzed_count, total_chunks_to_analyze, {"stage": "analysis"})
                    return chunk.id, result
            except (LLMServiceError, ConfigurationError, AnalyzerError) as e: # Catch specific errors from analyzer/LLM
                self.logger.error(f"Failed to analyze chunk {chunk.id}: {str(e)}")
                # Return a default/fallback result associated with the chunk ID
                return chunk.id, self.semantic_analyzer._create_default_analysis_result(chunk.id, f"Analysis failed: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing chunk {chunk.id}: {str(e)}", exc_info=debug_mode)
                # Return a default/fallback result on unexpected errors too
                return chunk.id, self.semantic_analyzer._create_default_analysis_result(chunk.id, f"Unexpected analysis error: {str(e)}")

        tasks = [analyze_single_chunk_wrapper(chunk, i) for i, chunk in enumerate(chunks)]
        try:
            # Use gather to run analysis tasks concurrently
            chunk_results = await asyncio.gather(*tasks) # Exceptions inside tasks are returned in results
        except Exception as e:
            # This catches errors in asyncio.gather itself, unlikely if tasks handle exceptions
            self.logger.error(f"Critical error during batch chunk analysis execution: {e}", exc_info=True)
            raise ChunkingError(f"Batch analysis collection failed critically: {e}")

        failed_analyses = 0
        for result_tuple in chunk_results:
            if result_tuple: # Should always be a tuple now
                chunk_id, analysis_result = result_tuple
                results[chunk_id] = analysis_result
                # Check if it's a fallback result based on notes
                if analysis_result.notes and "failed" in analysis_result.notes.lower():
                    failed_analyses += 1
            else:
                 # Should not happen with current logic, but log if it does
                  self.logger.error("analyze_single_chunk_wrapper returned None unexpectedly.")


        successful_analyses = total_chunks_to_analyze - failed_analyses
        self.logger.info(f"Finished analyzing chunks. Successful: {successful_analyses}, Failed/Fallback: {failed_analyses}")
        if failed_analyses > 0:
             self.logger.warning(f"{failed_analyses} chunks failed analysis or used fallback results.")

        return results

    def _complete_processing(self, doc_info: Dict[str, Any], questions: List[Question], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize processing, compile stats, and report completion."""
        self.logger.info(f"Completing processing for document '{doc_info.get('title', doc_info.get('id'))}'. Final valid question count: {len(questions)}")
        # Mark final stage and stop timers
        self.progress_reporter.update_stage(ProcessingStage.COMPLETE)
        self._ensure_timers_stopped(ProcessingStage.COMPLETE.name) # Ensure all timers ended

        # Compile performance metrics
        performance_data = self._compile_performance_metrics()
        final_stats = stats.copy() # Work on a copy
        final_stats["performance"] = performance_data
        # Update final counts based on the final list of questions
        final_stats["total_valid_questions_final"] = len(questions)
        # Recalculate rejected based on generated vs final valid, if generated count exists
        if "total_generated_questions" in final_stats:
             final_stats["total_rejected_questions"] = final_stats["total_generated_questions"] - final_stats["total_valid_questions_final"]

        # Pass key completion stats to the progress reporter
        completion_stats_summary = {
            "Total Chunks": final_stats.get('total_chunks', 'N/A'),
            "Processed Chunks": final_stats.get('processed_chunks', 'N/A'),
            "Failed QA Chunks": final_stats.get('failed_qa_chunks', 0),
            "Valid Questions": final_stats.get('total_valid_questions_final', 0),
            "Total Time": f"{performance_data.get('total_seconds', 0.0):.2f}s"
        }
        self.progress_reporter.complete(completion_stats_summary)
        self.logger.info("Semantic pipeline processing finished.")

        return self._construct_result(final_stats, questions, doc_info)

    def _construct_result(self, final_stats: Dict[str, Any], questions: List[Question], doc_info: Dict[str, Any]) -> Dict[str, Any]:
         """Helper to construct the final return dictionary."""
         # Clean None values from document info just before returning
         cleaned_doc_info = {k: v for k, v in doc_info.items() if v is not None}
         return {
              "questions": questions, # Return the list of Question objects
              "document": cleaned_doc_info,
              "statistics": final_stats # Return the aggregated stats
         }

    def _aggregate_stats(self, overall_stats: Dict[str, Any], chunk_stats: Optional[Dict[str, Any]]) -> None:
         """Helper to merge chunk statistics into overall pipeline statistics."""
         if not chunk_stats: return

         # Use .get with defaults for safer aggregation
         overall_stats["total_generated_questions"] += chunk_stats.get("generated_questions", 0)
         overall_stats["total_validated_questions"] += chunk_stats.get("validated_questions", 0)
         overall_stats["total_valid_questions_final"] += chunk_stats.get("valid_questions_final", 0)
         overall_stats["total_rejected_questions"] += chunk_stats.get("rejected_questions", 0)
         overall_stats["failed_qa_chunks"] += 1 if chunk_stats.get("errors") else 0

         # Aggregate category counts (only for valid questions)
         for cat, count in chunk_stats.get("categories", {}).items():
              overall_stats["categories"][cat] = overall_stats["categories"].get(cat, 0) + count

         # Store or discard detailed chunk stats based on config/need
         # overall_stats["chunk_details"][chunk_stats.get("chunk_id", "unknown")] = chunk_stats


    # --- Timer Methods (Unchanged) ---
    def _start_timer(self, stage_key: str) -> None:
        """Start timing a processing stage."""
        timestamp = time.monotonic()
        if stage_key == "total":
            if "start_time" not in self.performance_metrics or self.performance_metrics["start_time"] == 0.0:
                 self.performance_metrics["start_time"] = timestamp
        else:
            if stage_key not in self.performance_metrics["stage_times"]:
                 self.performance_metrics["stage_times"][stage_key] = {"count": 0, "total_duration": 0.0}
            # Only start if not already running (might happen with nested calls, though unlikely here)
            if "_current_start" not in self.performance_metrics["stage_times"][stage_key]:
                 self.performance_metrics["stage_times"][stage_key]["_current_start"] = timestamp
        self.logger.debug(f"Timer started for stage: {stage_key}")

        # filename: semantic_qa_gen/pipeline/semantic.py
        # Corrected _end_timer method

    def _end_timer(self, stage_key: str) -> None:
        """End timing a processing stage and record duration."""
        timestamp = time.monotonic()
        duration: float = 0.0
        total_duration_for_stage: float = 0.0

        if stage_key == "total":
            start_time = self.performance_metrics.get("start_time", timestamp)  # Use current time if start missing
            self.performance_metrics["end_time"] = timestamp
            duration = timestamp - start_time
            self.performance_metrics["total_time"] = duration
            total_duration_for_stage = duration
            self.logger.debug(f"Timer ended for stage: {stage_key}. Total duration: {duration:.2f}s")

        elif stage_key in self.performance_metrics["stage_times"]:
            stage_metrics = self.performance_metrics["stage_times"][stage_key]
            # Check if this specific timer invocation was started
            if "_current_start" in stage_metrics:
                start_time = stage_metrics.pop("_current_start")
                duration = timestamp - start_time
                stage_metrics["count"] = stage_metrics.get("count", 0) + 1
                stage_metrics["total_duration"] = stage_metrics.get("total_duration", 0.0) + duration
                total_duration_for_stage = stage_metrics["total_duration"]
                self.logger.debug(
                    f"Timer ended for stage: {stage_key}. Duration: {duration:.2f}s. Total for stage: {total_duration_for_stage:.2f}s")
            # Optional: Handle case where _end_timer is called without a corresponding _start_timer
            # else:
            #    self.logger.debug(f"Timer '{stage_key}' ended but '_current_start' was not found (already ended?).")
            #    pass # Just ignore if timer wasn't running
            else:
                self.logger.warning(f"Attempted to end timer for unknown or uninitialized stage key: '{stage_key}'")
                pass  # Take no action if the stage was never started or is invalid

    def _ensure_timers_stopped(self, current_stage_name: Optional[str]) -> None:
         """Ensure current stage and total timers are stopped, typically on error."""
         # Stop timer for the stage that failed (if provided and running)
         if current_stage_name and current_stage_name in self.performance_metrics.get("stage_times", {}):
             if "_current_start" in self.performance_metrics["stage_times"][current_stage_name]:
                 self._end_timer(current_stage_name)
         # Ensure total timer is stopped if it was started
         if self.performance_metrics.get("start_time") and not self.performance_metrics.get("end_time"):
              self._end_timer("total")

    def _compile_performance_metrics(self) -> Dict[str, Any]:
         """Compile performance metrics into a structured dictionary."""
         total_time = self.performance_metrics.get("total_time", 0.0)
         compiled_metrics: Dict[str, Any] = {"total_seconds": round(total_time, 2)}
         stage_details = {}
         # Iterate through defined ProcessingStage enum for order
         for stage_enum in ProcessingStage:
              stage_name = stage_enum.name
              metrics = self.performance_metrics.get("stage_times", {}).get(stage_name)
              if metrics:
                   stage_details[stage_name.lower()] = {
                        "duration_seconds": round(metrics.get("total_duration", 0.0), 2),
                        "invocations": metrics.get("count", 0)
                   }
         compiled_metrics["stages"] = stage_details
         return compiled_metrics

    def _get_document_info(self, document: Document, path: str) -> Dict[str, Any]:
        """Helper to extract basic document info for the result dict."""
        meta = document.metadata
        doc_type_value = document.doc_type.value if isinstance(document.doc_type, Enum) else str(document.doc_type)

        # Initialize doc_info with guaranteed values
        doc_info: Dict[str, Any] = {
            "id": document.id,
            "path": path,  # Keep the original input path
            "doc_type": doc_type_value  # Ensure doc_type is always present
        }

        # Process metadata if it exists
        meta_dict = {}  # Initialize meta_dict as an empty dictionary
        if meta:
            # Priority 1: Check if it's a Pydantic BaseModel
            if isinstance(meta, BaseModel):
                try:
                    meta_dict = meta.model_dump(exclude_none=True)
                except Exception as dump_err:
                    self.logger.warning(
                        f"Failed Pydantic model_dump on metadata: {dump_err}. Skipping metadata fields.")
                    meta_dict = {}  # Reset on dump error
            # Priority 2: Check if it's already a dictionary
            elif isinstance(meta, dict):
                meta_dict = meta  # Use the dictionary directly
            # Priority 3: Try converting other object types (like dataclass/simple object)
            else:
                try:
                    # vars() is generally preferred over direct __dict__ access
                    meta_dict = vars(meta)
                except TypeError:  # vars() might fail on some types
                    # Fallback to trying __dict__ if vars() fails
                    try:
                        meta_dict = meta.__dict__
                    except AttributeError:
                        # Last resort: If it can't be converted, log a warning and use an empty dict
                        self.logger.warning(
                            f"Metadata object of type {type(meta)} could not be converted to dict. Skipping metadata fields.")
                        meta_dict = {}  # Ensure meta_dict remains empty
                except Exception as e:  # Catch unexpected errors during vars() or __dict__ access
                    self.logger.warning(
                        f"Unexpected error converting metadata of type {type(meta)} to dict: {e}. Skipping metadata fields.")
                    meta_dict = {}  # Ensure meta_dict remains empty

        # Now use the potentially populated meta_dict to update doc_info
        # Add metadata fields only if they have a value
        doc_info.update({
            k: v for k, v in {
                "title": meta_dict.get('title'),
                "source": meta_dict.get('source', path),  # Default source to path if not in meta
                "author": meta_dict.get('author'),
                "date": meta_dict.get('date'),
                "language": meta_dict.get('language'),
                "custom": meta_dict.get('custom'),  # Include custom metadata field
            }.items() if v is not None  # Filter out None values explicitly here
        })

        # Function already has doc_info initialized and updated. Return it.
        return doc_info
