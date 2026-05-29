# filename: semantic_qa_gen/question/processor.py

"""Processor for generating, validating, and repairing questions for a chunk.

Phase 2 adds a repair stage between validation and the final verdict:

    generate -> validate -> classify repair candidates -> decontextualize+revalidate
             -> keep-better-of-two merge -> stats

Repair routing (set by design discussion):
  * A question is a repair candidate when its faithfulness passed AND it was
    either leak-FLAGged or failed the source-free standalone judge.
  * Faithfulness FAILURES are never routed to repair — if the answer is not
    grounded, rewriting the wording cannot fix it. Those are dropped.
  * Leak DROPs are not routed (an explicit source reference, discarded upstream).

Merge rule — keep the better of two, never regress:
  * If a rewrite re-validates clean, it replaces the original.
  * Else, if the original was already valid (a FLAGged-but-passing pair), the
    original is kept.
  * Else the question is dropped.

Diversity is a set-level property; it is decided on the first pass and not re-run
per rewrite (rewrites replace originals in place, preserving the set). Cross-
rewrite dedup is a later (Phase 5) concern.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.question.generator import QuestionGenerator
from semantic_qa_gen.question.validation.engine import ValidationEngine
from semantic_qa_gen.utils.error import ValidationError, LLMServiceError


class QuestionProcessor:
    """Generates, validates, and (optionally) repairs questions for one chunk."""

    def __init__(self,
                 config_manager: ConfigManager,
                 question_generator: QuestionGenerator,
                 validation_engine: ValidationEngine,
                 decontextualizer: Optional[Any] = None):
        """
        Args:
            decontextualizer: Optional Decontextualizer. If None, or if the
                'decontextualization' config section is disabled, the repair stage
                is skipped and behavior matches Phase 0/1.
        """
        self.config_manager = config_manager
        self.question_generator = question_generator
        self.validation_engine = validation_engine
        self.decontextualizer = decontextualizer
        self.logger = logging.getLogger(__name__)

        # Resolve repair settings defensively (works with Pydantic model or dict,
        # and tolerates the section being absent entirely).
        self._repair_enabled = False
        self._max_attempts = 1
        try:
            section = config_manager.get_section("decontextualization")
        except Exception:
            section = None
        if section is not None and decontextualizer is not None:
            self._repair_enabled = bool(getattr(section, "enabled", False)
                                        if not isinstance(section, dict)
                                        else section.get("enabled", False))
            raw_attempts = (getattr(section, "max_attempts", 1)
                            if not isinstance(section, dict)
                            else section.get("max_attempts", 1))
            try:
                self._max_attempts = max(1, int(raw_attempts))
            except (TypeError, ValueError):
                self._max_attempts = 1

    async def process_chunk(self, chunk: Chunk,
                            analysis: AnalysisResult) -> Tuple[List[Question], Dict[str, Any]]:
        chunk_stats = {
            "chunk_id": chunk.id,
            "generated_questions": 0,
            "validated_questions": 0,
            "valid_questions_final": 0,
            "rejected_questions": 0,
            "repair_attempted": 0,
            "repair_succeeded": 0,
            "repair_failed": 0,
            # Per-dimension tally of WHY questions were dropped (Phase 4 funnel seed).
            "rejected_by": {
                "faithfulness": 0,   # answer not grounded (not repairable by rewrite)
                "standalone": 0,     # failed source-free judge
                "diversity": 0,      # near-duplicate within the chunk
                "leak_drop": 0,      # explicit source reference, dropped upstream
                "repair_failed": 0,  # was a repair candidate; no rewrite re-validated
                "other": 0,          # invalid for some other/combined reason
            },
            "errors": [],
            "categories": {"factual": 0, "inferential": 0, "conceptual": 0},
        }
        generated: List[Question] = []

        try:
            # === Step 1: Generate ===
            try:
                generated = await self.question_generator.generate_questions(chunk=chunk, analysis=analysis)
                chunk_stats["generated_questions"] = len(generated)
                self.logger.info(f"Generated {len(generated)} questions for chunk {chunk.id}.")
            except (ValidationError, LLMServiceError) as e:
                self.logger.error(f"Question generation failed for chunk {chunk.id}: {e}")
                chunk_stats["errors"].append(f"Generation Error: {str(e)}")
                return [], chunk_stats
            except Exception as e:
                self.logger.exception(f"Unexpected generation error for chunk {chunk.id}")
                chunk_stats["errors"].append(f"Unexpected Generation Error: {str(e)}")
                return [], chunk_stats

            if not generated:
                return [], chunk_stats

            # === Step 2: Validate (first pass, includes diversity) ===
            try:
                results = await self.validation_engine.validate_questions(questions=generated, chunk=chunk)
                chunk_stats["validated_questions"] = len(generated)
            except (ValidationError, LLMServiceError) as e:
                self.logger.error(f"Validation failed for chunk {chunk.id}: {e}")
                chunk_stats["errors"].append(f"Validation Error: {str(e)}")
                chunk_stats["rejected_questions"] = len(generated)
                return [], chunk_stats
            except Exception as e:
                self.logger.exception(f"Unexpected validation error for chunk {chunk.id}")
                chunk_stats["errors"].append(f"Unexpected Validation Error: {str(e)}")
                chunk_stats["rejected_questions"] = len(generated)
                return [], chunk_stats

            # === Step 3: Repair stage (optional) + merge ===
            final_valid: List[Question] = []
            for q in generated:
                agg = results.get(q.id, {}) or {}
                original_valid = bool(agg.get("is_valid", False))

                if self._repair_enabled and self._is_repair_candidate(q, agg):
                    chunk_stats["repair_attempted"] += 1
                    repaired = await self._attempt_repair(q, chunk, agg)
                    if repaired is not None:
                        chunk_stats["repair_succeeded"] += 1
                        final_valid.append(repaired)
                    elif original_valid:
                        # Rewrite didn't help, but the original already passed.
                        chunk_stats["repair_failed"] += 1
                        final_valid.append(q)
                    else:
                        chunk_stats["repair_failed"] += 1
                        chunk_stats["rejected_by"]["repair_failed"] += 1
                        self._log_rejection(q, agg, chunk.id, repair="attempted, no clean rewrite")
                else:
                    if original_valid:
                        final_valid.append(q)
                    else:
                        self._tally_rejection(q, agg, chunk_stats)
                        self._log_rejection(q, agg, chunk.id, repair="not a repair candidate")

            chunk_stats["valid_questions_final"] = len(final_valid)
            chunk_stats["rejected_questions"] = chunk_stats["generated_questions"] - len(final_valid)

            # === Step 4: Category stats ===
            for q in final_valid:
                cat = q.category
                chunk_stats["categories"][cat] = chunk_stats["categories"].get(cat, 0) + 1

            rb = chunk_stats["rejected_by"]
            rb_summary = ", ".join(f"{k}={v}" for k, v in rb.items() if v) or "none"
            self.logger.info(
                f"Chunk {chunk.id}: {len(final_valid)} valid "
                f"(repaired {chunk_stats['repair_succeeded']}/"
                f"{chunk_stats['repair_attempted']} attempted), "
                f"{chunk_stats['rejected_questions']} rejected [{rb_summary}]."
            )
            return final_valid, chunk_stats

        except Exception as e:
            self.logger.exception(f"Critical error processing chunk {chunk.id}")
            chunk_stats["errors"].append(f"Critical Processor Error: {str(e)}")
            return [], chunk_stats

    def _is_repair_candidate(self, question: Question, agg: Dict[str, Any]) -> bool:
        """Repairable iff faithfulness passed AND (leak-FLAGged OR standalone failed)."""
        indiv = agg.get("validation_results", {}) or {}

        def passed(name: str) -> bool:
            r = indiv.get(name)
            return bool(getattr(r, "is_valid", False))

        # Faithfulness gate: only repair pairs whose facts are sound. If the
        # faithfulness validators didn't run (e.g. a leak DROP short-circuited),
        # they read as not-passed, so DROPs are correctly excluded.
        faithfulness_ok = passed("factual_accuracy") and passed("answer_completeness")
        if not faithfulness_ok:
            return False

        standalone_failed = ("standalone" in indiv) and (not passed("standalone"))

        leak_meta = ((question.metadata or {}).get("validation", {}) or {}).get("leak_filter", {}) or {}
        flagged = leak_meta.get("action") == "flag"

        return standalone_failed or flagged

    async def _attempt_repair(self, question: Question, chunk: Chunk,
                              agg: Dict[str, Any]) -> Optional[Question]:
        """Rewrite + re-validate, up to max_attempts. Returns a clean rewritten
        Question, or None if no attempt re-validated successfully.

        The Decontextualizer mutates a copy in place; we re-validate that copy via
        the engine (leak + faithfulness + standalone, no diversity). Only a copy
        that re-validates clean is returned, so the original is never lost on a
        failed attempt.
        """
        reason = self._build_reason(agg)
        for attempt in range(1, self._max_attempts + 1):
            candidate = question.model_copy(deep=True)
            try:
                changed = await self.decontextualizer.rewrite(candidate, chunk, reason=reason, attempt=attempt)
            except Exception:
                self.logger.exception(f"Decontextualize rewrite raised for Q:{question.id}")
                return None
            if not changed:
                return None
            try:
                re_agg = await self.validation_engine.revalidate_pair(candidate, chunk)
            except Exception:
                self.logger.exception(f"Re-validation raised for rewritten Q:{question.id}")
                return None
            if re_agg.get("is_valid", False):
                return candidate
            # Feed the new failure reason into the next attempt, if any remain.
            reason = self._build_reason(re_agg) or reason
        return None

    @staticmethod
    def _build_reason(agg: Dict[str, Any]) -> str:
        """Assemble a focused hint for the rewrite from the standalone reason and
        any recorded leak matches in the aggregated result."""
        indiv = agg.get("validation_results", {}) or {}
        bits: List[str] = []
        standalone = indiv.get("standalone")
        if standalone is not None and not getattr(standalone, "is_valid", True):
            bits.extend(getattr(standalone, "reasons", []) or [])
        leak = indiv.get("leak_filter")
        if leak is not None:
            bits.extend(getattr(leak, "reasons", []) or [])
        return " ".join(bits).strip()

    @staticmethod
    def _failed_dimensions(agg: Dict[str, Any]) -> List[str]:
        """Names of the validator dimensions that marked this question invalid."""
        indiv = agg.get("validation_results", {}) or {}
        failed = []
        for name, res in indiv.items():
            if not getattr(res, "is_valid", True):
                failed.append(name)
        return failed

    def _tally_rejection(self, question: Question, agg: Dict[str, Any],
                         chunk_stats: Dict[str, Any]) -> None:
        """Increment the per-dimension rejection counters for a dropped question.
        A question can fail several dimensions; each failing dimension is counted,
        so the buckets sum to >= the number of rejected questions."""
        failed = self._failed_dimensions(agg)
        rb = chunk_stats["rejected_by"]
        mapped = False
        if "factual_accuracy" in failed or "answer_completeness" in failed:
            rb["faithfulness"] += 1; mapped = True
        if "standalone" in failed:
            rb["standalone"] += 1; mapped = True
        if "diversity" in failed:
            rb["diversity"] += 1; mapped = True
        if "leak_filter" in failed:
            # An invalid leak_filter result is the DROP action (FLAG is is_valid=True).
            rb["leak_drop"] += 1; mapped = True
        if not mapped:
            rb["other"] += 1

    def _log_rejection(self, question: Question, agg: Dict[str, Any],
                       chunk_id: str, repair: str) -> None:
        """Emit one INFO line per dropped question naming the failing dimension(s),
        the leak action, and the repair disposition. This is the visibility that
        turns 'N rejected' into 'why those N were dropped'."""
        failed = self._failed_dimensions(agg) or ["unknown"]
        leak_meta = ((question.metadata or {}).get("validation", {}) or {}).get("leak_filter", {}) or {}
        leak_action = leak_meta.get("action", "none")
        # Pull the standalone reason if present, for the most actionable detail.
        indiv = agg.get("validation_results", {}) or {}
        detail = ""
        sa = indiv.get("standalone")
        if sa is not None and not getattr(sa, "is_valid", True):
            reasons = getattr(sa, "reasons", []) or []
            if reasons:
                detail = f" | standalone: {reasons[0][:160]}"
        q_preview = (question.text or "")[:80]
        self.logger.info(
            f"REJECTED Q:{question.id} (chunk {chunk_id}) "
            f"failed={failed} leak={leak_action} repair={repair}{detail} | q=\"{q_preview}\""
        )