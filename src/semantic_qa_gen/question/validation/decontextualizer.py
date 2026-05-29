# filename: semantic_qa_gen/question/decontextualizer.py

"""Decontextualization rewrite stage (Stage C, remediation mode).

This is the repair half of the pipeline. The leak filter (Stage D) and the
source-free standalone judge (Stage F) *detect* questions that lean on the
source — a flagged lexical marker ("is mentioned"), or a dangling reference the
judge caught ("the main advantage of this technique" with no antecedent). This
stage *fixes* them, instead of dropping them, by rewriting the pair to name its
subject explicitly.

Why a rewrite, not a smarter generation prompt: making a sentence stand alone is
the *decontextualization* task (Choi 2021; QADECONTEXT, Newman 2023). The
canonical framing is question-generation → question-answering → rewrite, where
QG/QA surface what is missing and the rewrite folds it back in. Here the judge
has already done the hard QG/QA work — its `reason` names the unresolved
reference — so this stage only has to perform the (empirically easy) rewrite,
using the chunk to recover the referent.

Key contract differences from a judge:
  * The rewriter IS given the source. It needs the chunk to resolve
    "this technique" -> "product quantization". This does not contaminate
    anything: the rewriter produces text, it does not score standalone-ness.
    The standalone JUDGE (which must not see the source) re-checks the result.
  * The rewrite must not introduce facts absent from the chunk, must preserve
    the original meaning, category, and difficulty, and must restate the subject
    in the answer so the answer also stands alone.

The component is a pure transformer over Question objects: it mutates `text` and
`answer` in place (preserving id, chunk_id, context, category, and metadata),
records what it did under metadata['decontextualization'], and returns the same
objects. Re-validation is the caller's job (the processor), so this module has
no dependency on the validation engine and the loop control lives in one place.
"""

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, Question
from semantic_qa_gen.llm.router import TaskRouter, LLMTaskService
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError, ValidationError


class Decontextualizer:
    """Rewrites Q/A pairs to stand alone, using the source chunk for reference
    recovery. Stateless across chunks; safe to share one instance."""

    PROMPT_KEY = "decontextualize"

    def __init__(self,
                 config_manager: ConfigManager,
                 task_router: TaskRouter,
                 prompt_manager: PromptManager):
        self.config_manager = config_manager
        self.task_router = task_router
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)

    async def rewrite(self,
                      question: Question,
                      chunk: Chunk,
                      reason: Optional[str] = None,
                      attempt: int = 1) -> bool:
        """Rewrite a single Q/A pair in place so it stands alone.

        Args:
            question: The Question to repair. Mutated in place on success.
            chunk: Source chunk, used to recover the missing referent.
            reason: The judge/filter explanation of what was wrong, passed to the
                rewrite prompt to focus it. Optional but improves results.
            attempt: 1-based attempt number, recorded in metadata.

        Returns:
            True if the pair was rewritten (text and/or answer changed), False if
            the rewrite failed, returned nothing usable, or was a no-op. A False
            return leaves the question unmodified so the caller can drop it.
        """
        original_q, original_a = question.text, question.answer

        # The generation task handler is reused for rewriting; rewriting is a
        # generation-shaped task (free-form text out), not a scoring task. This
        # keeps the rewrite on the stronger remote model rather than the local
        # validation model, which matters for faithful reference resolution.
        try:
            llm_service: LLMTaskService = self.task_router.get_task_handler("generation")
        except LLMServiceError as e:
            self.logger.error(f"Decontextualizer could not get LLM service: {e}")
            return False

        prompt_vars = {
            "chunk_content": chunk.content,
            "question_text": original_q,
            "answer_text": original_a,
            "category": question.category,
            # Reason may be multi-line; collapse so it stays a single prompt field.
            "issue": " ".join((reason or "The question may not stand alone without the source.").split()),
        }

        try:
            formatted = llm_service.prompt_manager.format_prompt(self.PROMPT_KEY, **prompt_vars)
            response_text = await llm_service.adapter.generate_completion(
                prompt=formatted,
                model_config=llm_service.task_model_config,
            )
        except (LLMServiceError, ConfigurationError) as e:
            self.logger.error(f"Decontextualize call failed for Q:{question.id}: {e}")
            return False
        except Exception:
            self.logger.exception(f"Unexpected error during decontextualize call for Q:{question.id}")
            return False

        parsed = self._parse_rewrite(response_text, question.id)
        if not parsed:
            return False

        new_q = (parsed.get("question") or "").strip()
        new_a = (parsed.get("answer") or "").strip()
        if not new_q or not new_a:
            self.logger.warning(f"Decontextualize for Q:{question.id} returned empty question/answer.")
            return False

        if new_q == original_q and new_a == original_a:
            self.logger.debug(f"Decontextualize for Q:{question.id} was a no-op.")
            return False

        # Apply the rewrite in place, preserving identity and provenance.
        question.text = new_q
        question.answer = new_a

        try:
            if not question.metadata:
                question.metadata = {}
            history = question.metadata.get("decontextualization", {})
            attempts = history.get("attempts", [])
            attempts.append({
                "attempt": attempt,
                "reason": reason,
                "original_question": original_q,
                "original_answer": original_a,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            question.metadata["decontextualization"] = {
                "rewritten": True,
                "attempts": attempts,
            }
        except Exception as e:
            self.logger.warning(f"Could not record decontextualization metadata for Q:{question.id}: {e}")

        self.logger.debug(f"Decontextualized Q:{question.id} (attempt {attempt}).")
        return True

    def _parse_rewrite(self, response_text: Optional[str], question_id: str) -> Optional[Dict[str, Any]]:
        """Parse the rewrite response into {'question', 'answer'}.

        Tolerant of bare JSON objects and ```json fenced blocks, mirroring the
        parsing posture used elsewhere in the framework. Returns None on failure.
        """
        if not response_text or not response_text.strip():
            self.logger.warning(f"Empty decontextualize response for Q:{question_id}.")
            return None
        text = response_text.strip()

        data: Optional[Dict[str, Any]] = None
        try:
            if text.startswith("{") and text.endswith("}"):
                data = json.loads(text)
        except json.JSONDecodeError:
            data = None

        if data is None:
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                try:
                    data = json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            self.logger.warning(f"Could not parse decontextualize JSON for Q:{question_id}: {text[:200]}")
            return None
        return data