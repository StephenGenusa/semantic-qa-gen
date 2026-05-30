"""Regression tests for the Phase 2 decontextualization repair stage.

Covers the two units that carry the logic:
  * Decontextualizer.rewrite / _parse_rewrite — parse variants, null handling
    (model declines to rewrite), and no-regression (a failed rewrite leaves the
    pair untouched).
  * QuestionProcessor repair-merge — the keep-better-of-two rule and the routing
    gate (faithfulness failures are never repaired; disabled stage is inert).

These use fakes for the LLM/engine so they run without a model. Run:
    pytest tests/test_decontextualization.py
"""

import asyncio
from types import SimpleNamespace

import pytest

from semantic_qa_gen.document.models import Question, Chunk, AnalysisResult
from semantic_qa_gen.question.processor import QuestionProcessor
from semantic_qa_gen.question.decontextualizer import Decontextualizer


# --------------------------- shared fakes -----------------------------------
class _VR:
    def __init__(self, is_valid, reasons=None):
        self.is_valid = is_valid
        self.reasons = reasons or []


def _agg(is_valid, *, faith=True, complete=True, standalone=None, leak=False):
    indiv = {"factual_accuracy": _VR(faith), "answer_completeness": _VR(complete)}
    if standalone is not None:
        indiv["standalone"] = _VR(standalone)
    if leak:
        indiv["leak_filter"] = _VR(True, ["leak note"])
    return {"is_valid": is_valid, "scores": {}, "reasons": [], "validation_results": indiv}


class _FakeEngine:
    def __init__(self, first, reval):
        self.first, self.reval, self.reval_calls = first, reval, 0

    async def validate_questions(self, questions, chunk):
        return {q.id: self.first[q.id] for q in questions}

    async def revalidate_pair(self, question, chunk):
        self.reval_calls += 1
        return self.reval.get(question.id, _agg(False))


class _FakeGen:
    def __init__(self, qs):
        self.qs = qs

    async def generate_questions(self, chunk, analysis):
        return self.qs


class _FakeDecon:
    def __init__(self, nochange=()):
        self.nochange, self.calls = set(nochange), 0

    async def rewrite(self, question, chunk, reason=None, attempt=1):
        self.calls += 1
        if question.id in self.nochange:
            return False
        question.text = "REWRITTEN: " + question.text
        question.answer = "REWRITTEN: " + question.answer
        return True


def _mk(qid, leak_action=None):
    md = {"validation": {"leak_filter": {"action": leak_action}}} if leak_action else {}
    return Question(id=qid, text=f"q{qid}", answer=f"a{qid}", category="factual", metadata=md)


def _run(qs, first, reval, *, enabled=True, max_attempts=1, nochange=()):
    cm = SimpleNamespace(get_section=lambda n: SimpleNamespace(enabled=enabled, max_attempts=max_attempts))
    eng, dec = _FakeEngine(first, reval), _FakeDecon(nochange)
    p = QuestionProcessor(cm, _FakeGen(qs), eng, dec)
    out = asyncio.new_event_loop().run_until_complete(p.process_chunk(Chunk(content="src"), AnalysisResult()))
    return out, eng, dec


# --------------------------- processor merge --------------------------------
def test_valid_unflagged_kept_without_repair():
    (vq, st), eng, dec = _run([_mk("1")], {"1": _agg(True, standalone=True)}, {})
    assert [q.id for q in vq] == ["1"] and st["repair_attempted"] == 0 and dec.calls == 0


def test_standalone_failure_repaired():
    (vq, st), eng, dec = _run([_mk("2")], {"2": _agg(False, standalone=False)}, {"2": _agg(True, standalone=True)})
    assert len(vq) == 1 and vq[0].text.startswith("REWRITTEN") and st["repair_succeeded"] == 1


def test_flagged_valid_failed_rewrite_keeps_original():
    # No regression: original was valid; rewrite fails revalidation; keep original.
    (vq, st), eng, dec = _run([_mk("3", "flag")], {"3": _agg(True, standalone=True, leak=True)}, {"3": _agg(False)})
    assert len(vq) == 1 and vq[0].text == "q3" and st["repair_failed"] == 1


def test_standalone_failure_failed_rewrite_dropped():
    (vq, st), eng, dec = _run([_mk("4")], {"4": _agg(False, standalone=False)}, {"4": _agg(False)})
    assert len(vq) == 0 and st["repair_failed"] == 1


def test_faithfulness_failure_not_routed():
    (vq, st), eng, dec = _run([_mk("5")], {"5": _agg(False, faith=False, standalone=False)}, {})
    assert len(vq) == 0 and st["repair_attempted"] == 0 and dec.calls == 0


def test_disabled_stage_is_inert():
    (vq, st), eng, dec = _run([_mk("6")], {"6": _agg(False, standalone=False)}, {}, enabled=False)
    assert len(vq) == 0 and st["repair_attempted"] == 0 and dec.calls == 0


# --------------------------- decontextualizer unit --------------------------
def test_parse_variants():
    d = Decontextualizer(None, None, None)
    assert d._parse_rewrite('{"question":"q","answer":"a"}', "x") == {"question": "q", "answer": "a"}
    assert d._parse_rewrite('```json\n{"question":"q","answer":"a"}\n```', "x")["answer"] == "a"
    assert d._parse_rewrite('{"question":null,"answer":null}', "x") == {"question": None, "answer": None}
    assert d._parse_rewrite("not json", "x") is None
    assert d._parse_rewrite("", "x") is None


def _router(resp):
    class _Adapter:
        async def generate_completion(self, prompt, model_config):
            return resp

    class _Svc:
        prompt_manager = SimpleNamespace(format_prompt=lambda self, key, **kw: "P")
        adapter = _Adapter()
        task_model_config = None

    # format_prompt needs to be callable without self-binding weirdness:
    svc = _Svc()
    svc.prompt_manager = SimpleNamespace(format_prompt=lambda key, **kw: "P")
    return SimpleNamespace(get_task_handler=lambda task: svc)


def test_rewrite_null_returns_false_and_leaves_pair_untouched():
    q = Question(id="z", text="What is mentioned about X?", answer="It is Y.", category="factual", metadata={})
    dec = Decontextualizer(None, _router('{"question":null,"answer":null}'), None)
    changed = asyncio.new_event_loop().run_until_complete(dec.rewrite(q, Chunk(content="s"), reason="leak"))
    assert changed is False and q.text == "What is mentioned about X?"


def test_rewrite_success_mutates_and_records_metadata():
    q = Question(id="z2", text="What is mentioned about X?", answer="It is Y.", category="factual", metadata={})
    dec = Decontextualizer(None, _router('{"question":"What is the value of X in system S?","answer":"X is Y."}'), None)
    changed = asyncio.new_event_loop().run_until_complete(dec.rewrite(q, Chunk(content="s"), reason="leak"))
    assert changed is True
    assert q.text.startswith("What is the value")
    assert q.metadata["decontextualization"]["rewritten"] is True