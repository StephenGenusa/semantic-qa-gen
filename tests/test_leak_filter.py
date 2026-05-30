# tests/test_leak_filter.py
"""Golden-set tests for the deterministic leak filter (Stage D).

These encode the precision contract: DROP fires only on explicit source-container
references, FLAG covers the ambiguous lexical middle, and clean / anaphoric
questions PASS untouched (anaphora is the standalone judge's job, not the
filter's).

Run: pytest tests/test_leak_filter.py
"""

import pytest

from semantic_qa_gen.question.filters.leak_filter import LeakFilter, LeakAction

filt = LeakFilter()


# --- DROP: explicit references to the source container -----------------------
@pytest.mark.parametrize("q", [
    "According to the passage, what are the three components of a transformer block?",
    "What does the text say about gradient clipping?",
    "As described above, what is the learning rate schedule?",
    "In this document, which optimizer is used?",
    "The author argues what about overfitting?",
    "Based on the passage, what is the main limitation?",
    "What does this article discuss regarding batch normalization?",
])
def test_drop_container_references(q):
    assert filt.check(q).action is LeakAction.DROP


# --- FLAG: ambiguous lexical markers (the original failure mode, etc.) --------
@pytest.mark.parametrize("q", [
    "What technique is mentioned for grouping the data in each sub-vector into k centroids?",
    "Which enzyme is described as the rate-limiting step?",
    "Which of the following best defines entropy?",
    "Given the passage, what is the time complexity?",
])
def test_flag_ambiguous_markers(q):
    assert filt.check(q).action is LeakAction.FLAG


# --- PASS: clean, named-subject questions must not trip either tier ----------
@pytest.mark.parametrize("q", [
    "In product quantization, what algorithm groups the data in each sub-vector into k centroids?",
    "What are the three components of a transformer block?",
    # 'the author' is legitimate here (authorship convention), so must NOT drop:
    "How is the author's surname formatted in an MLA in-text citation?",
    # Leading demonstratives are NOT handled by the filter (judge's job):
    "What is the purpose of this design pattern in the Observer model?",
    "What happens in this reaction when a catalyst is added?",
])
def test_pass_clean_questions(q):
    assert filt.check(q).action is LeakAction.PASS


# --- Anaphora is intentionally delegated to the standalone judge -------------
def test_anaphora_not_dropped_or_flagged():
    # A dangling "this technique" with no antecedent is a real leak, but detecting
    # it is a semantics problem. The string filter must NOT act on it; the
    # source-free standalone judge will catch it instead.
    assert filt.check("What is the main advantage of this technique?").action is LeakAction.PASS


# --- DROP outranks FLAG ------------------------------------------------------
def test_drop_precedence():
    q = "According to the text, what is mentioned about regularization?"
    assert filt.check(q).action is LeakAction.DROP


# --- Answer scanning: DROP applies to answers; ambiguous markers do not ------
def test_answer_drop_caught():
    v = filt.check("What is the penalty?", "The passage states the penalty is 10%.")
    assert v.action is LeakAction.DROP


def test_answer_ambiguous_not_flagged():
    # "is mentioned" inside an answer is common and often legitimate, so the
    # answer is scanned for DROP patterns only, never FLAG patterns.
    v = filt.check("What is the penalty?", "The penalty is mentioned in clause 4.")
    assert v.action is LeakAction.PASS


# --- The verdict carries the matched snippet for auditing --------------------
def test_verdict_reports_matches():
    v = filt.check("According to the passage, what is X?")
    assert v.action is LeakAction.DROP
    assert v.matches and "question:" in v.matches[0]


# --- Disabled filter passes everything ---------------------------------------
def test_disabled_filter_passes():
    off = LeakFilter(enabled=False)
    assert off.check("According to the passage, what is X?").action is LeakAction.PASS