"""Deterministic source-leakage filter (Stage D).

A pure, LLM-free gate that runs BEFORE the validation judges. It catches
phrasings that reference the *source container* ("according to the passage",
"the text states") which can never be legitimate in a stand-alone Q/A pair.

Design rules (deliberate, and load-bearing):

  * DROP only for explicit container-references — phrasings that name the
    source artifact ("passage", "text", "document", "excerpt") or attribute a
    statement to it / its author with a reporting verb ("the author argues").
    These have no legitimate use in a decontextualized question, so they are
    safe to delete outright. High precision is the whole point of this tier.

  * FLAG for ambiguous lexical markers ("is mentioned", "as described") that a
    real question could plausibly contain. Flagged items are routed back
    through decontextualization (Phase 2), never deleted. Flagging is
    non-destructive: it costs a rewrite, not a data point.

  * NO anaphora detection here. Deciding whether a leading "this technique" is a
    dangling reference or a fully-specified noun phrase is a parsing/semantics
    problem, not a string problem, and string matching has a bad false-positive
    profile for it. That judgment belongs to the source-free standalone judge
    (Stage F), which reads the question alone and can see there is no antecedent.

Scope of scanning:
  * The QUESTION is scanned for DROP and FLAG patterns.
  * The ANSWER is scanned for DROP patterns only. An explicit container
    reference is a leak wherever it appears, but the ambiguous lexical markers
    occur far too often in legitimate answers ("the following steps", "X is
    mentioned in clause 4") to justify flagging them there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Pattern, Sequence


class LeakAction(str, Enum):
    """Outcome of a leak check. Ordered by severity: PASS < FLAG < DROP."""
    PASS = "pass"
    FLAG = "flag"
    DROP = "drop"


# --- DROP: explicit references to the source container -----------------------
# A match here names the source itself and therefore cannot belong to a
# stand-alone question. Patterns are word-boundaried and matched case-insensitively.
_DEFAULT_DROP_PATTERNS: List[str] = [
    # "according to the passage / this text / the author / ..."
    r"\baccording to (?:the|this) "
    r"(?:passage|text|document|excerpt|article|author|chapter|source|content|material|reading)\b",

    # "the passage states / this document describes / the article notes / ..."
    # The reporting verb is what makes "section/chapter/article" safe to drop here:
    # it is an attribution to the source, not a generic mention of a section.
    r"\b(?:the|this) "
    r"(?:passage|text|document|excerpt|article|chapter|section|author|source|reading) "
    r"(?:states?|says?|mentions?|describes?|discuss(?:es)?|notes?|explains?|indicates?|"
    r"argues?|claims?|reports?|suggests?|shows?|asserts?|contends?|posits?|observes?|concludes?)\b",

    # "as described above / as mentioned in the passage / as shown below / ..."
    r"\bas (?:mentioned|described|discussed|stated|shown|noted|explained|outlined|defined) "
    r"(?:above|below|earlier|previously|here|"
    r"in (?:the|this) (?:passage|text|document|excerpt|section|chapter|article))\b",

    # "in the passage / in this excerpt" — restricted to unambiguous container words
    # only (passage/text/document/excerpt). "section/chapter/article" WITHOUT a
    # reporting verb are intentionally left to FLAG to avoid false drops on
    # questions that legitimately discuss document structure.
    r"\bin (?:the|this) (?:passage|text|document|excerpt)\b",

    # "the author writes / this author argues" — author + reporting verb only.
    # Bare "the author" is deliberately NOT dropped: it is legitimate in questions
    # about authorship conventions ("how is the author's surname formatted in MLA?").
    r"\b(?:the|this) authors?(?:'s)? "
    r"(?:states?|say[s]?|writes?|argues?|mentions?|describes?|notes?|claims?|"
    r"suggests?|concludes?|asserts?|contends?|posits?|explains?|observes?)\b",

    # "based on the passage / per the text / from this document"
    r"\b(?:based on|per|from) (?:the|this) "
    r"(?:passage|text|document|excerpt|article|reading)\b",
]


# --- FLAG: ambiguous lexical markers (question only) --------------------------
# A legitimate question can contain these, so they are never dropped. They are
# routed back through decontextualization for a single rewrite attempt.
_DEFAULT_FLAG_PATTERNS: List[str] = [
    # "What enzyme is mentioned ..." — the original failure mode. Almost always a
    # leak in a question, but "is mentioned" can be legitimate in some phrasings,
    # so flag rather than drop.
    r"\b(?:is|are|was|were) (?:mentioned|described|discussed|referenced|stated|noted|outlined|defined)\b",

    # bare "as mentioned" / "as described" with no anchor word
    r"\bas (?:mentioned|described|discussed|stated|noted)\b",

    # MCQ-style stems that presuppose unseen options/material
    r"\b(?:which|what|all) of the following\b",
    r"\bthe (?:following|above|below)\b",

    # "given the passage / given the context" — conditional-question phrasing that
    # often (not always) presupposes the source
    r"\bgiven (?:the|this) (?:passage|text|excerpt|context|information|document)\b",
]


@dataclass
class LeakVerdict:
    """Result of a leak check.

    Attributes:
        action: PASS, FLAG, or DROP.
        matches: Human-readable list of matched snippets, prefixed by field
            ("question: 'according to the passage'").
        reason: One-line explanation suitable for a ValidationResult reason.
    """
    action: LeakAction
    matches: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        """True when the pair passed cleanly (no flag, no drop)."""
        return self.action is LeakAction.PASS


class LeakFilter:
    """Deterministic, LLM-free leak gate.

    Construct once and reuse; compiled patterns are cached on the instance.
    Patterns may be overridden for testing or domain tuning, but the defaults
    are intended to be conservative (high-precision DROP, low-noise FLAG).
    """

    def __init__(
        self,
        drop_patterns: Optional[Sequence[str]] = None,
        flag_patterns: Optional[Sequence[str]] = None,
        enabled: bool = True,
        scan_answer_for_drop: bool = True,
    ) -> None:
        self.enabled = enabled
        self.scan_answer_for_drop = scan_answer_for_drop

        drop_src = _DEFAULT_DROP_PATTERNS if drop_patterns is None else list(drop_patterns)
        flag_src = _DEFAULT_FLAG_PATTERNS if flag_patterns is None else list(flag_patterns)

        self._drop: List[Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in drop_src]
        self._flag: List[Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in flag_src]

    def check(self, question_text: str, answer_text: str = "") -> LeakVerdict:
        """Return a LeakVerdict for a single Q/A pair.

        DROP takes precedence over FLAG. The question is checked against both
        tiers; the answer against the DROP tier only (see module docstring).
        """
        if not self.enabled:
            return LeakVerdict(LeakAction.PASS)

        drop_hits: List[str] = []
        flag_hits: List[str] = []

        q = question_text or ""
        for rx in self._drop:
            m = rx.search(q)
            if m:
                drop_hits.append(f"question: '{m.group(0).strip()}'")
        for rx in self._flag:
            m = rx.search(q)
            if m:
                flag_hits.append(f"question: '{m.group(0).strip()}'")

        if self.scan_answer_for_drop and answer_text:
            for rx in self._drop:
                m = rx.search(answer_text)
                if m:
                    drop_hits.append(f"answer: '{m.group(0).strip()}'")

        if drop_hits:
            return LeakVerdict(
                LeakAction.DROP,
                matches=drop_hits,
                reason="Explicit reference to the source container; cannot stand alone.",
            )
        if flag_hits:
            return LeakVerdict(
                LeakAction.FLAG,
                matches=flag_hits,
                reason="Possible source reference; route through decontextualization.",
            )
        return LeakVerdict(LeakAction.PASS)
    