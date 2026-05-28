# src/semantic_qa_gen/llm/prompts/manager.py
"""Prompt management system for SemanticQAGen."""

import os
import yaml
import logging
import string
from typing import Dict, Any, Optional, List, Set

from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError


class PromptTemplate:
    """
    Template for LLM prompts with variable substitution.

    Prompt templates allow for consistent prompt formatting with
    dynamic content insertion and metadata management.
    """

    def __init__(self, template: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt template.

        Args:
            template: Template string with {variable} placeholders.
            metadata: Optional metadata about the prompt.
        """
        self.template = template
        self.metadata = metadata or {}

    def format(self, **kwargs) -> str:
        """
        Format the template by substituting variables.

        Args:
            **kwargs: Variables to substitute.

        Returns:
            Formatted prompt string.

        Raises:
            KeyError: If a required variable is missing.
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise KeyError(f"Missing required variable in prompt template: {missing_var}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {str(e)}")


class PromptManager:
    """
    Manager for organizing and retrieving prompt templates.

    This class handles loading prompt templates from files and
    providing them on demand for various LLM tasks.
    """

    # Define essential prompt keys that the system requires
    ESSENTIAL_PROMPTS = {
        "chunk_analysis",
        "question_generation",
        "question_validation",
    }

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Optional directory for loading prompts.
        """
        self.prompts: Dict[str, PromptTemplate] = {}
        self.logger = logging.getLogger(__name__)

        # Default prompts directory within the package
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )

        # Track which prompts were loaded from files
        self.loaded_from_files: Set[str] = set()

        # Register built-in prompts
        self._register_builtin_prompts()

        # Verify essential prompts
        self._verify_essential_prompts()

    def _register_builtin_prompts(self) -> None:
        """Register built-in prompt templates."""
        # Load from YAML files if directory exists
        loaded_any = False
        missing_essential = set()

        if os.path.exists(self.prompts_dir):
            for filename in os.listdir(self.prompts_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    try:
                        path = os.path.join(self.prompts_dir, filename)
                        keys_loaded = self._load_from_file(path)
                        if keys_loaded:
                            loaded_any = True
                            self.logger.info(f"Loaded {len(keys_loaded)} prompt templates from {filename}")

                            # Check if we loaded essential prompts
                            for key in self.ESSENTIAL_PROMPTS:
                                if key in keys_loaded:
                                    self.logger.info(f"Loaded essential prompt '{key}' from {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to load prompt from {filename}: {str(e)}")

        # Register fallback prompts if none were loaded
        if not loaded_any:
            self.logger.warning("No prompt templates loaded from files. Using fallback prompts.")
            self._register_fallback_prompts()
        else:
            # Check for missing essential prompts
            missing_essential = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
            if missing_essential:
                self.logger.warning(f"Missing essential prompts after loading files: {missing_essential}")
                self._register_selected_fallbacks(missing_essential)

    def _load_from_file(self, path: str) -> Set[str]:
        """
        Load prompt templates from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Set of keys loaded from this file.

        Raises:
            ConfigurationError: If the file contains invalid prompt definitions.
        """
        loaded_keys = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)

            if not isinstance(prompt_data, dict):
                raise ConfigurationError(f"Invalid prompt file format: {path}. Expected a dictionary.")

            for name, data in prompt_data.items():
                if not isinstance(data, dict):
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: Not a dictionary")
                    continue

                if 'template' not in data:
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: Missing 'template' field")
                    continue

                template = data.pop('template')
                if not isinstance(template, str):
                    self.logger.error(f"Invalid prompt definition for {name} in {path}: 'template' is not a string")
                    continue

                metadata = data

                # Add file source to metadata
                metadata['source'] = os.path.basename(path)

                # Log template preview for debugging
                template_preview = template[:100] + '...' if len(template) > 100 else template
                self.logger.debug(f"Loading prompt '{name}' from {path}: {template_preview}")

                # Check for essential prompt
                if name in self.ESSENTIAL_PROMPTS:
                    if name == "question_generation" and "analyze" in template.lower()[:100]:
                        self.logger.warning(
                            f"WARNING: The prompt '{name}' in {path} appears to be an analysis prompt, "
                            f"not a question generation prompt. This may cause failures."
                        )

                self.register_prompt(name, template, metadata)
                loaded_keys.add(name)
                self.loaded_from_files.add(name)

            return loaded_keys

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load prompts from {path}: {str(e)}")

    def _register_selected_fallbacks(self, missing_keys: Set[str]) -> None:
        """Register only the specified fallback prompts.

        Note: This now also covers 'question_validation'. The original
        implementation registered fallbacks only for 'chunk_analysis' and
        'question_generation', which meant a YAML directory missing only the
        validation prompt would silently lose validation at runtime.
        """
        if "chunk_analysis" in missing_keys:
            self.logger.info("Registering fallback chunk_analysis prompt")
            self.register_prompt(
                "chunk_analysis",
                _CHUNK_ANALYSIS_TEMPLATE,
                {
                    "description": "Analyzes a text chunk for information density and question potential",
                    "json_output": True,
                    "system_prompt": _CHUNK_ANALYSIS_SYSTEM_PROMPT,
                },
            )

        if "question_generation" in missing_keys:
            self.logger.info("Registering fallback question_generation prompt")
            self.register_prompt(
                "question_generation",
                _QUESTION_GENERATION_TEMPLATE,
                {
                    "description": "Generates source-grounded question/answer pairs from a text chunk",
                    "json_output": True,
                    "system_prompt": _QUESTION_GENERATION_SYSTEM_PROMPT,
                },
            )

        if "question_validation" in missing_keys:
            self.logger.info("Registering fallback question_validation prompt")
            self.register_prompt(
                "question_validation",
                _QUESTION_VALIDATION_TEMPLATE,
                {
                    "description": "Validates a question-answer pair against the source text",
                    "json_output": True,
                    "system_prompt": _QUESTION_VALIDATION_SYSTEM_PROMPT,
                },
            )

    def _register_fallback_prompts(self) -> None:
        """Register fallback prompts if no prompts were loaded from files."""
        self.register_prompt(
            "chunk_analysis",
            _CHUNK_ANALYSIS_TEMPLATE,
            {
                "description": "Analyzes a text chunk for information density and question potential",
                "json_output": True,
                "system_prompt": _CHUNK_ANALYSIS_SYSTEM_PROMPT,
            },
        )

        self.register_prompt(
            "question_generation",
            _QUESTION_GENERATION_TEMPLATE,
            {
                "description": "Generates source-grounded question/answer pairs from a text chunk",
                "json_output": True,
                "system_prompt": _QUESTION_GENERATION_SYSTEM_PROMPT,
            },
        )

        self.register_prompt(
            "question_validation",
            _QUESTION_VALIDATION_TEMPLATE,
            {
                "description": "Validates a question-answer pair against the source text",
                "json_output": True,
                "system_prompt": _QUESTION_VALIDATION_SYSTEM_PROMPT,
            },
        )

    def _verify_essential_prompts(self) -> None:
        """Verify that all essential prompts are available."""
        missing = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
        if missing:
            error_msg = f"Critical prompt templates missing: {', '.join(missing)}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)

        # Verify question_generation is actually generating questions, not analysis
        if "question_generation" in self.prompts:
            template = self.prompts["question_generation"].template.lower()
            if "analyze" in template[:100] and "density" in template and "coherence" in template:
                warning = (
                    "WARNING: The 'question_generation' prompt appears to be an analysis prompt, "
                    "not a question generation prompt. This will cause generation failures."
                )
                self.logger.error(warning)
                # Consider raising an error here if you want to fail fast

    def register_prompt(self, name: str, template: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new prompt template.

        Args:
            name: Prompt name/identifier.
            template: Template string.
            metadata: Optional metadata.
        """
        self.prompts[name] = PromptTemplate(template, metadata)
        self.logger.info(f"Registered prompt template: {name}")

    def get_prompt(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.

        Args:
            name: Prompt name/identifier.

        Returns:
            Prompt template.

        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        if name not in self.prompts:
            raise LLMServiceError(f"Prompt template not found: {name}")

        return self.prompts[name]

    def format_prompt(self, name: str, **kwargs) -> str:
        """
        Format a prompt template with variable substitution.

        Args:
            name: Prompt name/identifier.
            **kwargs: Variables to substitute.

        Returns:
            Formatted prompt string.

        Raises:
            LLMServiceError: If the prompt doesn't exist or formatting fails.
        """
        try:
            template = self.get_prompt(name)
            return template.format(**kwargs)
        except KeyError as e:
            raise LLMServiceError(f"Missing variable in prompt template {name}: {str(e)}")
        except Exception as e:
            raise LLMServiceError(f"Error formatting prompt {name}: {str(e)}")

    def get_system_prompt(self, name: str) -> Optional[str]:
        """
        Get the system prompt for a template, if defined.

        Args:
            name: Prompt name/identifier.

        Returns:
            System prompt string if defined, otherwise None.

        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        template = self.get_prompt(name)
        return template.metadata.get('system_prompt')

    def is_json_output(self, name: str) -> bool:
        """
        Check if a prompt expects JSON output.

        Args:
            name: Prompt name/identifier.

        Returns:
            True if the prompt expects JSON output.

        Raises:
            LLMServiceError: If the prompt doesn't exist.
        """
        template = self.get_prompt(name)
        return template.metadata.get('json_output', False)


# ---------------------------------------------------------------------------
# Fallback prompt templates
#
# These are the prompts used when no YAML templates are present in
# self.prompts_dir, or when a YAML file is missing one of the essential
# prompts. They are intentionally module-level constants so that
# _register_fallback_prompts() and _register_selected_fallbacks() share a
# single source of truth.
#
# Brace-escaping rule: PromptTemplate.format() invokes str.format() on these
# strings. Every literal '{' or '}' that should appear in the final output
# (i.e., every brace that is part of a JSON example) MUST be doubled to '{{'
# or '}}'. Single-braced names ARE format placeholders and must match the
# keys passed in by the caller.
#
# Placeholder contracts:
#   chunk_analysis     -> chunk_content
#   question_generation -> chunk_content, total_questions, factual_count,
#                          inferential_count, conceptual_count, key_concepts,
#                          analysis_details
#   question_validation -> chunk_content, question_text, answer_text
#
# The JSON examples below use plausible non-zero values rather than literal
# zeros so the LLM has an anchor for the expected ranges. Earlier revisions
# used 0.0 for every numeric field; that combined with the strict-JSON
# instruction (zeros are valid JSON, ranges like '0.0-1.0' are not) but cost
# the range hint that the original prompts conveyed via inline annotations.
# ---------------------------------------------------------------------------


_CHUNK_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert analyst who evaluates source passages to determine how "
    "many high-quality, source-grounded questions can be generated from them. "
    "You return strictly valid JSON and nothing else."
)


_CHUNK_ANALYSIS_TEMPLATE = """\
Analyze the following text passage to estimate how many source-grounded \
questions can be generated from it.

Text passage:
---
{chunk_content}
---

Score the passage on three dimensions, each a float between 0.0 and 1.0:
- information_density: how much factual content per unit of text
- topic_coherence: how focused the passage is on a single topic
- complexity: how technically or conceptually difficult the content is

Then estimate how many questions of each category could be answered using \
only information present in this passage. Use these category definitions:

- factual: answerable by quoting or paraphrasing a single span of the text.
- inferential: requires combining two or more distinct pieces of information \
from the text.
- conceptual: requires identifying a principle or generalization that the \
text supports.

Estimate counts assuming roughly one substantive question per 100-150 words \
of meaningful content. The total across all three categories must not exceed \
10. If the passage is short, contains little substantive content, or is \
largely boilerplate, return zeros.

Finally, list 3-7 key concepts, terms, or named entities from the passage. \
These will be passed to the question generator to anchor what the questions \
should cover.

Return your analysis in EXACTLY this JSON format. The values shown below are \
illustrative of the expected ranges; replace them with values appropriate to \
the passage you analyzed:
{{
    "information_density": 0.65,
    "topic_coherence": 0.80,
    "complexity": 0.50,
    "estimated_question_yield": {{
        "factual": 3,
        "inferential": 2,
        "conceptual": 1
    }},
    "key_concepts": ["concept1", "concept2", "concept3"],
    "notes": "Brief observations about the passage that would help a question writer."
}}

**CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON structure.**
"""


_QUESTION_GENERATION_SYSTEM_PROMPT = (
    "You are an expert at extracting question-answer pairs from source text. "
    "You only generate questions that can be fully answered from the provided "
    "passage. You never use outside knowledge. Your question-answer pairs are "
    "used to train language models, so each pair must be fully self-contained: "
    "the question must make sense on its own, with no reference to the source "
    "passage, and the answer must read as a freestanding statement of fact. "
    "You return strictly valid JSON and nothing else."
)


_QUESTION_GENERATION_TEMPLATE = """\
Generate question-answer pairs from the following source text. The answers \
must be derivable from the source text alone, but the questions and answers \
themselves must be fully self-contained and readable without access to the \
source. These pairs will be used as training data for language models, where \
the source passage will NOT be available alongside the question.

Source text:
---
{chunk_content}
---

Analyzer context (use this to choose what to ask about and how deeply to \
answer; do not quote it back in your output):
- Key concepts to target: {key_concepts}
- Full analyzer output (JSON): {analysis_details}

Target counts (these are upper bounds, not quotas):
- Up to {factual_count} factual questions
- Up to {inferential_count} inferential questions
- Up to {conceptual_count} conceptual questions
- Total target: up to {total_questions} questions

Category definitions (use these exact criteria when labeling each question):

- factual: answerable by quoting or paraphrasing a single span of the source \
text. The answer is directly stated in the passage.
- inferential: requires combining two or more distinct pieces of information \
from the source text to answer. The answer is not directly stated but \
follows from what is stated.
- conceptual: requires identifying a principle, pattern, or generalization \
that the source text supports. The answer requires synthesizing across the \
passage, but must still be grounded in what the text says.

Grounding rules (these are mandatory):

1. The answer to every question must be fully supported by the source text. \
Do not draw on outside knowledge, common knowledge, or anything not present \
in the passage above.
2. If the source text does not contain enough information to fully answer a \
question, DO NOT generate that question. In this case, returning fewer questions than the \
target is correct and expected.
3. If the passage is too thin to generate any questions of a given category, \
omit that category entirely. Returning an empty list is acceptable.
4. The answer must directly address the question. Do not pad answers with \
background that is not in the source text.

Self-containment rules (these are mandatory):

Every question and answer must stand alone as if the source passage does not \
exist. A reader seeing only the question must be able to understand what is \
being asked without any external context. A reader seeing only the answer \
must read it as a freestanding statement of fact.

5. DO NOT refer to the source text in any way. Forbidden phrasings include, \
but are not limited to: "according to the text", "according to the passage", \
"the text says", "the passage states", "the document mentions", "the author \
writes", "as described", "as mentioned", "as discussed", "as shown above", \
"in the excerpt", "in this section", "is mentioned", "is described", \
"is discussed", "is referenced", "is stated", and any similar construction \
that points back at the source.
6. DO NOT use unresolved pronouns or demonstratives that depend on the \
source for their referent. Phrases like "this technique", "this method", \
"the algorithm", "the author", "this approach" are only acceptable if the \
question itself has already introduced what they refer to. When in doubt, \
name the thing explicitly.
7. Treat the question as if you are writing it for a quiz or exam where the \
test-taker has never seen the passage. The question must supply enough \
context — names of concepts, systems, people, or entities — that it can be \
understood and answered on its own.
8. The answer must also be self-contained. Do not write "the technique \
mentioned is X"; write "X is the technique used to ..." or simply "X". \
Restate the subject of the question rather than referring back to it as \
"this" or "that".

Examples of the bad/good distinction:

- BAD question: "What technique is mentioned for grouping the data in each \
sub-vector into k centroids?"
- GOOD question: "In product quantization, what algorithm groups the data \
in each sub-vector into k centroids?"

- BAD question: "According to the passage, what are the three components of \
a transformer block?"
- GOOD question: "What are the three components of a transformer block?"

- BAD answer: "The text states that k-means clustering is used."
- GOOD answer: "K-means clustering groups the data in each sub-vector into \
k centroids."

- BAD question: "What does the author argue is the main limitation of this \
approach?"
- GOOD question: "What is the main limitation of retrieval-augmented \
generation when the retrieval index is stale?"

If you find yourself writing a question that only makes sense because the \
reader has the passage, rewrite it to name the subject explicitly. If the \
subject cannot be named without the passage (e.g., the passage never \
identifies what system or method it is describing), do not generate that \
question.

Return your output as a JSON array. Each element is one question:
[
    {{
        "question": "The question text.",
        "answer": "The answer, fully supported by the source text.",
        "category": "factual"
    }}
]

The "category" field must be exactly one of: "factual", "inferential", "conceptual".

**CRITICAL: Your response MUST be ONLY a JSON array conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON array.**
"""


_QUESTION_VALIDATION_SYSTEM_PROMPT = (
    "You are an expert evaluator of question-answer pairs. You assess each "
    "pair against the source text on factual accuracy, answer completeness, "
    "and question clarity, providing a separate score and reason for each "
    "dimension. Question clarity includes whether the question and answer "
    "are self-contained and readable without access to the source passage, "
    "since these pairs are used as training data where the source is not "
    "available. You return strictly valid JSON and nothing else."
)


_QUESTION_VALIDATION_TEMPLATE = """\
Evaluate the following question-answer pair against the source text. \
Score each of the three dimensions independently. Each dimension has its \
own score (a float between 0.0 and 1.0) and its own reason explaining that \
score.

Source text:
---
{chunk_content}
---

Question: {question_text}

Answer: {answer_text}

Score these three dimensions:

1. factual_accuracy: Is every claim in the answer supported by the source text?
   - 1.0: every claim is directly supported by the source text.
   - 0.5: the answer is partially supported; some claims have no support in the text.
   - 0.0: the answer contradicts the source text or relies on outside knowledge.

2. answer_completeness: Does the answer fully address what the question asks?
   - 1.0: the answer addresses every part of the question.
   - 0.5: the answer addresses the question but omits material that the source text contains and the question implies.
   - 0.0: the answer does not actually address the question.

3. question_clarity: Is the question clear, specific, unambiguous, AND \
self-contained? Self-contained means the question and answer can be \
understood without access to the source passage, since these pairs are used \
as training data where the source is not available. Penalize phrasings like \
"according to the text", "the passage states", "as mentioned", "is \
described", "the author writes", or unresolved references like "this \
technique" or "the algorithm" where the referent is only knowable from the \
source.
   - 1.0: the question is unambiguous and fully self-contained; a reader \
seeing only the question (without the source) understands what is being \
asked, and the answer reads as a freestanding statement of fact.
   - 0.5: the question is understandable but vague, open to multiple \
interpretations, or contains a mild source reference (e.g., one unresolved \
pronoun, or a phrase like "as described") that a reader could still work \
around.
   - 0.0: the question is unclear, malformed, cannot be answered as \
written, OR depends on the source passage to make sense (e.g., "What \
technique is mentioned for X?", "According to the passage, what is Y?", \
"What does the author argue about Z?").

For each dimension, provide both the score and a concise reason (one or two \
sentences) explaining that specific score. The reason must address the \
dimension it belongs to.

Return your evaluation in EXACTLY this JSON format. The values shown below \
are illustrative of the expected ranges; replace them with values \
appropriate to the question-answer pair you evaluated:
{{
    "factual_accuracy": {{
        "score": 0.85,
        "reason": "Concise explanation of the factual_accuracy score."
    }},
    "answer_completeness": {{
        "score": 0.75,
        "reason": "Concise explanation of the answer_completeness score."
    }},
    "question_clarity": {{
        "score": 0.90,
        "reason": "Concise explanation of the question_clarity score."
    }},
    "suggested_improvements": "Optional. A short suggestion for improving the question or answer, or an empty string."
}}

**CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON structure.**
"""