# src/semantic_qa_gen/llm/prompts/manager.py
"""Prompt management system for SemanticQAGen."""

import os
import yaml
import logging
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

    This class handles loading prompt templates from YAML files and providing
    them on demand for various LLM tasks.

    Loading contract (important):
      * YAML templates in `prompts_dir` are the primary source.
      * Module-level `_*_TEMPLATE` constants are the fallback, used when a YAML
        file is absent or fails to register a given essential prompt.
      * EVERY key in `ESSENTIAL_PROMPTS` MUST have a corresponding fallback
        constant registered by `_register_selected_fallbacks`, so the system can
        always start even if the YAML directory is empty or malformed. A runtime
        component that requires a prompt key it cannot fall back to is a bug:
        add the key here, give it a fallback, or both.
      * A YAML parse failure is treated as LOUD: it is logged at ERROR and
        counted. If files exist but none parse, that is reported explicitly
        rather than being silently indistinguishable from an empty directory.
        This is the safeguard against the failure mode where startup looks
        healthy (essentials fall back) but a non-essential, runtime-required
        key is silently missing.
    """

    # Keys the system requires to be present after initialization. Each MUST
    # have a fallback constant (see _register_selected_fallbacks).
    ESSENTIAL_PROMPTS = {
        "chunk_analysis",
        "question_generation",
        "faithfulness_validation",
        "standalone_validation",
        "decontextualize",
    }

    def __init__(self, prompts_dir: Optional[str] = None, strict_yaml: bool = False,
                 write_missing: bool = True):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Optional directory for loading prompts. When provided,
                this is treated as the PROJECT prompts dir and missing default
                templates are materialized into it on startup (see write_missing).
            strict_yaml: If True, a YAML file that fails to parse raises
                ConfigurationError immediately instead of falling back. Defaults
                to False (fall back, but log loudly), to preserve the ability to
                run on the built-in constants.
            write_missing: If True (default), after loading, any default template
                file that is absent from `prompts_dir` is written there so it can
                be customized. This happens ONLY when an explicit `prompts_dir`
                was supplied (never into the read-only bundled package dir), and
                NEVER overwrites an existing file. A write failure is logged and
                swallowed: the run continues on the in-memory constants, because
                materialization is a convenience, not a prerequisite for running.
        """
        self.prompts: Dict[str, PromptTemplate] = {}
        self.logger = logging.getLogger(__name__)
        self.strict_yaml = strict_yaml
        self.write_missing = write_missing

        # Distinguish a caller-supplied PROJECT dir from the bundled package dir.
        # Defaults are only materialized into a project dir, never the package.
        self._is_project_dir = prompts_dir is not None

        # Default prompts directory within the package
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )

        # Track which prompts were loaded from files (vs. fallbacks)
        self.loaded_from_files: Set[str] = set()
        # Track YAML files that failed to parse, for loud diagnostics
        self.failed_files: List[str] = []

        # Register built-in prompts
        self._register_builtin_prompts()

        # Materialize default templates into a project dir if any are missing.
        # Runs AFTER loading so prompts are already in memory; a write failure
        # here cannot affect this run.
        if self.write_missing and self._is_project_dir:
            self.materialize_defaults(self.prompts_dir, overwrite=False)

        # Verify essential prompts
        self._verify_essential_prompts()

    def _register_builtin_prompts(self) -> None:
        """Load prompts from YAML, then backfill any missing essentials from
        constants. Parse failures are surfaced loudly rather than swallowed."""
        yaml_files: List[str] = []
        if os.path.exists(self.prompts_dir):
            yaml_files = [
                f for f in os.listdir(self.prompts_dir)
                if f.endswith(".yaml") or f.endswith(".yml")
            ]

        any_keys_loaded = False
        for filename in sorted(yaml_files):
            path = os.path.join(self.prompts_dir, filename)
            try:
                keys_loaded = self._load_from_file(path)
            except ConfigurationError as e:
                # LOUD: a present file that fails to parse is an error condition,
                # not a reason to silently fall through to fallbacks.
                self.failed_files.append(filename)
                self.logger.error(f"Failed to load prompt file '{filename}': {e}")
                if self.strict_yaml:
                    raise
                continue

            if keys_loaded:
                any_keys_loaded = True
                self.logger.info(f"Loaded {len(keys_loaded)} prompt template(s) from {filename}")
                for key in self.ESSENTIAL_PROMPTS & keys_loaded:
                    self.logger.info(f"Loaded essential prompt '{key}' from {filename}")

        # If files were present but every one failed to parse, say so plainly.
        if yaml_files and not any_keys_loaded:
            self.logger.error(
                f"{len(self.failed_files)} prompt file(s) present in '{self.prompts_dir}' "
                f"but none parsed: {self.failed_files}. Falling back to built-in constants."
            )
        elif not yaml_files:
            self.logger.warning(
                f"No prompt files found in '{self.prompts_dir}'. Using built-in constants."
            )

        # Backfill any essential prompt not registered from a file. This always
        # runs (not just when nothing loaded), so a YAML file that defines only
        # SOME prompts can never leave an essential key missing.
        missing_essential = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
        if missing_essential:
            self.logger.info(f"Backfilling essential prompts from constants: {sorted(missing_essential)}")
            self._register_selected_fallbacks(missing_essential)

    def _load_from_file(self, path: str) -> Set[str]:
        """
        Load prompt templates from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Set of keys loaded from this file.

        Raises:
            ConfigurationError: If the file is not valid YAML or not a mapping.
        """
        loaded_keys: Set[str] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Could not read {path}: {e}")

        if prompt_data is None:
            # Empty document is not an error; just nothing to load.
            return loaded_keys
        if not isinstance(prompt_data, dict):
            raise ConfigurationError(
                f"Invalid prompt file format: {path}. Expected a top-level mapping "
                f"of prompt-name -> definition, got {type(prompt_data).__name__}. "
                f"(A leading Python-style \"\"\"docstring\"\"\" will cause this.)"
            )

        for name, data in prompt_data.items():
            if not isinstance(data, dict):
                self.logger.error(f"Invalid prompt definition for '{name}' in {path}: not a mapping")
                continue
            if "template" not in data:
                self.logger.error(f"Invalid prompt definition for '{name}' in {path}: missing 'template'")
                continue
            template = data.pop("template")
            if not isinstance(template, str):
                self.logger.error(f"Invalid prompt definition for '{name}' in {path}: 'template' is not a string")
                continue

            metadata = data
            metadata["source"] = os.path.basename(path)

            # Heuristic guard: a generation prompt that opens with analysis language
            # is almost certainly the wrong template pasted into the wrong key.
            if name == "question_generation" and "analyze" in template.lower()[:100]:
                self.logger.warning(
                    f"Prompt '{name}' in {path} opens like an analysis prompt, not a "
                    f"generation prompt. This will likely cause generation failures."
                )

            self.register_prompt(name, template, metadata)
            loaded_keys.add(name)
            self.loaded_from_files.add(name)

        return loaded_keys

    def _register_selected_fallbacks(self, missing_keys: Set[str]) -> None:
        """Register only the specified fallback prompts from module constants.

        Every key in ESSENTIAL_PROMPTS must be handled here.
        """
        registry = {
            "chunk_analysis": (
                _CHUNK_ANALYSIS_TEMPLATE,
                {
                    "description": "Analyzes a text chunk for information density and question potential",
                    "json_output": True,
                    "system_prompt": _CHUNK_ANALYSIS_SYSTEM_PROMPT,
                },
            ),
            "question_generation": (
                _QUESTION_GENERATION_TEMPLATE,
                {
                    "description": "Generates source-grounded question/answer pairs from a text chunk",
                    "json_output": True,
                    "system_prompt": _QUESTION_GENERATION_SYSTEM_PROMPT,
                },
            ),
            "faithfulness_validation": (
                _FAITHFULNESS_VALIDATION_TEMPLATE,
                {
                    "description": "Scores answer faithfulness (accuracy + completeness) against the source",
                    "json_output": True,
                    "system_prompt": _FAITHFULNESS_VALIDATION_SYSTEM_PROMPT,
                },
            ),
            "standalone_validation": (
                _STANDALONE_VALIDATION_TEMPLATE,
                {
                    "description": "Scores whether a Q/A pair is understandable without the source",
                    "json_output": True,
                    "system_prompt": _STANDALONE_VALIDATION_SYSTEM_PROMPT,
                },
            ),
            "decontextualize": (
                _DECONTEXTUALIZE_TEMPLATE,
                {
                    "description": "Rewrites a Q/A pair to stand alone, preserving grounded meaning",
                    "json_output": True,
                    "system_prompt": _DECONTEXTUALIZE_SYSTEM_PROMPT,
                },
            ),
        }
        for key in missing_keys:
            entry = registry.get(key)
            if entry is None:
                # An essential key with no fallback is a programming error; make it visible.
                self.logger.error(
                    f"No fallback constant registered for essential prompt '{key}'. "
                    f"This is a bug in PromptManager."
                )
                continue
            template, metadata = entry
            self.logger.info(f"Registering fallback prompt: {key}")
            self.register_prompt(key, template, metadata)

    def _register_fallback_prompts(self) -> None:
        """Register the full set of fallback prompts (all essentials)."""
        self._register_selected_fallbacks(set(self.ESSENTIAL_PROMPTS))

    def _verify_essential_prompts(self) -> None:
        """Verify that all essential prompts are available, else fail fast."""
        missing = self.ESSENTIAL_PROMPTS - set(self.prompts.keys())
        if missing:
            error_msg = f"Critical prompt templates missing after initialization: {', '.join(sorted(missing))}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)

        # Sanity check: generation prompt should not actually be an analysis prompt.
        if "question_generation" in self.prompts:
            template = self.prompts["question_generation"].template.lower()
            if "analyze" in template[:100] and "density" in template and "coherence" in template:
                self.logger.error(
                    "The 'question_generation' prompt appears to be an analysis prompt. "
                    "This will cause generation failures."
                )

    def register_prompt(self, name: str, template: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new prompt template."""
        self.prompts[name] = PromptTemplate(template, metadata)
        self.logger.debug(f"Registered prompt template: {name}")

    def get_prompt(self, name: str) -> PromptTemplate:
        """Get a prompt template by name. Raises LLMServiceError if absent."""
        if name not in self.prompts:
            raise LLMServiceError(f"Prompt template not found: {name}")
        return self.prompts[name]

    def format_prompt(self, name: str, **kwargs) -> str:
        """Format a prompt template with variable substitution."""
        try:
            template = self.get_prompt(name)
            return template.format(**kwargs)
        except KeyError as e:
            raise LLMServiceError(f"Missing variable in prompt template {name}: {str(e)}")
        except Exception as e:
            raise LLMServiceError(f"Error formatting prompt {name}: {str(e)}")

    def get_system_prompt(self, name: str) -> Optional[str]:
        """Get the system prompt for a template, if defined."""
        template = self.get_prompt(name)
        return template.metadata.get("system_prompt")

    def is_json_output(self, name: str) -> bool:
        """Check if a prompt expects JSON output."""
        template = self.get_prompt(name)
        return template.metadata.get("json_output", False)

    # -- Default file layout: which prompts get written into which YAML file. --
    # Each entry maps a target filename to the prompt keys it should contain.
    # A file is materialized only if it is missing from the target dir entirely.
    DEFAULT_TEMPLATE_FILES = {
        "analysis_prompts.yaml": ["chunk_analysis"],
        "generation_prompts.yaml": ["question_generation"],
        "validation_prompts.yaml": ["faithfulness_validation", "standalone_validation"],
        "decontextualize_prompts.yaml": ["decontextualize"],
    }

    def _default_definitions(self) -> Dict[str, Dict[str, Any]]:
        """The canonical default prompt definitions, sourced from the module
        constants. Returns {key: {template, description, json_output, system_prompt}}."""
        return {
            "chunk_analysis": {
                "template": _CHUNK_ANALYSIS_TEMPLATE,
                "description": "Analyzes a text chunk for information density and question potential.",
                "json_output": True,
                "system_prompt": _CHUNK_ANALYSIS_SYSTEM_PROMPT,
            },
            "question_generation": {
                "template": _QUESTION_GENERATION_TEMPLATE,
                "description": "Generates source-grounded question/answer pairs from a text chunk.",
                "json_output": True,
                "system_prompt": _QUESTION_GENERATION_SYSTEM_PROMPT,
            },
            "faithfulness_validation": {
                "template": _FAITHFULNESS_VALIDATION_TEMPLATE,
                "description": "Scores answer faithfulness (accuracy + completeness) against the source.",
                "json_output": True,
                "system_prompt": _FAITHFULNESS_VALIDATION_SYSTEM_PROMPT,
            },
            "standalone_validation": {
                "template": _STANDALONE_VALIDATION_TEMPLATE,
                "description": "Scores whether a Q/A pair is understandable without the source.",
                "json_output": True,
                "system_prompt": _STANDALONE_VALIDATION_SYSTEM_PROMPT,
            },
            "decontextualize": {
                "template": _DECONTEXTUALIZE_TEMPLATE,
                "description": "Rewrites a Q/A pair to stand alone, preserving its grounded meaning.",
                "json_output": True,
                "system_prompt": _DECONTEXTUALIZE_SYSTEM_PROMPT,
            },
        }

    def materialize_defaults(self, target_dir: str, overwrite: bool = False) -> List[str]:
        """Write default template files into `target_dir` so they can be edited.

        Behavior:
          * Only writes a file that is ABSENT (unless overwrite=True). An existing
            file is never clobbered, so user edits are preserved.
          * Every write is verified: the file is re-read, parsed as YAML, and each
            prompt is str.format()-checked with placeholder probes. If verification
            fails, the bad file is removed and the failure logged. This makes it
            impossible for this method to leave behind the broken-YAML class of
            file (docstring header, single braces) we otherwise guard against.
          * Any OS/permission error is logged and swallowed: materialization is a
            convenience, never a prerequisite for the program to run.

        Returns:
            The list of file paths actually written (verified-good).
        """
        written: List[str] = []
        defs = self._default_definitions()

        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            self.logger.warning(f"Could not create prompts dir '{target_dir}': {e}. "
                                 f"Skipping template materialization (run continues on constants).")
            return written

        for filename, keys in self.DEFAULT_TEMPLATE_FILES.items():
            path = os.path.join(target_dir, filename)
            if os.path.exists(path) and not overwrite:
                continue

            payload = {k: defs[k] for k in keys if k in defs}
            if not payload:
                continue

            try:
                content = self._render_yaml_file(filename, payload)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                self.logger.warning(f"Could not write default template '{path}': {e}. "
                                    f"Run continues on in-memory constants.")
                continue

            # Verify the file we just wrote actually loads and formats.
            if self._verify_written_file(path, payload):
                written.append(path)
                self.logger.info(f"Wrote default prompt template: {path}")
            else:
                # Don't leave a broken artifact behind.
                try:
                    os.remove(path)
                except OSError:
                    pass
                self.logger.error(f"Default template '{path}' failed post-write verification; removed.")

        return written

    def _render_yaml_file(self, filename: str, payload: Dict[str, Dict[str, Any]]) -> str:
        """Render a YAML template file. Templates are emitted as literal block
        scalars with their doubled braces intact (the loader runs str.format()
        on them exactly as it does for the constants)."""
        lines = [
            f"# Default prompt templates ({filename}), written by SemanticQAGen.",
            "# Edit freely. This file is NOT overwritten once it exists.",
            "#",
            "# This is YAML, not Python: do NOT add a leading \"\"\"docstring\"\"\".",
            "# Braces in JSON examples are doubled ({{ }}) because each template is",
            "# consumed by str.format(); single braces will break formatting.",
            "",
        ]
        for key, d in payload.items():
            lines.append(f"{key}:")
            desc = str(d.get("description", "")).replace('"', '\\"')
            lines.append(f'  description: "{desc}"')
            lines.append(f"  json_output: {str(bool(d.get('json_output', True))).lower()}")
            sysp = str(d.get("system_prompt", "")).replace('"', '\\"')
            lines.append(f'  system_prompt: "{sysp}"')
            lines.append("  template: |")
            for tline in d["template"].splitlines():
                lines.append(f"    {tline}" if tline else "")
            lines.append("")
        return "\n".join(lines) + "\n"

    def _verify_written_file(self, path: str, payload: Dict[str, Dict[str, Any]]) -> bool:
        """Re-read a written file: it must parse as a mapping and every template
        must str.format() with placeholder probes without raising."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            self.logger.error(f"Verification read/parse failed for '{path}': {e}")
            return False
        if not isinstance(data, dict):
            self.logger.error(f"Verification: '{path}' did not parse as a mapping.")
            return False

        import string as _string
        for key in payload:
            if key not in data or "template" not in data[key]:
                self.logger.error(f"Verification: '{key}' missing from written '{path}'.")
                return False
            tmpl = data[key]["template"]
            # Probe every placeholder the template declares, so format() can't KeyError.
            field_names = {fn for _, fn, _, _ in _string.Formatter().parse(tmpl) if fn}
            probes = {fn: "x" for fn in field_names}
            try:
                tmpl.format(**probes)
            except Exception as e:
                self.logger.error(f"Verification: template '{key}' in '{path}' fails to format: {e}")
                return False
        return True


# ---------------------------------------------------------------------------
# Fallback prompt templates
#
# Used when no YAML template is present for a given essential key, or when a
# YAML file fails to parse. These are the canonical content; the YAML files in
# templates/ should mirror them. Keeping both in sync is a Phase 1 cleanup; for
# now the constants are the guaranteed-correct floor.
#
# Brace-escaping rule: PromptTemplate.format() invokes str.format() on these
# strings (whether they came from here or from YAML — the format step is the
# same). Every literal '{' or '}' that should appear in the model's output
# (i.e., every brace in a JSON example) MUST be doubled to '{{' or '}}'.
# Single-braced names ARE format placeholders and must match the caller's keys.
#
# Placeholder contracts:
#   chunk_analysis           -> chunk_content
#   question_generation      -> chunk_content, total_questions, factual_count,
#                               inferential_count, conceptual_count,
#                               key_concepts, analysis_details
#   faithfulness_validation  -> chunk_content, question_text, answer_text
#   standalone_validation    -> question_text, answer_text   (NO chunk_content,
#                               by design — see below)
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
    "You are an expert at writing question-answer pairs from source text for "
    "training language models. You ground every answer in the provided passage "
    "and use no outside knowledge. You write each question so it names its own "
    "subject and reads as a complete, freestanding question, and each answer so "
    "it restates that subject as a standalone statement of fact, as concise as "
    "fully answering allows and never padded. You match the form of the worked "
    "examples you are given. You return strictly valid JSON and nothing else."
)


# NOTE: Phase 1 rewrite. Positive, form-first framing with good-only exemplars
# (no blocklist, no bad examples). Leakage is caught downstream by the
# deterministic leak filter and the source-free standalone judge, so this prompt
# teaches good form by demonstration rather than policing avoidance.
_QUESTION_GENERATION_TEMPLATE = """\
Write question-answer pairs from the source text below. Ground every answer in \
what the passage states, and write each pair so it stands on its own as \
training data, where the source passage will NOT be shown alongside it.

Source text:
---
{chunk_content}
---

Analyzer context (use this to choose what to ask about and how deeply to \
answer; do not quote it back in your output):
- Key concepts to target: {key_concepts}
- Full analyzer output (JSON): {analysis_details}

Target counts (upper bounds, not quotas):
- Up to {factual_count} factual questions
- Up to {inferential_count} inferential questions
- Up to {conceptual_count} conceptual questions
- Total: up to {total_questions} questions

Category definitions (label each question with exactly one):

- factual: answerable from a single span of the source text. The answer is \
directly stated in the passage.
- inferential: requires combining two or more distinct pieces of information \
from the source text. The answer follows from what is stated without being \
stated outright.
- conceptual: requires identifying a principle, pattern, or generalization the \
source text supports, synthesizing across the passage while staying grounded \
in what it says.

How to ground each pair:

1. Use only the source text. Draw on no outside or common knowledge beyond \
what the passage states.
2. Make the answer fully supported by the passage and directly responsive to \
the question, with no padding.
3. Generate a question only when the passage supports a complete answer.

How to make each pair stand on its own:

4. Name the subject of every question explicitly — the concept, system, \
method, person, or entity it is about — so a reader who has never seen the \
passage understands and can answer it. If the passage never names the subject \
it describes, do not ask about it.
5. Write each answer as a freestanding statement of fact that restates its \
subject. Prefer "The Peace of Westphalia ended ..." over "It ended ..." or \
"The treaty mentioned is ...".
6. Vary the phrasing and the question stems across the set; do not start every \
question the same way.

How long each answer should be:

7. Make each answer only as long as fully answering requires, and no longer. \
Length follows the question, not a fixed target. As a guide: factual answers \
are usually a single sentence; inferential answers give just enough to show how \
the combined facts connect; conceptual answers state the principle and what \
grounds it, typically two to three sentences.
8. There is no minimum length and no quota of words. Never pad an answer to seem \
thorough, never add background, restatement, or hedging the question did not \
ask for, and do not inflate the answer because the question was hard. A \
complete short answer is better than a padded long one.

Examples of well-formed pairs (these illustrate FORM ONLY — write your \
questions about the source text above, NOT about these topics. Note how the \
factual answers stop as soon as the fact is stated):
[
    {{
        "question": "What treaty ended the Thirty Years' War?",
        "answer": "The Peace of Westphalia ended the Thirty Years' War in 1648.",
        "category": "factual"
    }},
    {{
        "question": "Why does a central bank raising interest rates tend to reduce inflation but slow economic growth?",
        "answer": "Higher rates make borrowing more expensive, which curbs spending and eases upward pressure on prices; the same drop in spending also slows output and hiring, which dampens growth.",
        "category": "inferential"
    }},
    {{
        "question": "How does natural selection lead a population to become better adapted to its environment over time?",
        "answer": "Individuals whose traits suit their environment tend to survive and reproduce more, so those traits grow more common across generations, gradually shifting the population toward greater adaptation.",
        "category": "conceptual"
    }},
    {{
        "question": "Which clustering algorithm does product quantization use to group the vectors in each sub-space into k centroids?",
        "answer": "Product quantization uses k-means clustering to group the vectors in each sub-space into k centroids.",
        "category": "factual"
    }}
]

Return your output as a JSON array with this exact structure, one object per \
question:
[
    {{
        "question": "The question text, naming its own subject.",
        "answer": "The answer, fully supported by the source text.",
        "category": "factual"
    }}
]

The "category" field must be exactly one of: "factual", "inferential", "conceptual".

**CRITICAL: Your response MUST be ONLY a JSON array conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON array.**
"""


_FAITHFULNESS_VALIDATION_SYSTEM_PROMPT = (
    "You are an expert evaluator of question-answer pairs. You judge only "
    "whether the answer is supported by, and complete with respect to, the "
    "provided source text. You do not judge wording or style. You return "
    "strictly valid JSON and nothing else."
)


_FAITHFULNESS_VALIDATION_TEMPLATE = """\
Evaluate the following question-answer pair against the source text. Score the \
two dimensions independently. Each dimension has its own score (a float between \
0.0 and 1.0) and its own one- or two-sentence reason.

Source text:
---
{chunk_content}
---

Question: {question_text}

Answer: {answer_text}

Score these two dimensions:

1. factual_accuracy: Is every claim in the answer supported by the source text?
   - 1.0: every claim is directly supported by the source text.
   - 0.5: partially supported; some claims have no support in the text.
   - 0.0: the answer contradicts the source text or relies on outside knowledge.

2. answer_completeness: Does the answer fully address what the question asks, \
using what the source text contains?
   - 1.0: the answer addresses every part of the question.
   - 0.5: addresses the question but omits material the source contains and the question implies.
   - 0.0: does not actually address the question.

Return EXACTLY this JSON object. The numbers shown are illustrative of the \
expected ranges; replace them with values appropriate to the pair you scored:
{{
    "factual_accuracy": {{
        "score": 0.85,
        "reason": "Concise explanation of the factual_accuracy score."
    }},
    "answer_completeness": {{
        "score": 0.75,
        "reason": "Concise explanation of the answer_completeness score."
    }},
    "suggested_improvements": "Optional short suggestion, or an empty string."
}}

**CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON structure.**
"""


_STANDALONE_VALIDATION_SYSTEM_PROMPT = (
    "You are an expert evaluator who judges whether a question-answer pair can "
    "be understood on its own, with no access to any source document. You are "
    "NOT given the source, by design. You return strictly valid JSON and "
    "nothing else."
)


# IMPORTANT: this template intentionally has NO {chunk_content} placeholder.
# Whether a question stands alone cannot be judged while looking at the source it
# must stand apart from, so the standalone call is never given the source.
_STANDALONE_VALIDATION_TEMPLATE = """\
You are evaluating a single question-answer pair that will be used as training \
data, where the original source document will NOT be available alongside it. \
You have intentionally NOT been given the source. Judge the pair only on \
whether it stands on its own.

Question: {question_text}

Answer: {answer_text}

Score one dimension:

standalone: Can a reader who has never seen any source understand the question \
and read the answer as a freestanding statement of fact?

Penalize, in proportion to severity:
  - References to a source container: "according to the passage", "the text \
states", "as described above", "in this document". (If you see these, the \
score should be near 0.0.)
  - Unresolved references whose antecedent is not supplied by the question \
itself: a leading "this technique", "that method", "the algorithm", "the \
approach", "the author", where the question never names what is meant. A \
reader with no source cannot resolve these. This is the most important thing \
to check, since a deterministic filter does not catch it.
  - Questions only answerable if the reader can see some specific unnamed \
passage, figure, list, or example.

Reward questions that name their subject explicitly (e.g., "In product \
quantization, what algorithm groups each sub-vector into k centroids?") and \
answers that restate the subject rather than pointing back at it.

Scoring guide:
  - 1.0: fully self-contained; subject named; no dangling references; \
answerable as written by a reader with no source.
  - 0.5: understandable but with one mild unresolved reference or vague phrasing.
  - 0.0: depends on an unseen source to make sense (container reference, or a \
dangling "this/that X" with no antecedent in the question).

Return EXACTLY this JSON object. The number shown is illustrative; replace it \
with a value appropriate to the pair you scored:
{{
    "standalone": {{
        "score": 0.90,
        "reason": "Concise explanation, naming any unresolved reference you found."
    }},
    "suggested_improvements": "Optional rewrite suggestion that names the subject, or an empty string."
}}

**CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON structure.**
"""


_DECONTEXTUALIZE_SYSTEM_PROMPT = (
    "You revise question-answer pairs so they stand on their own as training "
    "data, without access to any source passage. You preserve the meaning: you "
    "ask about the same thing and keep the same answer fact. You may consult the "
    "source text only to resolve what a vague reference points to; you never add "
    "facts that are not in the source. You return strictly valid JSON and "
    "nothing else."
)


# Placeholder contract: chunk_content, question_text, answer_text, category, issue
_DECONTEXTUALIZE_TEMPLATE = """\
A question-answer pair needs revising so it stands on its own as training data, \
where the source passage will NOT be shown alongside it. Rewrite it so the \
question names its own subject explicitly and the answer reads as a freestanding \
statement of fact. Preserve the meaning — ask about the same thing, keep the \
same answer fact, and keep the same category and difficulty. Add no information \
that is not in the source text.

Source text (use only to resolve what a vague reference points to; do not quote \
it back):
---
{chunk_content}
---

Original question ({category}): {question_text}
Original answer: {answer_text}

What was wrong: {issue}

How to rewrite:
- Replace vague references ("this technique", "the algorithm", "it", "the \
author") with the explicit named subject drawn from the source.
- Remove any phrase that points back at the source ("is mentioned", "as \
described", "according to the passage").
- Keep the answer's fact identical; change only how it is phrased so it reads as \
a standalone statement, and keep it as concise as fully answering allows.
- Keep the question in its original category ({category}); do not turn a factual \
question into a conceptual one or vice versa.

If the source does not name the subject clearly enough to make the pair stand \
alone, do not invent one — return null for both fields.

Return EXACTLY this JSON object:
{{
    "question": "The rewritten question, naming its subject.",
    "answer": "The rewritten answer, a standalone statement of fact."
}}

If the pair cannot be made standalone from the source, return exactly:
{{"question": null, "answer": null}}

**CRITICAL: Your response MUST be ONLY a single, strictly valid JSON object conforming to RFC 8259.**
**DO NOT include any comments (// or #), markdown code fences, or any text outside the JSON structure.**
"""