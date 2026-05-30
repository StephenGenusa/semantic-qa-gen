"""Regression tests for default-template materialization.

Contract locked in here:
  * A fresh project dir gets all default template files written, and those files
    round-trip (a second manager loads + formats them).
  * An existing file is never overwritten (user edits preserved); only missing
    files are written.
  * The bundled package dir is never written into.
  * write_missing=False disables writing entirely.
  * A read-only target dir does not crash init; the run continues on constants.

Run: pytest tests/test_prompt_materialize.py
"""

import glob
import os
import tempfile

from semantic_qa_gen.llm.prompts.manager import PromptManager

ESSENTIAL = {
    "chunk_analysis", "question_generation",
    "faithfulness_validation", "standalone_validation",
}
PROMPT_VARS = {
    "chunk_analysis": {"chunk_content": "C"},
    "question_generation": {
        "chunk_content": "C", "total_questions": 5, "factual_count": 2,
        "inferential_count": 2, "conceptual_count": 1,
        "key_concepts": "k", "analysis_details": "{}",
    },
    "faithfulness_validation": {"chunk_content": "C", "question_text": "Q", "answer_text": "A"},
    "standalone_validation": {"question_text": "Q", "answer_text": "A"},
}


def test_fresh_dir_writes_all_defaults_and_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        PromptManager(prompts_dir=d)
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.yaml")))
        assert names == ["analysis_prompts.yaml", "generation_prompts.yaml", "validation_prompts.yaml"]
        # Second manager must load every essential FROM A FILE and format it.
        pm2 = PromptManager(prompts_dir=d)
        for k in ESSENTIAL:
            assert pm2.prompts[k].metadata.get("source", "CONSTANT").endswith(".yaml")
            pm2.format_prompt(k, **PROMPT_VARS[k])


def test_existing_file_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        custom = os.path.join(d, "validation_prompts.yaml")
        with open(custom, "w") as f:
            f.write(
                "faithfulness_validation:\n  json_output: true\n  template: |\n"
                "    CUSTOM {chunk_content} {question_text} {answer_text}\n"
                "standalone_validation:\n  json_output: true\n  template: |\n"
                "    CUSTOM {question_text} {answer_text}\n"
            )
        before = open(custom).read()
        pm = PromptManager(prompts_dir=d)
        assert open(custom).read() == before
        assert "CUSTOM" in pm.prompts["faithfulness_validation"].template
        # other missing defaults were still written
        assert os.path.exists(os.path.join(d, "analysis_prompts.yaml"))
        assert os.path.exists(os.path.join(d, "generation_prompts.yaml"))


def test_package_dir_not_written():
    pm = PromptManager()  # no prompts_dir -> bundled package dir
    assert not os.path.exists(os.path.join(pm.prompts_dir, "generation_prompts.yaml"))


def test_write_missing_false_disables_writing():
    with tempfile.TemporaryDirectory() as d:
        pm = PromptManager(prompts_dir=d, write_missing=False)
        assert glob.glob(os.path.join(d, "*.yaml")) == []
        assert not (ESSENTIAL - set(pm.prompts.keys()))


def test_readonly_dir_does_not_crash():
    d = tempfile.mkdtemp()
    os.chmod(d, 0o500)
    try:
        pm = PromptManager(prompts_dir=d)
        assert not (ESSENTIAL - set(pm.prompts.keys()))
    finally:
        os.chmod(d, 0o700)