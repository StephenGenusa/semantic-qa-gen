<div align="center">
    <h1>SemanticQAGen</h1>
  <p><em>Intelligent Question-Answer Generation with Advanced Semantic Understanding For AI Fine-Tuning</em></p>
</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#feature-overview">Feature Overview</a> •
  <a href="#core-capabilities">Core Capabilities</a> •
  <a href="#advanced-features">Advanced Features</a> •
  <a href="#project-file-organization">Project File Organization</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#cli-reference">CLI Reference</a> •
  <a href="#usage-examples">Usage Examples</a> •
  <a href="#extension">Extension</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#license">License</a>
</p>

---
> **Alpha Release (v0.10)**: This library is in active development and not functional yet. I estimate it is 90% complete, so the last %10 percent may take 90% of the time. However, the feature set is amazing IMO, and I am looking forward to completing this library. If you are interested in helping, please fork the repo and help bring this Python library to life. 
---

## Overview

SemanticQAGen is a powerful Python library for generating high-quality question-answer pairs from text documents. It uses advanced semantic understanding to intelligently process content, analyze information density, and create diverse questions across multiple cognitive levels.

This builds upon concepts from the [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) project by Evan Armstrong and other QA generation projects. I tried Evan's project on some test data and fell in love with it. I originally intended to fork his project and work on adding features I wanted to see. After making the plan, I realized that this wasn't simply a fork. It just felt wrong to try and completely redesign someone else's excellent library and I wanted a much different feature set and so this library is the result. 

SemanticQAGen features enhanced semantic chunking, dynamic question generation, optional *validation* of questions and answers using retrieval-augmented generation (RAG), and flexible LLM routing capabilities. You can run all tasks locally on an OpenAI compatible server, run them off an Internet hosted API, or split specific tasks (eg. chunking, validation, analysis, generation) between local and remote servers. The library is designed with the "for Humans" philosophy - simple for basic use cases while providing advanced capabilities for power users.

---

## Installation

### Basic Installation

```bash
pip install semantic-qa-gen
```

### With Optional Dependencies

```bash
# With PDF support
pip install semantic-qa-gen[pdf]

# With full document support (PDF, DOCX, etc.)
pip install semantic-qa-gen[docs]

# With RAG validation capabilities
pip install semantic-qa-gen[rag]

# With enhanced UI and NLP capabilities
pip install semantic-qa-gen[full]

# Development installation with testing tools
pip install semantic-qa-gen[dev]
```

### Requirements

- Python 3.10 or higher
- Required dependencies are automatically installed with the package

---

## Quickstart

### Basic Usage

```python
from semantic_qa_gen import SemanticQAGen

# Initialize with default settings
qa_gen = SemanticQAGen()

# Process a document
result = qa_gen.process_document("path/to/document.txt")

# Save the questions to a JSON file
qa_gen.save_questions(result, "output")
```

### CLI Usage

```bash
# Generate questions from a document with default settings
semantic-qa-gen process document.pdf -o questions_output

# Process a whole directory of documents
semantic-qa-gen process-batch input_directory/ -o output_directory/

# Create a config file interactively 
semantic-qa-gen init-config config.yml --interactive

# Process with a specific configuration
semantic-qa-gen process document.txt --config config.yml --format json
```

---

## Feature Overview

SemanticQAGen offers a comprehensive set of features designed to produce high-quality question and answer sets:

| Feature Category | Capability                                                         | Included |
|-----------------|--------------------------------------------------------------------|:--------:|
| **Document Processing** | Document format support: TXT, PDF, DOCX, MD                      | ✅ |
|  | Batch document processing                                          | ✅ |
|  | Automatic document type detection                                  | ✅ |
|  | Cross-page content handling                                        | ✅ |
|  | Header/footer detection and removal                                | ✅ |
| **Content Analysis** | Semantic document chunking                                         | ✅ |
|  | Information density analysis                                       | ✅ |
|  | Topic coherence evaluation                                         | ✅ |
|  | Key concept extraction                                             | ✅ |
|  | Educational level classification                                   | ✅ |
| **Question Generation** | Multi-level cognitive questions (factual, inferential, conceptual) | ✅ |
|  | Adaptive generation based on content quality                       | ✅ |
|  | Question diversity enforcement                                     | ✅ |
|  | Custom question categories                                         | ✅ |
| **Answer Validation** | Factual accuracy verification                                      | ✅ |
|  | RAG-enhanced fact checking                                         | ✅ |
|  | Question clarity evaluation                                        | ✅ |
|  | Answer completeness assessment                                     | ✅ |
| **LLM Integration** | OpenAI API support                                                 | ✅ |
|  | Local LLM support (Ollama, etc.)                                   | ✅ |
|  | Hybrid task routing                                                | ✅ |
|  | Automatic fallback mechanisms                                      | ✅ |
| **Processing Control** | Checkpoint and resume capability                                   | ✅ |
|  | Parallel processing                                                | ✅ |
|  | Progress tracking and reporting                                    | ✅ |
|  | Memory optimization                                                | ✅ |
| **Output Options** | Multiple export formats (JSON, CSV)                                | ✅ |
|  | Metadata inclusion                                                 | ✅ |
|  | Statistics and analytics                                           | ✅ |
|  | RAG retriever creation                                             | ✅ |
| **Extensibility** | Custom document loaders                                            | ✅ |
|  | Custom chunking strategies                                         | ✅ |
|  | Custom validators                                                  | ✅ |
|  | Plugin architecture                                                | ✅ |

---

## Core Capabilities

### Document Processing

#### Multiple Format Support
SemanticQAGen can read and process a variety of document formats including plain text, PDF, Markdown, and DOCX. Each format is handled by specialized loaders that extract content while preserving document structure.

```python
# Process different file types the same way
result_txt = qa_gen.process_document("document.txt")
result_pdf = qa_gen.process_document("document.pdf")
result_md = qa_gen.process_document("document.md")
result_docx = qa_gen.process_document("document.docx")
```

#### Batch Processing
Process multiple files in a single operation. The system can handle directories of mixed document types and will generate separate outputs for each.

```python
# Process all files in a directory
qa_gen.process_batch(
    input_dir="input/",
    output_dir="output/",
    file_types=["txt", "pdf", "docx", "md"]
)
```

#### Automatic Document Type Detection
The system automatically detects document types using both file extensions and content analysis, ensuring the correct loader is used even when file extensions are missing or incorrect.

#### Cross-Page Content Handling
For PDF documents, the system intelligently handles sentences and paragraphs that span across page boundaries, creating a seamless text flow for better semantic analysis.

#### Header/Footer Detection
Automatic detection and optional removal of repeating headers and footers in PDF documents, preventing them from being included in generated questions.

### Content Analysis

#### Semantic Document Chunking
Documents are intelligently broken down into semantically coherent chunks based on content structure rather than arbitrary size limits. This preserves context and produces more meaningful question-answer pairs.

```python
# Configure chunking strategy
config = {
    "chunking": {
        "strategy": "semantic",  # Options: semantic, fixed_size, hybrid
        "target_chunk_size": 1500,
        "preserve_headings": True
    }
}
```

#### Information Density Analysis
Each chunk is analyzed for information density - how rich in facts and teachable content it is. This analysis guides question generation to focus on content-rich sections.

#### Topic Coherence Evaluation
The system evaluates how well each chunk maintains a coherent topic or theme, which helps ensure generated questions relate to a consistent subject area.

#### Key Concept Extraction
Important concepts, terms, and ideas are automatically identified in each chunk, forming the basis for targeted question generation.

#### Educational Level Classification (Optional)
Content is classified by appropriate educational level (elementary, high school, undergraduate, etc.) to help generate questions at suitable complexity levels. You can remove this metadata if it isn't applicable to your dataset.

### Question Generation

#### Multi-level Cognitive Questions
The system generates questions across three cognitive domains:
- **Factual**: Direct recall of information stated in the content
- **Inferential**: Questions requiring connecting multiple pieces of information
- **Conceptual**: Higher-order questions about principles, implications, or broader understanding

```python
# Configure question categories
config = {
    "question_generation": {
        "categories": {
            "factual": {"min_questions": 3, "weight": 1.0},
            "inferential": {"min_questions": 2, "weight": 1.2},
            "conceptual": {"min_questions": 1, "weight": 1.5}
        }
    }
}
```

#### Adaptive Generation
The number and types of questions generated adapt automatically based on content quality. Information-dense chunks yield more questions, while sparse chunks yield fewer.

#### Question Diversity Enforcement
To avoid repetitive or overly similar questions, the system enforces diversity by comparing newly generated questions with existing ones and filtering out duplicates.

#### Custom Question Categories
Users can define custom question categories beyond the standard factual/inferential/conceptual to target specific learning objectives.

### Answer Validation

#### Factual Accuracy Verification
All generated answers are verified against the source content to ensure they do not contain factual errors or hallucinations.

#### RAG-Enhanced Fact Checking
Optional LlamaIndex-based retrieval-augmented validation ensures answers are firmly grounded in the source material. This provides an additional verification layer beyond prompt-based checking.

#### Question Clarity Evaluation
Questions are evaluated for clarity and unambiguity, filtering out poorly formed questions that might confuse learners.

#### Answer Completeness Assessment
The system checks that answers thoroughly address the questions asked, eliminating partial or incomplete responses.

---

## Advanced Features

### RAG-Enhanced Answer Validation

SemanticQAGen's most advanced validation feature uses Retrieval-Augmented Generation (RAG) through LlamaIndex to ensure factual correctness. Unlike many AI question generators that can "hallucinate" incorrect facts, the RAG validation system acts as a powerful factual grounding mechanism.

#### How RAG Validation Works

1. **Document Indexing**: Source documents are indexed using LlamaIndex's document-centric architecture
2. **Answer Verification**: Generated answers are validated against the source material using specialized evaluation metrics
3. **Factual Grounding**: The CorrectnessEvaluator compares answers directly with retrieved context to detect discrepancies
4. **Confidence Scoring**: Each answer receives a factual grounding score (0.0-1.0) indicating reliability

#### Why I Chose LlamaIndex Over Alternatives

I personally prefer LangChain architecture, however LlamaIndex was selected specifically for its document-centric approach, specialized evaluation tools, and embedding management capabilities, making it ideal for factual verification. This is strictly a validation mechanism - it does not pollute or alter the Q/A generation process. The source material remains the single source of truth.

```python
# Enable RAG validation
config = {
    "validation": {
        "factual_accuracy": {
            "enabled": True,
            "threshold": 0.7
        },
        "rag_factual": {
            "enabled": True,
            "model": "gpt-4",
            "threshold": 0.75
        },
        "use_enhanced_validation": True  # Combine traditional and RAG validation
    }
}
```

#### RAG Validator Benefits

- **Source-Grounded Answers**: Ensures all information in answers comes directly from the source material
- **Hallucination Prevention**: Drastically reduces the likelihood of generated answers containing false information
- **Enhanced Quality Control**: Provides a robust verification layer beyond simple prompt engineering

The RAG validation system can be configured to be more or less strict depending on your needs. When using the enhanced combined validation, it benefits from both traditional prompt-based and retrieval-based approaches.

### LLM Integration

#### OpenAI API Support
Full integration with OpenAI with optimized prompting strategies for each task in the pipeline.

#### Local LLM Support
Support for local LLM deployment via Ollama, LM Studio, and similar services, allowing use of models like Llama, Mistral, etc., without requiring external API access.

#### Hybrid Task Routing
Intelligently route different tasks to the most appropriate LLM based on task complexity and model capability. For example, use GPT-4 for complex question generation but a local model for simple validation tasks.

```python
config = {
    "llm_services": {
        "local": {
            "enabled": True,
            "url": "http://localhost:11434/api",
            "model": "mistral:7b",
            "preferred_tasks": ["chunking", "validation"]
        },
        "remote": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-4",
            "preferred_tasks": ["analysis", "generation"]
        }
    }
}
```

#### Automatic Fallback Mechanisms
If a primary LLM service fails, the system automatically tries fallback services, ensuring robustness in production environments.

### Processing Control

#### Checkpoint and Resume Capability
Processing can be interrupted and resumed later using a checkpoint system. This is essential for large documents or when processing must be paused.

```python
config = {
    "processing": {
        "enable_checkpoints": True,
        "checkpoint_dir": "./checkpoints",
        "checkpoint_interval": 10  # Save every 10 chunks
    }
}
```

#### Parallel Processing
Multi-threaded processing of documents and chunks with configurable concurrency levels to maximize throughput on multi-core systems.

#### Progress Tracking and Reporting
Detailed progress reporting during processing, with support for both simple console output and rich interactive displays.

#### Memory Optimization
Smart memory management techniques to handle very large documents without exhausting system resources.

### Output Options

#### Multiple Export Formats
Export question-answer pairs in various formats including JSON, CSV, with customizable formatting options.

```python
# Save questions in different formats
qa_gen.save_questions(result, "questions_output", format_name="json")
qa_gen.save_questions(result, "questions_output", format_name="csv")
```

#### Metadata Inclusion
Include rich metadata about source documents, generation parameters, and validation results with the generated questions.

#### Statistics and Analytics
Comprehensive statistics about generated questions, including category distribution, validation success rates, and content coverage.

#### RAG Retriever Creation
Generated questions and answers can be exported as a LlamaIndex retriever for use in downstream applications.

```python
# Create a retriever from generated QA pairs
retriever = qa_gen.create_qa_retriever(result, api_key="YOUR_OPENAI_API_KEY")
response = retriever.retrieve("How does photosynthesis work?")
```

---

## Project File Organization
```txt
.
├── 📁 semantic_qa_gen
│   ├── 📄 __init__.py
│   ├── 📄 version.py
│   ├── 📄 semantic_qa_gen.py
│   ├── 📁 chunking
│   │   ├── 📄 analyzer.py
│   │   ├── 📄 engine.py
│   │   └── 📁 strategies
│   │       ├── 📄 base.py
│   │       ├── 📄 fixed_size.py
│   │       ├── 📄 nlp_helpers.py
│   │       └── 📄 semantic.py
│   ├── 📁 cli
│   │   ├── 📄 __init__.py
│   │   └── 📄 commands.py
│   ├── 📁 config
│   │   ├── 📄 manager.py
│   │   └── 📄 schema.py
│   ├── 📁 document
│   │   ├── 📄 models.py
│   │   ├── 📄 processor.py
│   │   └── 📁 loaders
│   │       ├── 📄 base.py
│   │       ├── 📄 docx.py
│   │       ├── 📄 markdown.py
│   │       ├── 📄 pdf.py
│   │       └── 📄 text.py
│   ├── 📁 llm
│   │   ├── 📄 router.py
│   │   ├── 📄 service.py
│   │   ├── 📁 adapters
│   │   │   ├── 📄 local.py
│   │   │   └── 📄 remote.py
│   │   └── 📁 prompts
│   │       └── 📄 manager.py
│   ├── 📁 output
│   │   ├── 📄 __init__.py
│   │   ├── 📄 formatter.py
│   │   └── 📁 adapters
│   │       ├── 📄 __init__.py
│   │       ├── 📄 csv.py
│   │       └── 📄 json.py
│   ├── 📁 pipeline
│   │   └── 📄 semantic.py
│   ├── 📁 question
│   │   ├── 📄 generator.py
│   │   ├── 📄 processor.py
│   │   └── 📁 validation
│   │       ├── 📄 base.py
│   │       ├── 📄 diversity.py
│   │       ├── 📄 engine.py
│   │       └── 📄 factual.py
│   └── 📁 utils
│       ├── 📄 checkpoint.py
│       ├── 📄 error.py
│       ├── 📄 logging.py
│       └── 📄 progress.py
├── 📄 setup.py
└── 📁 tests
    ├── 📄 conftest.py
    ├── 📄 test_chunking.py
    ├── 📄 test_config.py
    ├── 📄 test_document_processor.py
    ├── 📄 test_llm_service.py
    ├── 📄 test_main.py
    ├── 📄 test_pipeline.py
    ├── 📄 test_question_generation.py
    ├── 📄 test_utils.py
    ├── 📄 test_validation.py
    └── 📁 integration
        └── 📄 test_minimal_pipeline.py
```

## Architecture

SemanticQAGen implements a modular pipeline architecture with clearly defined components and interfaces:

```
                                   ARCHITECTURE
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                       ┌─────────────────────────────────┐                     │
│                       │       SemanticQAGen Class       │                     │
│                       │      (Main User Interface)      │                     │
│                       └─────────────┬───────────────────┘                     │
│                                     │                                         │
│                                     ▼                                         │
│           ┌────────────────────────────────────────────────────┐              │
│           │              SemanticPipeline Orchestrator         │              │
│           └┬────────────────┬───────────────────┬─────────────┬┘              │
│            │                │                   │             │               │
│  ┌─────────▼──────────┐     │    ┌──────────────▼───────────┐ │               │
│  │  Document Manager  │     │    │   Chunking & Analysis    │ │               │
│  └┬────────────────┬──┘     │    └┬─────────────────────┬───┘ │               │
│   │                │        │     │                     │     │               │
│┌──▼───────┐   ┌────▼────┐   │  ┌──▼───────┐       ┌────▼────┐ │               │
││ Document │   │Document │   │  │ Chunking │       │Semantic │ │               │
││ Loaders  │   │Processor│   │  │ Engine   │       │Analyzer │ │               │
│└──────────┘   └─────────┘   │  └──────────┘       └─────────┘ │               │
│                             │                                 │               │
│      ┌─────────────────────────────────────────────────┐      │               │
│      │               LLM Service Router                │      │               │
│      │                                                 │      │               │
│      │  ┌────────────────┐         ┌────────────────┐  │      │               │
│      │  │ Remote LLM     │         │ Local LLM      │  │      │               │
│      │  │ (OpenAI, etc.) │         │ (Ollama, etc.) │  │      │               │
│      │  └────────────────┘         └────────────────┘  │      │               │
│      └─────────────────────────────────────────────────┘      │               │
│                             │                                 │               │
│  ┌────────────────────────┐ │ ┌───────────────────────────────▼────────┐      │
│  │  Question Generator    │ │ │         Validation Engine              │      │
│  │                        │◄┼─┼────┐                                   │      │
│  │  ┌──────────────────┐  │ │ │    │                                   │      │
│  │  │Category: Factual │  │ │ │    │  ┌─────────────┐ ┌──────────────┐ │      │
│  │  └──────────────────┘  │ │ │    ├─►│ Traditional │ │  RAG-based   │ │      │
│  │  ┌──────────────────┐  │ │ │    │  │ Validators  │ │  Validator   │ │      │
│  │  │Cat: Inferential  │  │ │ │    │  └─────────────┘ └──────────────┘ │      │
│  │  └──────────────────┘  │ │ │    │                                   │      │
│  │  ┌──────────────────┐  │ │ │    │  ┌──────────────────────────────┐ │      │
│  │  │Cat: Conceptual   │  │ │ │    └─►│  Enhanced Validator          │ │      │
│  │  └──────────────────┘  │ │ │       └──────────────────────────────┘ │      │
│  └─────────┬──────────────┘ │ └───────────────────────────────────────┬┘      │
│            │                │                                         │       │
│            └────────────────┼─────────────────────────────────────────┘       │
│                             │                                                 │
│          ┌──────────────────▼─────────────────────┐                           │
│          │           Output Formatter             │                           │
│          │                                        │                           │
│          │  ┌─────────────┐    ┌────────────────┐ │                           │
│          │  │ JSON Adapter│    │  CSV Adapter   │ │                           │
│          │  └─────────────┘    └────────────────┘ │                           │
│          └────────────────────────────────────────┘                           │
│                             │                                                 │
│           ┌─────────────────▼────────────────────┐                            │
│           │           Output Results             │                            │
│           │  • Questions & Answers               │                            │
│           │  • Document Metadata                 │                            │
│           │  • Statistics                        │                            │
│           └──────────────────────────────────────┘                            │
│                                                                               │
│                      ┌─────────────────────────────┐                          │
│                      │     Checkpoint Manager      │                          │
│                      │   (Resume Capabilities)     │                          │
│                      └─────────────────────────────┘                          │
│                                                                               │
│                      ┌─────────────────────────────┐                          │
│                      │     Progress Reporter       │                          │
│                      │   (Processing Feedback)     │                          │
│                      └─────────────────────────────┘                          │
└───────────────────────────────────────────────────────────────────────────────┘

                                 DATA FLOW
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌───────────┐      ┌───────────┐     ┌───────────┐      ┌───────────┐        │
│  │  Document │      │ Document  │     │   List    │      │Chunk List │        │
│  │  Files    │─────►│   Model   │────►│ of Chunks │─────►│w/Analysis │        │
│  └───────────┘      └───────────┘     └───────────┘      └───────────┘        │
│                                                                │              │
│                                                                ▼              │
│                                         ┌────────────────────────────────┐    │
│  ┌───────────┐     ┌────────────┐       │ Question-Answer Generation     │    │
│  │   Final   │     │  Validated │       │ ┌────────────┐ ┌────────────┐  │    │
│  │  Output   │◄────┤ Question   │◄──────┤ │ Question 1 │ │ Question 2 │  │    │
│  │           │     │  List      │       │ └────────────┘ └────────────┘  │    │
│  └───────────┘     └────────────┘       └────────────────────────────────┘    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

                           VALIDATION SUBSYSTEM
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│   ┌───────────────┐     ┌─────────────────────────────────────────────────┐   │
│   │ Generated     │     │             Validation Engine                   │   │
│   │ Questions     │────►│                                                 │   │
│   └───────────────┘     │    ┌────────────────────┐  ┌─────────────────┐  │   │
│                         │    │  Traditional Flow  │  │   RAG Flow      │  │   │
│                         │    │                    │  │                 │  │   │
│                         │    │ ┌──────────────┐   │  │ ┌─────────────┐ │  │   │
│                         │    │ │  Factual     │   │  │ │ Source      │ │  │   │
│                         │    │ │  Accuracy    │   │  │ │ Document    │ │  │   │
│                         │    │ └──────────────┘   │  │ │ Index       │ │  │   │
│                         │    │ ┌──────────────┐   │  │ └──────┬──────┘ │  │   │
│                         │    │ │  Answer      │   │  │        │        │  │   │
│                         │    │ │ Completeness │   │  │ ┌──────▼──────┐ │  │   │
│                         │    │ └──────────────┘   │  │ │ LlamaIndex  │ │  │   │
│                         │    │ ┌──────────────┐   │  │ │ Correctness │ │  │   │
│                         │    │ │  Question    │   │  │ │ Evaluator   │ │  │   │
│                         │    │ │  Clarity     │   │  │ └──────┬──────┘ │  │   │
│                         │    │ └──────────────┘   │  │        │        │  │   │
│                         │    │ ┌──────────────┐   │  │ ┌──────▼──────┐ │  │   │
│                         │    │ │  Diversity   │   │  │ │ Accuracy    │ │  │   │
│                         │    │ │  Check       │   │  │ │ Score       │ │  │   │
│                         │    │ └──────────────┘   │  │ └─────────────┘ │  │   │
│                         │    └────────────────────┘  └─────────────────┘  │   │
│                         │                                                 │   │
│                         │            ┌────────────────────┐               │   │
│                         │            │   Combine Results  │               │   │
│                         │            └─────────┬──────────┘               │   │
│                         │                      │                          │   │
│                         └──────────────────────┼──────────────────────────┘   │
│                                                │                              │
│   ┌───────────────┐                  ┌─────────▼──────────┐                   │
│   │ Final Valid   │◄─────────────────┤ Validation Results │                   │
│   │ Questions     │                  └────────────────────┘                   │
│   └───────────────┘                                                           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

                             BATCH PROCESSING 
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│   ┌─────────────┐         ┌───────────────────────┐      ┌──────────────┐     │
│   │ Input       │         │ Batch Processor       │      │ Checkpoint   │     │
│   │ Directory   │─────────►                       │◄────►│ Store        │     │
│   └─────────────┘         │ ┌────────────────┐    │      └──────────────┘     │
│                           │ │ File Type      │    │                           │
│                           │ │ Detection      │    │                           │
│                           │ └────────────────┘    │                           │
│                           │ ┌────────────────┐    │                           │
│                           │ │ Processing     │    │                           │
│                           │ │ Queue          │    │                           │
│                           │ └────────────────┘    │                           │
│                           │ ┌────────────────┐    │      ┌──────────────┐     │
│                           │ │ Parallel       ├────────────► File 1      │     │
│                           │ │ Scheduler      │    │      │ Processing   │     │
│                           │ └──┬────────────┬┘    │      └──────────────┘     │
│                           │    │            │     │      ┌──────────────┐     │
│                           │    │            ├────────────► File 2       │     │
│                           │    │            │     │      │ Processing   │     │
│                           │    └────────────┘     │      └──────────────┘     │
│                           └───────────────────────┘                           │
│                                      │                                        │
│                          ┌───────────▼───────────┐                            │
│                          │ Results Aggregator    │                            │
│                          └───────────┬───────────┘                            │
│                                      │                                        │
│   ┌─────────────┐         ┌──────────▼───────────┐                            │
│   │ Output      │◄────────┤ Final Outputs        │                            │
│   │ Directory   │         └──────────────────────┘                            │
│   └─────────────┘                                                             │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```
### Core Components

1. **Document Processor**: Handles document loading and preprocessing
2. **Chunking Engine**: Splits documents into semantically coherent chunks
3. **Semantic Analyzer**: Evaluates information density and question potential
4. **Question Generator**: Creates diverse questions based on content analysis
5. **Validation Engine**: Ensures question quality and diversity
6. **Output Formatter**: Formats and exports the generated Q&A pairs

### Processing Pipeline

```
Document → Chunks → Analysis → Questions → Validation → Output
```

The pipeline implements a two-phase approach:
1. **Analysis Phase**: Document is processed, chunked, and analyzed for content quality
2. **Generation Phase**: Questions are generated, validated, and formatted based on analysis

---

## Configuration

SemanticQAGen uses a hierarchical YAML configuration system with schema validation.

### Configuration File Example

```yaml
# SemanticQAGen configuration
version: 1.0

# Document processing settings
document:
  loaders:
    text:
      enabled: true
      encoding: utf-8
    pdf:
      enabled: true
      extract_images: false
      ocr_enabled: false
      detect_headers_footers: true
    markdown:
      enabled: true
      extract_metadata: true
    docx:
      enabled: true
      extract_tables: true

# Chunking settings
chunking:
  strategy: semantic
  target_chunk_size: 1500
  overlap_size: 150
  preserve_headings: true
  min_chunk_size: 500
  max_chunk_size: 2500

# LLM services configuration
llm_services:
  local:
    enabled: true
    url: "http://localhost:11434/api"
    model: "mistral:7b"
    preferred_tasks: [chunking, validation]
    timeout: 60
  remote:
    enabled: true
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    preferred_tasks: [analysis, generation]
    timeout: 120
    rate_limit_tokens: 90000
    rate_limit_requests: 100

# Question generation settings
question_generation:
  max_questions_per_chunk: 10
  adaptive_generation: true
  categories:
    factual:
      min_questions: 2
      weight: 1.0
    inferential:
      min_questions: 2
      weight: 1.2
    conceptual:
      min_questions: 1
      weight: 1.5
  diversity:
    required: true
    min_similarity_threshold: 0.75

# Validation settings
validation:
  factual_accuracy:
    enabled: true
    threshold: 0.8
  answer_completeness:
    enabled: true
    threshold: 0.8
  question_clarity:
    enabled: true
    threshold: 0.8
  diversity:
    enabled: true
    similarity_metric: cosine
  rag_factual:
    enabled: true
    model: "gpt-4"
    threshold: 0.7
  use_enhanced_validation: true

# Batch processing settings
processing:
  concurrency: 3
  enable_checkpoints: true
  checkpoint_interval: 10
  checkpoint_dir: "./checkpoints"
  log_level: "INFO"
  batch:
    enabled: true
    input_dir: "./documents"
    output_dir: "./output"
    supported_types: ["txt", "pdf", "md", "docx"]
    parallel_processing: true
    max_concurrent_files: 2
    continue_on_error: true
    track_processed_files: true
    skip_completed_files: true
    resume_strategy: "auto-detect"

# Output settings
output:
  format: json
  include_metadata: true
  include_statistics: true
  output_dir: "./output"
  json_indent: 2
  csv_delimiter: ","
```

### Example JSON Output
In the JSON output format, the metadata for each question/answer pair is integrated directly with the question/answer data. Each question is represented as a single JSON object that contains all its associated information, including the question text, answer text, category, and all metadata

```json
{
  "questions": [
    {
      "id": "q_c05a3e9b",
      "text": "What is photosynthesis?",
      "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It converts light energy into chemical energy, specifically glucose, while releasing oxygen as a byproduct.",
      "category": "factual",
      "chunk_id": "chunk_28a1e5",
      "metadata": {
        "source_page": 12,
        "confidence_score": 0.92,
        "generated_at": "2025-03-15T14:23:47Z",
        "validation_scores": {
          "factual_accuracy": 0.95,
          "question_clarity": 0.89,
          "answer_completeness": 0.91,
          "rag_factual_accuracy": 0.97
        },
        "topic": "Biology",
        "educational_level": "High School",
        "keywords": ["photosynthesis", "plants", "chlorophyll", "glucose", "energy conversion"]
      }
    },
```
    // Additional questions...


### Environment Variables

Configuration values can be specified using environment variables:

```yaml
llm_services:
  remote:
    api_key: ${OPENAI_API_KEY}
```

### Configuration Layering

Configuration is resolved in the following order:
1. Default values
2. Configuration file
3. Environment variables
4. Command-line arguments
5. Programmatic overrides

---

## API Reference

### Class Heirarchy

```
                                 CLASS HIERARCHY
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                         ┌─────────────────────────┐                           │
│                         │     SemanticQAGen       │                           │
│                         │                         │                           │
│                         │ - process_document()    │                           │
│                         │ - process_batch()       │                           │
│                         │ - save_questions()      │                           │
│                         │ - create_qa_retriever() │                           │
│                         └────────────┬────────────┘                           │
│                                      │                                        │
│                                      │                                        │
│                                      │                                        │
│                 ┌────────────────────▼──────────────────┐                     │
│                 │         SemanticPipeline              │                     │
│                 │                                       │                     │
│                 │ - build_pipeline()                    │                     │
│                 │ - run_pipeline()                      │                     │
│                 └───┬─────────────┬────────────┬────────┘                     │
│                     │             │            │                              │
│      ┌──────────────┘    ┌────────┘      ┌─────┘                              │
│      │                   │               │                                    │
│      ▼                   ▼               ▼                                    │
│ ┌──────────────┐  ┌─────────────┐  ┌───────────────┐                          │
│ │ConfigManager │  │EventEmitter │  │ProgressTracker│                          │
│ └──────────────┘  └─────────────┘  └───────────────┘                          │
│                                                                               │
│                          DOCUMENT PROCESSING                                  │
│                                                                               │
│  ┌───────────────────────┐                                                    │
│  │   DocumentProcessor   │                                                    │
│  │                       │                                                    │
│  │ - load_document()     │                                                    │
│  │ - extract_sections()  │                                                    │
│  └───────────┬───────────┘                                                    │
│              │                                                                │
│      ┌───────┴──────────────────────────┐                                     │
│      │                                  │                                     │
│      ▼                                  ▼                                     │
│ ┌──────────────┐                ┌─────────────────┐                           │
│ │DocumentLoader│◄───────────────┤  BaseLoader     │                           │
│ │Manager       │                │  (Abstract)     │                           │
│ └──────────────┘                └────────┬────────┘                           │
│                                          │                                    │
│                        ┌─────────────────┼────────────────┬───────────────┐   │
│                        │                 │                │               │   │
│                        ▼                 ▼                ▼               ▼   │
│              ┌───────────────┐  ┌─────────────┐  ┌─────────────┐ ┌───────────┐│
│              │TextFileLoader │  │PDFLoader    │  │DocxLoader   │ │MarkdownLdr││
│              └───────────────┘  └─────────────┘  └─────────────┘ └───────────┘│
│                                                                               │
│                                 CHUNKING                                      │
│                                                                               │
│  ┌─────────────────────┐                                                      │
│  │   ChunkingEngine    │                                                      │
│  │                     │                                                      │
│  │ - chunk_document()  │                                                      │
│  │ - process_chunks()  │                                                      │
│  └─────────┬───────────┘                                                      │
│            │                                                                  │
│            │                                                                  │
│            ▼                                                                  │
│  ┌───────────────────────┐                                                    │
│  │BaseChunkingStrategy   │                                                    │
│  │(Abstract)             │                                                    │
│  └───────────┬───────────┘                                                    │
│              │                                                                │
│    ┌─────────┼────────────┬───────────────────┐                               │
│    │         │            │                   │                               │
│    ▼         ▼            ▼                   ▼                               │
│┌───────────┐ │ ┌────────────┐ ┌───────────────┐ ┌───────────────┐             │
││Semantic   │ │ │FixedSize   │ │RecursiveChunk │ │HybridChunking │             │
││Chunking   │ │ │Chunking    │ │Strategy       │ │Strategy       │             │
│└───────────┘ │ └────────────┘ └───────────────┘ └───────────────┘             │
│              │                                                                │
│              ▼                                                                │
│  ┌───────────────────────┐                                                    │
│  │    SemanticAnalyzer   │                                                    │
│  │                       │                                                    │
│  │ - analyze_chunk()     │                                                    │
│  │ - get_content_score() │                                                    │
│  └───────────────────────┘                                                    │
│                                                                               │
│                           LLM SERVICES                                        │
│                                                                               │
│  ┌──────────────────────┐                                                     │
│  │   LLMServiceRouter   │                                                     │
│  │                      │                                                     │
│  │ - route_request()    │                                                     │
│  │ - select_service()   │                                                     │
│  └──────────┬───────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                      │
│  │    BaseLLMService   │                                                      │
│  │     (Abstract)      │                                                      │
│  └─────────┬───────────┘                                                      │
│            │                                                                  │
│      ┌─────┴──────┐                                                           │
│      │            │                                                           │
│      ▼            ▼                                                           │
│ ┌─────────────┐ ┌─────────────┐                                               │
│ │OpenAIService│ │LocalLLMSvc  │                                               │
│ └─────────────┘ └─────────────┘                                               │
│                                                                               │
│                     QUESTION GENERATION                                       │
│                                                                               │
│  ┌────────────────────────┐                                                   │
│  │  QuestionGenerator     │                                                   │
│  │                        │                                                   │
│  │ - generate_questions() │                                                   │
│  │ - create_prompts()     │                                                   │
│  └───────────┬────────────┘                                                   │
│              │                                                                │
│              ▼                                                                │
│  ┌────────────────────────┐    ┌─────────────────────┐                        │
│  │  CategoryHandler       │    │   PromptTemplate    │                        │
│  │                        │    │                     │                        │
│  │ - get_category_prompts()    │ - format()          │                        │
│  │ - get_question_count() │    │ - get_variables()   │                        │
│  └────────────────────────┘    └─────────────────────┘                        │
│                                                                               │
│                        VALIDATION                                             │
│                                                                               │
│  ┌────────────────────┐                                                       │
│  │ ValidationEngine   │                                                       │
│  │                    │                                                       │
│  │ - validate()       │                                                       │
│  │ - combine_results()│                                                       │
│  └────────┬───────────┘                                                       │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─────────────────┐                                                          │
│  │  BaseValidator  │                                                          │
│  │   (Abstract)    │                                                          │
│  └────────┬────────┘                                                          │
│           │                                                                   │
│     ┌─────┼────────┬─────────────┬───────────────┐                            │
│     │     │        │             │               │                            │
│     ▼     ▼        ▼             ▼               ▼                            │
│ ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────────┐ │
│ │FactualAcc  │ │Question  │ │AnswerComp  │ │Diversity   │ │RAGFactualValid  │ │
│ │Validator   │ │Clarity   │ │Validator   │ │Validator   │ │                 │ │
│ └────────────┘ └──────────┘ └────────────┘ └────────────┘ └────────┬────────┘ │
│                                                                    │          │
│                                                                    ▼          │
│                                                           ┌─────────────────┐ │
│                                                           │LlamaIndexAdapter│ │
│                                                           └─────────────────┘ │
│                                                                               │
│                          OUTPUT                                               │
│                                                                               │
│  ┌────────────────────┐                                                       │
│  │  OutputFormatter   │                                                       │
│  │                    │                                                       │
│  │ - format_output()  │                                                       │
│  │ - save_output()    │                                                       │
│  └────────┬───────────┘                                                       │
│           │                                                                   │
│           ▼                                                                   │
│  ┌────────────────────┐                                                       │
│  │  BaseOutputAdapter │                                                       │
│  │     (Abstract)     │                                                       │
│  └────────┬───────────┘                                                       │
│           │                                                                   │
│      ┌────┴───────┐                                                           │
│      │            │                                                           │
│      ▼            ▼                                                           │
│ ┌───────────┐  ┌──────────┐                                                   │
│ │JSONAdapter│  │CSVAdapter│                                                   │
│ └───────────┘  └──────────┘                                                   │
│                                                                               │
│                           MODELS                                              │
│                                                                               │
│  ┌────────────────┐  ┌────────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │  Document      │  │  Chunk     │   │  Question    │   │ValidationResult│   │
│  │ (dataclass)    │  │ (dataclass)│   │ (dataclass)  │   │  (dataclass)   │   │
│  └────────────────┘  └────────────┘   └──────────────┘   └────────────────┘   │
│                                                                               │
│  ┌────────────────┐  ┌────────────┐   ┌──────────────┐                        │
│  │DocumentMetadata│  │Section     │   │ChunkAnalysis │                        │
│  │  (dataclass)   │  │(dataclass) │   │ (dataclass)  │                        │
│  └────────────────┘  └────────────┘   └──────────────┘                        │
│                                                                               │
│                      BATCH PROCESSING                                         │
│                                                                               │
│  ┌───────────────────┐                                                        │
│  │   BatchProcessor  │                                                        │
│  │                   │                                                        │
│  │ - process_batch() │                                                        │
│  │ - queue_files()   │                                                        │
│  └────────┬──────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌────────────────────┐   ┌────────────────────┐                              │
│  │ ParallelScheduler  │   │ CheckpointManager  │                              │
│  │                    │   │                    │                              │
│  │ - run_parallel()   │   │ - save_checkpoint()│                              │
│  │ - handle_results() │   │ - load_checkpoint()│                              │
│  └────────────────────┘   └────────────────────┘                              │
│                                                                               │
│                         UTILITIES                                             │
│                                                                               │
│  ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────┐     │
│  │ TokenCounter  │   │ Logger     │   │TextCleaner │   │ExceptionHandler│     │
│  └───────────────┘   └────────────┘   └────────────┘   └────────────────┘     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Main Class: `SemanticQAGen`

```python
class SemanticQAGen:
    """Main interface for generating question-answer pairs from text documents."""
    
    def __init__(self, config_path: Optional[str] = None, 
                config_dict: Optional[Dict[str, Any]] = None,
                verbose: bool = False):
        """Initialize SemanticQAGen with optional configuration."""
        
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document to generate question-answer pairs.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dictionary containing questions, statistics, and metadata.
        """
        
    def process_batch(self, input_dir: Optional[str] = None,
                     output_dir: Optional[str] = None,
                     file_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple documents from a directory.
        
        Args:
            input_dir: Directory containing documents to process.
            output_dir: Directory to save outputs.
            file_types: List of file types to process (e.g., ["txt", "pdf"]).
            
        Returns:
            Dictionary mapping filenames to results.
        """
        
    def save_questions(self, result: Dict[str, Any], 
                      output_path: str,
                      format_name: Optional[str] = None) -> str:
        """
        Save generated questions to a file.
        
        Args:
            result: Results from process_document.
            output_path: Path where to save the output.
            format_name: Format to save in (json, csv, etc.).
            
        Returns:
            Path to the saved file.
        """
        
    def create_default_config_file(self, output_path: str) -> None:
        """Create a default configuration file."""
        
    def create_qa_retriever(self, result: Dict[str, Any], api_key: Optional[str] = None) -> Any:
        """
        Create a LlamaIndex retriever from generated QA pairs.
        
        Args:
            result: Results from process_document.
            api_key: Optional OpenAI API key.
            
        Returns:
            LlamaIndex retriever object for RAG applications.
        """
```

For additional API details, see the complete [API Documentation](https://semanticqagen.readthedocs.io/).

---

## CLI Reference

SemanticQAGen provides a comprehensive command-line interface:

### Main Commands

```
semantic-qa-gen process <document> [-o OUTPUT] [-f {json,csv}] [-c CONFIG] [-v]
semantic-qa-gen process-batch [input_dir] [-o OUTPUT_DIR] [--types TYPES] [-c CONFIG]
semantic-qa-gen init-config <output> [-i]
semantic-qa-gen interactive
semantic-qa-gen version
```

### Command Details

```
process             Process a document and generate questions
  document          Path to the document file
  -o, --output      Path for output file
  -f, --format      Output format (json, csv)
  -c, --config      Path to config file
  -v, --verbose     Enable verbose output

process-batch       Process multiple documents from a directory
  input_dir         Input directory containing documents
  -o, --output-dir  Output directory for results
  --types           File types to process (comma-separated)
  --exclude         Files to exclude (comma-separated)
  -c, --config      Path to config file
  --no-checkpoints  Disable checkpointing
  -v, --verbose     Enable verbose output

init-config         Create a default configuration file
  output            Path for the config file
  -i, --interactive Create config interactively

interactive         Run in interactive mode
version             Show the version and exit
```

### Examples

```bash
# Process a PDF document
semantic-qa-gen process document.pdf -o questions_output

# Process all files in a directory
semantic-qa-gen process-batch input_docs/ -o output_results/

# Process specific file types
semantic-qa-gen process-batch input_docs/ --types pdf,txt -o output_results/

# Process with a specific configuration
semantic-qa-gen process document.txt --config my_config.yml --format csv

# Create a default configuration file
semantic-qa-gen init-config config.yml

# Create a configuration file interactively
semantic-qa-gen init-config config.yml --interactive
```

---

## Usage Examples

### Basic Document Processing

```python
from semantic_qa_gen import SemanticQAGen

# Initialize with default settings
qa_gen = SemanticQAGen()

# Process a document
result = qa_gen.process_document("path/to/document.txt")

# Save the questions to a JSON file
qa_gen.save_questions(result, "qa_pairs.json")

# Display stats
print(f"Generated {len(result['questions'])} questions")
print(f"Factual questions: {result['statistics']['categories']['factual']}")
print(f"Inferential questions: {result['statistics']['categories']['inferential']}")
print(f"Conceptual questions: {result['statistics']['categories']['conceptual']}")
```

### Batch Processing with Checkpoints

```python
from semantic_qa_gen import SemanticQAGen

# Configuration for batch processing with checkpoints
config = {
    "processing": {
        "enable_checkpoints": True,
        "batch": {
            "input_dir": "./documents",
            "output_dir": "./results",
            "supported_types": ["txt", "pdf", "md", "docx"],
            "continue_on_error": True,
            "skip_completed_files": True
        }
    }
}

# Initialize with batch processing config
qa_gen = SemanticQAGen(config_dict=config)

# Process all files in the input directory
# Will resume from previous checkpoints if available
batch_results = qa_gen.process_batch()

print(f"Processed {len(batch_results)} documents")

# You can also specify directories explicitly
batch_results = qa_gen.process_batch(
    input_dir="./other_documents",
    output_dir="./other_results",
    file_types=["pdf", "txt"]
)
```

### Using Local and Remote LLMs Together

```python
from semantic_qa_gen import SemanticQAGen

# Configuration for hybrid LLM setup
config = {
    "llm_services": {
        "local": {
            "enabled": True,
            "url": "http://localhost:11434/api",
            "model": "mistral:7b",
            "preferred_tasks": ["validation"]
        },
        "remote": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "YOUR_API_KEY",
            "preferred_tasks": ["analysis", "generation"]
        }
    }
}

# Initialize with hybrid LLM config
qa_gen = SemanticQAGen(config_dict=config)

# Process document using hybrid approach
# - Local model will handle chunking and validation
# - Remote model will handle analysis and question generation
result = qa_gen.process_document("document.pdf")
```

### RAG-Enhanced Validation

```python
from semantic_qa_gen import SemanticQAGen

# Configuration focusing on validation
config = {
    "validation": {
        "factual_accuracy": {
            "enabled": True,
            "threshold": 0.7
        },
        "rag_factual": {
            "enabled": True,
            "model": "gpt-4",
            "threshold": 0.75,
            "strict_mode": True
        },
        "use_enhanced_validation": True  # Combined approach
    }
}

# Process document with enhanced validation
qa_gen = SemanticQAGen(config_dict=config)
result = qa_gen.process_document("document.pdf")

# Create a retriever from the validated questions
retriever = qa_gen.create_qa_retriever(result, api_key="YOUR_OPENAI_API_KEY")

# Use the retriever in a separate application
response = retriever.retrieve("What is the most important concept?")
```

### Custom Question Categories

```python
config = {
    "question_generation": {
        "max_questions_per_chunk": 12,
        "categories": {
            "factual": {
                "min_questions": 4,  # Prefer more factual questions
                "weight": 1.5
            },
            "inferential": {
                "min_questions": 3,
                "weight": 1.2
            },
            "conceptual": {
                "min_questions": 2,
                "weight": 1.0
            },
            "applied": {  # Custom category - practical applications
                "min_questions": 3,
                "weight": 1.3
            }
        }
    }
}

qa_gen = SemanticQAGen(config_dict=config)
```

### Processing Large Documents Efficiently

```python
from semantic_qa_gen import SemanticQAGen

config = {
    "chunking": {
        "strategy": "semantic",
        "target_chunk_size": 1200,  # Smaller chunks
        "max_chunk_size": 1800
    },
    "processing": {
        "concurrency": 2,  # Lower concurrency to reduce memory usage
        "enable_checkpoints": True,
        "checkpoint_interval": 5  # Save checkpoints frequently
    }
}

qa_gen = SemanticQAGen(config_dict=config)
result = qa_gen.process_document("large_document.pdf")
```

---

## Extension

SemanticQAGen is designed to be easily extended with custom components.

### Creating a Custom Document Loader

```python
from semantic_qa_gen.document.loaders.base import BaseLoader
from semantic_qa_gen.document.models import Document, DocumentType, DocumentMetadata
from semantic_qa_gen.utils.error import DocumentError

class CustomFileLoader(BaseLoader):
    """Loader for custom file format."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
    def load(self, path: str) -> Document:
        """Load a document from a custom file format."""
        if not self.supports_type(path):
            raise DocumentError(f"Unsupported file type: {path}")
            
        # Implementation for loading custom format
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Create and return document
        return Document(
            content=content,
            doc_type=DocumentType.TEXT,
            path=path,
            metadata=self.extract_metadata(path)
        )
        
    def supports_type(self, file_path: str) -> bool:
        """Check if this loader supports the given file type."""
        _, ext = os.path.splitext(file_path.lower())
        return ext == '.custom'
        
    def extract_metadata(self, path: str) -> DocumentMetadata:
        """Extract metadata from the custom file."""
        # Implementation for extracting metadata
        return DocumentMetadata(
            title=os.path.basename(path),
            source=path
        )
```

### Registering a Custom Loader

```python
from semantic_qa_gen import SemanticQAGen
from semantic_qa_gen.document.processor import DocumentProcessor

# Create your custom loader
custom_loader = CustomFileLoader()

# Initialize SemanticQAGen
qa_gen = SemanticQAGen()

# Get the document processor
doc_processor = qa_gen.pipeline.document_processor

# Register your custom loader
doc_processor.loaders.append(custom_loader)

# Now you can process custom file formats
result = qa_gen.process_document("document.custom")
```

### Creating a Custom Validator

```python
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.document.models import Question, Chunk

class CustomValidator(BaseValidator):
    """Custom validator for specialized validation logic."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.7)
    
    async def validate(self, question: Question, chunk: Chunk) -> ValidationResult:
        """Implement custom validation logic."""
        # Custom validation implementation
        score = 0.8  # Example score
        
        return ValidationResult(
            question_id=question.id,
            is_valid=score >= self.threshold,
            scores={"custom_score": score},
            reasons=[f"Custom validation: {score:.2f}"],
            suggested_improvements=None if score >= self.threshold else "Suggestion for improvement"
        )
```

### Creating a Custom Chunking Strategy

```python
from semantic_qa_gen.chunking.strategies.base import BaseChunkingStrategy
from semantic_qa_gen.document.models import Document, Section, Chunk

class CustomChunkingStrategy(BaseChunkingStrategy):
    """Custom strategy for document chunking."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.target_size = self.config.get('target_chunk_size', 1500)
        
    def chunk_document(self, document: Document, sections: List[Section]) -> List[Chunk]:
        """Break a document into chunks using a custom strategy."""
        chunks = []
        
        # Custom implementation of chunking algorithm
        
        return chunks
```

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems

**Issue**: Missing dependencies when installing
**Solution**: Install with the appropriate extra dependencies:
```bash
pip install semantic-qa-gen[full]
```

**Issue**: Conflicts with existing packages
**Solution**: Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install semantic-qa-gen
```

#### Processing Issues

**Issue**: Out of memory errors with large documents
**Solution**: Adjust chunking and processing settings:
```python
config = {
    "chunking": {
        "target_chunk_size": 1000,  # Smaller chunks
        "max_chunk_size": 1500
    },
    "processing": {
        "concurrency": 1,  # Reduce concurrency
        "enable_checkpoints": True,
        "checkpoint_interval": 3  # More frequent checkpoints
    }
}
```

**Issue**: Slow processing with PDF documents
**Solution**: Disable unnecessary PDF features:
```python
config = {
    "document": {
        "loaders": {
            "pdf": {
                "extract_images": False,
                "ocr_enabled": False,
                "fix_cross_page_sentences": False
            }
        }
    }
}
```

#### LLM Service Issues

**Issue**: OpenAI rate limits
**Solution**: Adjust rate limiting settings:
```python
config = {
    "llm_services": {
        "remote": {
            "rate_limit_tokens": 60000,  # Reduce token usage
            "rate_limit_requests": 50  # Reduce requests per minute
        }
    }
}
```

**Issue**: Local LLM not responding
**Solution**: Check connection settings and increase timeout:
```python
config = {
    "llm_services": {
        "local": {
            "url": "http://localhost:11434/api",  # Verify URL
            "timeout": 120  # Increase timeout
        }
    }
}
```

### Logging and Debugging

To enable detailed logging for troubleshooting:

```python
from semantic_qa_gen import SemanticQAGen
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or enable verbose mode
qa_gen = SemanticQAGen(verbose=True)
```

For CLI usage:
```bash
semantic-qa-gen process document.pdf -o output --verbose
```

### Getting Help

- **Documentation**: Visit [readthedocs.io](https://readthedocs.io)
- **GitHub Issues**: Submit bugs or feature requests on our GitHub repository
- **Community Forum**: Join our community forum for discussions and help

---

## License

SemanticQAGen is released under the MIT License.

Copyright © 2025 Stephen Genusa
