<div align="center">
    <h1>SemanticQAGen</h1>
  <p><em>Intelligent Question-Answer Generation with Advanced Semantic Understanding</em></p>
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
> **Alpha Release (v0.1.0)**: This library is in active development. The core functionality is implemented, but some advanced features are still under development.

> **UPDATE 2025/05/05** I have done a significant amount of work and the library is now functional. The code in this repo has not been updated yet. I intend to make further changes and test the library thoroughly before pushing the code. Please do not fork and issue pull requests. A significant amount of refactoring and feature additions have been made.
---

## Overview

SemanticQAGen is a powerful Python library for generating high-quality question-answer pairs from text documents. It uses advanced semantic understanding to intelligently process content, analyze information density, and create diverse questions across multiple cognitive levels.

SemanticQAGen features enhanced semantic chunking, dynamic question generation, validation of questions and answers, and flexible LLM routing capabilities. You can run all tasks locally on an OpenAI-compatible server, run them via a remote API, or split specific tasks (e.g., validation, analysis, generation) between local and remote servers. The library is designed with a "for Humans" philosophy - simple for basic use cases while providing advanced capabilities for power users.

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

# With DOCX support
pip install semantic-qa-gen[docx]

# With all document format support
pip install semantic-qa-gen[formats]

# With OpenAI LLM support
pip install semantic-qa-gen[openai]

# With NLP capabilities
pip install semantic-qa-gen[nlp]

# With all features
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

# Create a config file interactively 
semantic-qa-gen init-config config.yml --interactive

# Process with a specific configuration
semantic-qa-gen process document.txt --config config.yml --format json
```

---

## Feature Overview

SemanticQAGen offers a comprehensive set of features designed to produce high-quality question and answer sets:

| Feature Category | Capability                                                         | Status |
|-----------------|--------------------------------------------------------------------|:--------:|
| **Document Processing** | Document format support: TXT, PDF, DOCX, MD                      | ✅ |
|  | Automatic document type detection                                  | ✅ |
|  | Cross-page content handling                                        | ✅ |
|  | Header/footer detection and removal                                | ✅ |
| **Content Analysis** | Semantic document chunking                                         | ✅ |
|  | Information density analysis                                       | ✅ |
|  | Topic coherence evaluation                                         | ✅ |
|  | Key concept extraction                                             | ✅ |
| **Question Generation** | Multi-level cognitive questions (factual, inferential, conceptual) | ✅ |
|  | Adaptive generation based on content quality                       | ✅ |
|  | Question diversity enforcement                                     | ✅ |
|  | Custom question categories                                         | ✅ |
| **Answer Validation** | Factual accuracy verification                                      | ✅ |
|  | Question clarity evaluation                                        | ✅ |
|  | Answer completeness assessment                                     | ✅ |
| **LLM Integration** | OpenAI API support                                                 | ✅ |
|  | Local LLM support (Ollama, etc.)                                   | ✅ |
|  | Hybrid task routing                                                | ✅ |
|  | Automatic fallback mechanisms                                      | ✅ |
| **Processing Control** | Checkpoint and resume capability                                   | ✅ |
|  | Concurrent processing                                              | ✅ |
|  | Progress tracking and reporting                                    | ✅ |
| **Output Options** | Multiple export formats (JSON, CSV, JSONL)                         | ✅ |
|  | Metadata inclusion                                                 | ✅ |
|  | Statistics and analytics                                           | ✅ |
| **Extensibility** | Custom document loaders                                            | ✅ |
|  | Custom chunking strategies                                         | ✅ |
|  | Custom validators                                                  | ✅ |

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
Process multiple files from a directory:

```python
# Process all files in a directory
batch_results = qa_gen.process_input_directory()
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
        "strategy": "semantic",  # Options: semantic, fixed_size
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

#### Question Clarity Evaluation
Questions are evaluated for clarity and unambiguity, filtering out poorly formed questions that might confuse learners.

#### Answer Completeness Assessment
The system checks that answers thoroughly address the questions asked, eliminating partial or incomplete responses.

---

## Advanced Features

### LLM Integration

#### OpenAI API Support
Full integration with OpenAI with optimized prompting strategies for each task in the pipeline.

#### Local LLM Support
Support for local LLM deployment via Ollama and similar services, allowing use of models like Mistral, running on your own hardware without requiring external API access.

#### Hybrid Task Routing
Intelligently route different tasks to the most appropriate LLM based on task complexity and model capability. For example, use GPT-4 for complex question generation but a local model for simple validation tasks.

```python
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
            "model": "gpt-4o",
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

#### Concurrent Processing
Multi-threaded processing of chunks with configurable concurrency levels to maximize throughput on multi-core systems.

#### Progress Tracking and Reporting
Detailed progress reporting during processing, with support for both simple console output and rich interactive displays (when installed with Rich).

### Output Options

#### Multiple Export Formats
Export question-answer pairs in various formats including JSON, CSV, and JSONL with customizable formatting options.

```python
# Save questions in different formats
qa_gen.save_questions(result, "questions_output", format_name="json")
qa_gen.save_questions(result, "questions_output", format_name="csv")
qa_gen.save_questions(result, "questions_output", format_name="jsonl")
```

#### Metadata Inclusion
Include rich metadata about source documents, generation parameters, and validation results with the generated questions.

#### Statistics and Analytics
Comprehensive statistics about generated questions, including category distribution, validation success rates, and content coverage.

---

## Project File Organization
```
.
├── semantic_qa_gen
│   ├── __init__.py
│   ├── version.py
│   ├── semantic_qa_gen.py
│   ├── chunking
│   │   ├── analyzer.py
│   │   ├── engine.py
│   │   └── strategies
│   │       ├── base.py
│   │       ├── fixed_size.py
│   │       ├── nlp_helpers.py
│   │       └── semantic.py
│   ├── cli
│   │   ├── __init__.py
│   │   └── commands.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── schema.py
│   ├── document
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── processor.py
│   │   └── loaders
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── docx.py
│   │       ├── markdown.py
│   │       ├── pdf.py
│   │       └── text.py
│   ├── llm
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── service.py
│   │   ├── adapters
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── openai_adapter.py
│   │   └── prompts
│   │       ├── __init__.py
│   │       └── manager.py
│   ├── output
│   │   ├── __init__.py
│   │   ├── formatter.py
│   │   └── adapters
│   │       ├── __init__.py
│   │       ├── csv.py
│   │       ├── json.py
│   │       └── jsonl.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   └── semantic.py
│   ├── question
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── processor.py
│   │   └── validation
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── diversity.py
│   │       ├── engine.py
│   │       └── factual.py
│   └── utils
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── error.py
│       ├── logging.py
│       ├── progress.py
│       └── project.py
```

## Architecture

SemanticQAGen implements a modular pipeline architecture with clearly defined components and interfaces:

```
                              ARCHITECTURE OVERVIEW
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
│  │  │Category: Factual │  │ │ │    │  ┌─────────────┐                  │      │
│  │  └──────────────────┘  │ │ │    ├─►│ Traditional │                  │      │
│  │  ┌──────────────────┐  │ │ │    │  │ Validators  │                  │      │
│  │  │Cat: Inferential  │  │ │ │    │  └─────────────┘                  │      │
│  │  └──────────────────┘  │ │ │    │                                   │      │
│  │  ┌──────────────────┐  │ │ │    │                                   │      │
│  │  │Cat: Conceptual   │  │ │ │    │                                   │      │
│  │  └──────────────────┘  │ │ │    │                                   │      │
│  └─────────┬──────────────┘ │ │    │                                   │      │
│            │                │ │    │                                   │      │
│            └────────────────┼─┼────┘                                   │      │
│                             │ └───────────────────────────────────────┬┘      │
│                             │                                         │       │
│                             │                                         │       │
│          ┌──────────────────▼─────────────────────┐                   │       │
│          │           Output Formatter             │                   │       │
│          │                                        │                   │       │
│          │  ┌─────────────┐    ┌────────────────┐ │                   │       │
│          │  │ JSON Adapter│    │  CSV Adapter   │ │                   │       │
│          │  └─────────────┘    └────────────────┘ │                   │       │
│          └────────────────────────────────────────┘                   │       │
│                             │                                         │       │
│           ┌─────────────────▼────────────────────┐                    │       │
│           │           Output Results             │                    │       │
│           │  • Questions & Answers               │                    │       │
│           │  • Document Metadata                 │                    │       │
│           │  • Statistics                        │                    │       │
│           └──────────────────────────────────────┘                    │       │
│                                                                       │       │
│                      ┌─────────────────────────────┐                  │       │
│                      │     Checkpoint Manager      │◄─────────────────┘       │
│                      │   (Resume Capabilities)     │                          │
│                      └─────────────────────────────┘                          │
│                                                                               │
│                      ┌─────────────────────────────┐                          │
│                      │     Progress Reporter       │                          │
│                      │   (Processing Feedback)     │                          │
│                      └─────────────────────────────┘                          │
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
      extract_images: false

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
    preferred_tasks: [validation]
    timeout: 60
  remote:
    enabled: true
    provider: openai
    model: gpt-4o
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

# Validation settings
validation:
  factual_accuracy:
    enabled: true
    threshold: 0.7
  answer_completeness:
    enabled: true
    threshold: 0.7
  question_clarity:
    enabled: true
    threshold: 0.7
  diversity:
    enabled: true
    threshold: 0.85

# Processing settings
processing:
  concurrency: 3
  enable_checkpoints: true
  checkpoint_interval: 10
  checkpoint_dir: "./checkpoints"
  log_level: "INFO"
  debug_mode: false

# Output settings
output:
  format: json
  include_metadata: true
  include_statistics: true
  output_dir: "./output"
  fine_tuning_format: "default"
  json_indent: 2
  json_ensure_ascii: false
  csv_delimiter: ","
  csv_quotechar: "\""
```

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

### Main Class: `SemanticQAGen`

```python
class SemanticQAGen:
    """Main interface for generating question-answer pairs from text documents."""
    
    def __init__(self, config_path: Optional[str] = None, 
                config_dict: Optional[Dict[str, Any]] = None,
                verbose: bool = False,
                project_path: Optional[str] = None):
        """Initialize SemanticQAGen with optional configuration."""
        
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document to generate question-answer pairs.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dictionary containing questions, statistics, and metadata.
        """
        
    def process_input_directory(self, output_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes all readable files in the project's input directory.
        
        Args:
            output_format: Optional output format to override config.
            
        Returns:
            A dictionary summarizing the batch processing results.
        """
        
    def save_questions(self, result: Dict[str, Any], 
                      output_path: str,
                      format_name: Optional[str] = None) -> str:
        """
        Save generated questions to a file.
        
        Args:
            result: Results from process_document.
            output_path: Path where to save the output.
            format_name: Format to save in (json, csv, jsonl).
            
        Returns:
            Path to the saved file.
        """
        
    def create_default_config_file(self, output_path: str, include_comments: bool = True) -> None:
        """Create a default configuration file."""
        
    def dump_failed_chunks(self, output_path: Optional[str] = None) -> int:
        """
        Generate a detailed report of failed chunks for debugging.
        
        Args:
            output_path: Optional path to write the report.
            
        Returns:
            Number of failed chunks reported.
        """
```

---

## CLI Reference

SemanticQAGen provides a command-line interface:

### Main Commands

```
semantic-qa-gen process <document> [-o OUTPUT] [-f {json,csv,jsonl}] [-c CONFIG] [-v]
semantic-qa-gen create-project [path]
semantic-qa-gen init-config <output> [-i]
semantic-qa-gen interactive
semantic-qa-gen formats
semantic-qa-gen info
semantic-qa-gen version
```

### Command Details

```
process             Process a document and generate questions
  document          Path to the document file
  -o, --output      Path for output file
  -f, --format      Output format (json, csv, jsonl)
  -c, --config      Path to config file
  -p, --project     Path to QAGenProject directory
  -v, --verbose     Enable verbose output

create-project      Create a new QAGenProject structure
  path              Path for the new project (default: current directory)

init-config         Create a default configuration file
  output            Path for the config file
  -i, --interactive Create config interactively
  -p, --project     Path to QAGenProject directory

interactive         Run in interactive mode
formats             List supported file formats
info                Show system information
version             Show the version and exit
```

### Examples

```bash
# Process a PDF document
semantic-qa-gen process document.pdf -o questions_output

# Create a new project
semantic-qa-gen create-project my_qa_project

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
qa_gen.save_questions(result, "qa_pairs")

# Display stats
print(f"Generated {len(result['questions'])} questions")
print(f"Factual questions: {result['statistics']['categories'].get('factual', 0)}")
print(f"Inferential questions: {result['statistics']['categories'].get('inferential', 0)}")
print(f"Conceptual questions: {result['statistics']['categories'].get('conceptual', 0)}")
```

### Using a Project Structure

```python
from semantic_qa_gen import SemanticQAGen

# Create or use an existing project structure
qa_gen = SemanticQAGen(project_path="my_qa_project")

# Process a document (can be in project's input directory)
result = qa_gen.process_document("input/document.txt")

# Save questions (will save to project's output directory)
qa_gen.save_questions(result, "questions_output")

# Process all documents in the input directory
batch_results = qa_gen.process_input_directory()
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
            "model": "gpt-4o",
            "api_key": "YOUR_API_KEY",
            "preferred_tasks": ["analysis", "generation"]
        }
    }
}

# Initialize with hybrid LLM config
qa_gen = SemanticQAGen(config_dict=config)

# Process document using hybrid approach
# - Local model will handle validation
# - Remote model will handle analysis and question generation
result = qa_gen.process_document("document.pdf")
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

### Processing with Checkpoints

```python
from semantic_qa_gen import SemanticQAGen

config = {
    "processing": {
        "enable_checkpoints": True,
        "checkpoint_interval": 5  # Save checkpoints every 5 chunks
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

### Creating a Custom Validator

```python
from semantic_qa_gen.question.validation.base import BaseValidator, ValidationResult
from semantic_qa_gen.document.models import Question, Chunk

class CustomValidator(BaseValidator):
    """Custom validator for specialized validation logic."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.7)
    
    async def validate(self, question: Question, 
                    chunk: Chunk,
                    llm_validation_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Implement custom validation logic."""
        # Custom validation implementation
        score = 0.8  # Example score
        
        return ValidationResult(
            question_id=question.id,
            validator_name=self.name,
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
                "use_advanced_reading_order": False
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

---

## License

SemanticQAGen is released under the MIT License.

Copyright © 2025 Stephen Genusa
