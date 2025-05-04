# filename: setup.py

"""Setup script for the SemanticQAGen package."""

from setuptools import setup, find_packages
import os # Import os for path manipulation

# Function to read requirements from a file (Keep as is)
def parse_requirements(filename):
    """Load requirements from a requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Function to read the version from the version file
def get_version():
    version_file = os.path.join('semantic_qa_gen', 'version.py')
    with open(version_file, 'r') as f:
        exec(f.read())
    return locals()['__version__']


# Define core requirements
core_requirements = [
    "pyyaml>=6.0,<7.0",
    "httpx>=0.25.0,<0.28.0",
    # Updated Pydantic to V2
    "pydantic>=2.0.0,<3.0.0",
    "tqdm>=4.65.0,<5.0.0",
    # Removed python-magic - Mime type detection integrated, but not strictly core?
    # User can install it if needed for more robust detection
    "commonmark>=0.9.1,<0.10.0", # For markdown parsing
    "tiktoken>=0.4.0,<0.8.0",
    "ruamel.yaml>=0.17.21,<0.19.0", # For comment-preserving YAML dump
]

# Define extras requires
extras_require={
    "pdf": [
        "pymupdf>=1.24.0,<1.25.0", # Updated PyMuPDF range
    ],
    "ocr": [
        "pytesseract>=0.3.10",
        "pillow>=9.5.0",
        "ftfy>=6.1.1",
    ],
    "advanced": [
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
    ],
    "docx": [
        "python-docx>=1.0.0,<1.2.0", # Updated python-docx range
    ],
    # Combine document format extras
    "formats": [
        "pymupdf>=1.24.0,<1.25.0",
        "python-docx>=1.0.0,<1.2.0",
        "pytesseract>=0.3.10",
        "pillow>=9.5.0",
        "ftfy>=6.1.1",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
    ],
    # LLM Provider Extras
    "openai": [
        "openai>=1.3.0,<2.0.0" # Updated openai range for v1+ API
    ],
    "anthropic": [
        # "anthropic>=0.20.0,<0.21.0" # Example if needed
    ],
    "local": [
        # Usually no specific deps for local, maybe 'requests' if not using httpx for some reason
    ],
    # NLP/RAG Features (RAG deferred, but keeping NLP deps)
    "nlp": [
        "nltk>=3.8.1,<4.0.0",
        # Add others if chunking/analysis strategies need them beyond nltk
        # "spacy>=3.0.0,<4.0.0"
    ],
    # Full includes specific providers and formats
    "full": [
        "pymupdf>=1.24.0,<1.25.0",
        "python-docx>=1.0.0,<1.2.0",
        "pytesseract>=0.3.10",
        "pillow>=9.5.0",
        "ftfy>=6.1.1",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "openai>=1.3.0,<2.0.0",
        # Add other providers like anthropic here if supported
        "nltk>=3.8.1,<4.0.0",
        "rich>=13.3.5,<14.0.0", # Rich is optional but useful for CLI
    ],
    # Development Dependencies
    "dev": [
        "pytest>=7.3.1,<9.0.0",
        "pytest-asyncio>=0.21.0,<0.24.0",
        "pytest-cov>=4.1.0,<6.0.0",
        "black>=23.3.0,<25.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<8.0.0",
        "mypy>=1.8.0,<2.0.0", # Updated mypy range
        "pre-commit>=3.0.0,<4.0.0",
        "mkdocs>=1.5.0,<1.7.0",
        "mkdocs-material>=9.1.15,<10.0.0",
        # Add types-pyyaml for better type checking with pyyaml
        "types-pyyaml>=6.0.0,<7.0.0",
    ],
}


setup(
    name="semantic_qa_gen",
    version=get_version(), # Get version dynamically
    author="Stephen Genusa",
    author_email="github@genusa.com",
    description="A Python library for generating high-quality question-answer pairs from text content",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stephengenusa/semantic-qa-gen",
    # Find packages within the 'semantic_qa_gen' directory relative to setup.py
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Typing :: Typed", # Indicate type hints are used
    ],
    python_requires=">=3.10",
    # Use core requirements list
    install_requires=core_requirements,
    # Use structured extras_require
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "semantic-qa-gen=semantic_qa_gen.cli.commands:main",
        ],
    },
    # Include package data (like prompt templates)
    include_package_data=True,
    package_data={
        # Ensure YAML files in llm/prompts/templates are included
        'semantic_qa_gen': ['llm/prompts/templates/*.yaml', 'llm/prompts/templates/*.yml'],
    },
)
