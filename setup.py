"""Setup script for the SemanticQAGen package."""

from setuptools import setup, find_packages

setup(
    name="semantic_qa_gen",
    version="0.1.0",
    author="Stephen Genusa",
    author_email="github@genusa.com",
    description="A Python library for generating high-quality question-answer pairs from text content",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stephengenusa/semantic-qa-gen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0,<7.0",
        "httpx>=0.23.0,<0.25.0",
        "pydantic>=1.10.0,<2.0.0",
        "tqdm>=4.65.0,<5.0.0",
        "python-magic>=0.4.27,<0.5.0",
        "commonmark>=0.9.1,<0.10.0",
        "tiktoken>=0.4.0",
        "openai>=0.27.0",
    ],
    extras_require={
        "pdf": [
            "pymupdf>=1.22.3",
        ],
        "docs": [
            "pymupdf>=1.22.3",
            "python-docx>=0.8.11",
        ],
        "rag": [
            "nltk>=3.8.1",
            "transformers>=4.28.0",
            "sentence-transformers>=2.2.2",
        ],
        "full": [
            "pymupdf>=1.22.3",
            "python-docx>=0.8.11",
            "nltk>=3.8.1",
            "transformers>=4.28.0",
            "sentence-transformers>=2.2.2",
            "rich>=13.3.5",
        ],
        "dev": [
            "pytest>=7.3.1,<8.0.0",
            "pytest-asyncio>=0.21.0,<0.22.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "mkdocs>=1.4.3",
            "mkdocs-material>=9.1.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "semantic-qa-gen=semantic_qa_gen.cli:main",
        ],
    },
)
