
"""Configuration schema definitions for SemanticQAGen."""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator


class LoaderConfig(BaseModel):
    """Base configuration for document loaders."""
    enabled: bool = True


class TextLoaderConfig(LoaderConfig):
    """Configuration for text document loader."""
    encoding: str = "utf-8"
    detect_encoding: bool = True


class PDFLoaderConfig(LoaderConfig):
    """Configuration for PDF document loader."""
    extract_images: bool = False
    ocr_enabled: bool = False
    detect_headers_footers: bool = True
    fix_cross_page_sentences: bool = True
    preserve_page_numbers: bool = True


class MarkdownLoaderConfig(LoaderConfig):
    """Configuration for Markdown document loader."""
    extract_metadata: bool = True
    encoding: str = "utf-8"


class DocxLoaderConfig(LoaderConfig):
    """Configuration for DOCX document loader."""
    extract_images: bool = False
    extract_tables: bool = True


class DocumentConfig(BaseModel):
    """Configuration for document processing."""
    
    class LoadersConfig(BaseModel):
        text: TextLoaderConfig = Field(default_factory=TextLoaderConfig)
        pdf: PDFLoaderConfig = Field(default_factory=PDFLoaderConfig)
        markdown: MarkdownLoaderConfig = Field(default_factory=MarkdownLoaderConfig)
        docx: DocxLoaderConfig = Field(default_factory=DocxLoaderConfig)
    
    loaders: LoadersConfig = Field(default_factory=LoadersConfig)
    normalize_whitespace: bool = True
    fix_encoding_issues: bool = True
    extract_metadata: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    
    strategy: str = "semantic"  # Options: semantic, fixed_size
    target_chunk_size: int = 1500
    overlap_size: int = 150
    preserve_headings: bool = True
    min_chunk_size: int = 500
    max_chunk_size: int = 2500
    
    @validator("strategy")
    def validate_strategy(cls, v):
        """Validate chunking strategy."""
        valid_strategies = ["semantic", "fixed_size", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f"Chunking strategy must be one of {valid_strategies}")
        return v
    
    @validator("min_chunk_size", "max_chunk_size", "target_chunk_size")
    def validate_chunk_size(cls, v):
        """Validate chunk size ranges."""
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        return v
    
    @root_validator
    def validate_chunk_size_relationships(cls, values):
        """Validate chunk size relationships."""
        min_size = values.get('min_chunk_size', 500)
        target_size = values.get('target_chunk_size', 1500)
        max_size = values.get('max_chunk_size', 2500)
        
        if min_size > target_size:
            raise ValueError("min_chunk_size must be less than or equal to target_chunk_size")
        if target_size > max_size:
            raise ValueError("target_chunk_size must be less than or equal to max_chunk_size")
            
        return values


class LLMServiceConfig(BaseModel):
    """Configuration for LLM services."""
    
    class LocalServiceConfig(BaseModel):
        enabled: bool = True
        url: str = "http://localhost:11434/api"
        model: str = "mistral:7b"
        default_for: List[str] = ["chunking", "validation"]
        timeout: int = 60
        concurrency_limit: int = 3
    
    class RemoteServiceConfig(BaseModel):
        enabled: bool = True
        provider: str = "openai"  # Options: openai, anthropic, etc.
        model: str = "gpt-4"
        default_for: List[str] = ["analysis", "generation"]
        api_key: Optional[str] = None
        timeout: int = 120
        rate_limit_tokens: int = 90000  # Tokens per minute
        rate_limit_requests: int = 100  # Requests per minute
        
        @validator("model")
        def recommend_gpt4_for_openai(cls, v, values):
            """Recommend GPT-4 for best results with OpenAI."""
            provider = values.get('provider', '')
            if provider == 'openai' and not ('gpt-4' in v):
                import warnings
                warnings.warn(
                    f"Model '{v}' is being used with OpenAI provider. "
                    "GPT-4 is recommended for best results."
                )
            return v
    
    local: LocalServiceConfig = Field(default_factory=LocalServiceConfig)
    remote: RemoteServiceConfig = Field(default_factory=RemoteServiceConfig)
    
    @root_validator
    def validate_at_least_one_enabled(cls, values):
        """Validate that at least one LLM service is enabled."""
        local_enabled = values.get('local', {}).get('enabled', False)
        remote_enabled = values.get('remote', {}).get('enabled', False)
        
        if not local_enabled and not remote_enabled:
            raise ValueError("At least one LLM service (local or remote) must be enabled")
            
        return values


class QuestionGenerationConfig(BaseModel):
    """Configuration for question generation."""
    
    class CategoryConfig(BaseModel):
        min_questions: int = 1
        weight: float = 1.0
        
    categories: Dict[str, CategoryConfig] = Field(default_factory=lambda: {
        "factual": CategoryConfig(min_questions=2, weight=1.0),
        "inferential": CategoryConfig(min_questions=2, weight=1.2),
        "conceptual": CategoryConfig(min_questions=1, weight=1.5)
    })
    
    class DiversityConfig(BaseModel):
        required: bool = True
        min_similarity_threshold: float = 0.75
        
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    max_questions_per_chunk: int = 10
    adaptive_generation: bool = True
    
    @validator("max_questions_per_chunk")
    def validate_max_questions(cls, v):
        """Validate maximum questions per chunk."""
        if v <= 0:
            raise ValueError("max_questions_per_chunk must be positive")
        return v


class ValidationConfig(BaseModel):
    """Configuration for question and answer validation."""
    
    class ValidatorConfig(BaseModel):
        enabled: bool = True
        threshold: float = 0.8
        
    factual_accuracy: ValidatorConfig = Field(default_factory=ValidatorConfig)
    answer_completeness: ValidatorConfig = Field(default_factory=ValidatorConfig)
    question_clarity: ValidatorConfig = Field(default_factory=ValidatorConfig)
    
    class DiversityValidatorConfig(ValidatorConfig):
        similarity_metric: str = "cosine"  # Options: cosine, jaccard
        
    diversity: DiversityValidatorConfig = Field(default_factory=DiversityValidatorConfig)


class OutputConfig(BaseModel):
    """Configuration for output formatting."""
    
    format: str = "json"  # Options: json, csv
    include_metadata: bool = True
    include_statistics: bool = True
    output_dir: str = "./output"
    
    # JSON output options
    json_indent: int = 2
    json_ensure_ascii: bool = False
    
    # CSV output options
    csv_delimiter: str = ","
    csv_quotechar: str = '"'
    
    @validator("format")
    def validate_format(cls, v):
        """Validate output format."""
        valid_formats = ["json", "csv"]
        if v not in valid_formats:
            raise ValueError(f"Output format must be one of {valid_formats}")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for processing controls."""
    
    concurrency: int = 3
    enable_checkpoints: bool = True
    checkpoint_interval: int = 10  # chunks
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    @validator("concurrency")
    def validate_concurrency(cls, v):
        """Validate concurrency."""
        if v <= 0:
            raise ValueError("Concurrency must be positive")
        if v > 10:
            import warnings
            warnings.warn(f"High concurrency value ({v}) may cause rate limiting or resource issues")
        return v


class SemanticQAGenConfig(BaseModel):
    """Root configuration for SemanticQAGen."""
    
    version: str = "1.0"
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm_services: LLMServiceConfig = Field(default_factory=LLMServiceConfig)
    question_generation: QuestionGenerationConfig = Field(default_factory=QuestionGenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "ignore"
        arbitrary_types_allowed = False
