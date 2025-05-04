# filename: semantic_qa_gen/config/schema.py

"""Configuration schema definitions for SemanticQAGen using Pydantic V2."""

import sys # For checking Python version for TypeAlias
from typing import Dict, List, Optional, Union, Any
# Pydantic V2 imports
from pydantic import (
    BaseModel, Field, field_validator, model_validator, ValidationInfo,
    TypeAdapter
)
from pydantic_core import PydanticCustomError
import warnings


# --- Helper Type ---
if sys.version_info >= (3, 10):
    from typing import TypeAlias
    # Basic string types, validation happens within Pydantic fields
    HttpUrl: TypeAlias = str
    DirectoryPath: TypeAlias = str
    FilePath: TypeAlias = str
else: # for Python 3.8/3.9 (Although project requires 3.10+)
    HttpUrl = str
    DirectoryPath = str
    FilePath = str

# --- Standalone Model Configuration ---
class ModelConfig(BaseModel):
    """Configuration specific to a model invocation for a task."""
    name: str = Field(..., description="The specific model identifier to use for this task.")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation (0.0-2.0). Higher values mean more random outputs."
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional maximum number of tokens to generate in the completion."
    )
    # model_config replaces the old Config class
    model_config = {
        "extra": "allow" # Allow other adapter-specific params like top_p, json_mode etc.
    }


# --- Loader Configurations ---
class LoaderConfig(BaseModel):
    enabled: bool = Field(True, description="Enable/disable this loader.")

class TextLoaderConfig(LoaderConfig):
    encoding: str = Field("utf-8")
    # `detect_encoding` moved - handled implicitly by trying utf-8 then standard libs if needed

class PDFLoaderConfig(LoaderConfig):
    extract_images: bool = Field(False)
    # ocr_enabled: bool = Field(False) # OCR support removed/deferred
    detect_headers_footers: bool = Field(True)
    fix_cross_page_sentences: bool = Field(True)
    preserve_page_numbers: bool = Field(True)

class MarkdownLoaderConfig(LoaderConfig):
    extract_metadata: bool = Field(True) # Keep YAML front matter parsing
    encoding: str = Field("utf-8")

class DocxLoaderConfig(LoaderConfig):
    extract_images: bool = Field(False)
    # extract_tables: bool = Field(True) # Table extraction removed/deferred

# --- Document Configuration ---
class DocumentConfig(BaseModel):
    class LoadersConfig(BaseModel):
        text: TextLoaderConfig = Field(default_factory=TextLoaderConfig)
        pdf: PDFLoaderConfig = Field(default_factory=PDFLoaderConfig)
        markdown: MarkdownLoaderConfig = Field(default_factory=MarkdownLoaderConfig)
        docx: DocxLoaderConfig = Field(default_factory=DocxLoaderConfig)

    loaders: LoadersConfig = Field(default_factory=LoadersConfig)
    normalize_whitespace: bool = Field(True)
    fix_encoding_issues: bool = Field(True) # Attempt basic encoding fixes
    fix_formatting_issues: bool = Field(True) # Attempt basic formatting heuristics
    # extract_metadata is handled within specific loaders

# --- Chunking Configuration ---
class ChunkingConfig(BaseModel):
    strategy: str = Field("semantic", description="Chunking strategy ('semantic', 'fixed_size').")
    target_chunk_size: int = Field(1500, gt=0, description="Approximate target size (chars/tokens depend on strategy).")
    overlap_size: int = Field(150, ge=0, description="Size of overlap between chunks (chars/tokens depend on strategy).")
    preserve_headings: bool = Field(True, description="Whether to attempt splitting chunks before major headings (semantic strategy).")
    min_chunk_size: int = Field(500, ge=0, description="Minimum size for a chunk (used in optimization/strategies).")
    max_chunk_size: int = Field(2500, gt=0, description="Maximum size for a chunk (used in optimization/strategies).")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str):
        valid_strategies = ["semantic", "fixed_size"] # Add more as implemented
        if v not in valid_strategies:
            raise ValueError(f"Chunking strategy must be one of {valid_strategies}")
        return v

    # Pydantic V2 model_validator for cross-field checks
    @model_validator(mode='after')
    def validate_chunk_size_relationships(self) -> 'ChunkingConfig':
        min_size = self.min_chunk_size
        target_size = self.target_chunk_size
        max_size = self.max_chunk_size
        overlap = self.overlap_size

        if not (0 <= min_size <= target_size <= max_size):
            raise PydanticCustomError(
                'value_error',
                f"Chunk sizes must follow: 0 <= min ({min_size}) <= target ({target_size}) <= max ({max_size})",
                {'min': min_size, 'target': target_size, 'max': max_size}
            )

        if overlap > min_size:
            # Use warnings module for user-facing warnings
            warnings.warn(
                f"config: overlap_size ({overlap}) is greater than min_chunk_size ({min_size}), "
                "which might lead to unexpected behavior or very small effective chunk content.",
                UserWarning
            )
        return self


# --- LLM Service Configuration ---
class BaseLLMServiceDetails(BaseModel): # Base class for shared fields
    """Common fields for LLM service details."""
    enabled: bool = Field(True, description="Enable/disable this specific service (e.g., 'local' or 'remote').")
    model: str = Field(..., description="Default model identifier used by this service (e.g., 'gpt-4', 'mistral:7b'). Tasks can override this.")
    api_key: Optional[str] = Field(None, description="API key. Can use ${ENV_VAR} format for interpolation.", validate_default=False) # Don't validate default None
    timeout: int = Field(120, ge=1, description="Request timeout in seconds.")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts for transient errors.")
    initial_delay: float = Field(1.0, gt=0, description="Initial delay in seconds before the first retry.")
    max_delay: float = Field(60.0, gt=0, description="Maximum delay in seconds between retries.")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Optional custom headers for the API requests.")

    # Shared validator for delay relationship
    @model_validator(mode='after')
    def check_delay_config(self) -> 'BaseLLMServiceDetails':
        initial = self.initial_delay
        maximum = self.max_delay
        if maximum < initial:
            raise PydanticCustomError(
                'value_error',
                f"LLM Service config: max_delay ({maximum}) cannot be smaller than initial_delay ({initial})",
                {'initial_delay': initial, 'max_delay': maximum}
            )
        return self

    # model_config replaces the old Config class
    model_config = {
        "extra": "allow" # Allow additional fields maybe used by specific adapters
                         # like rate_limit_requests (used by placeholder)
    }


class LocalServiceConfig(BaseLLMServiceDetails):
    """Configuration for locally running LLM services (e.g., Ollama)."""
    model: str = Field("mistral:7b", description="Default model name for the local service (e.g., 'mistral:7b', 'llama3:8b').")
    url: HttpUrl = Field(
        "http://localhost:11434", description="Base URL for the local LLM API (e.g., Ollama base /v1 endpoint, LM Studio OpenAI-compatible endpoint)."
    )
    preferred_tasks: List[str] = Field(
        default_factory=lambda: ["validation"],
        description="List of tasks this service is preferred for ('analysis', 'generation', 'validation'). Task assignment falls back if not preferred."
    )

    @field_validator('preferred_tasks', mode='after')
    @classmethod
    def check_tasks(cls, tasks: List[str]):
        valid_tasks = {'analysis', 'generation', 'validation'}
        for task in tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task '{task}' in preferred_tasks. Must be one of {valid_tasks}")
        return tasks


class RemoteServiceConfig(BaseLLMServiceDetails):
    """Configuration for remote LLM API services (e.g., OpenRouter, OpenAI)."""
    model: str = Field("gpt-4o", description="Default model name recognized by the remote API provider (e.g., 'gpt-4o', 'claude-3-opus-20240229').")
    provider: str = Field(
        "openai", description="Helps select adapter logic or API specifics ('openai', 'azure', 'openrouter', 'anthropic', etc.)."
    )
    api_base: Optional[HttpUrl] = Field(
         None, description="API base URL. Often needed for non-standard providers or Azure. Can sometimes be inferred if provider is known (e.g., default OpenAI API base)."
    )
    preferred_tasks: List[str] = Field(
        default_factory=lambda: ["analysis", "generation"],
        description="List of tasks this service is preferred for ('analysis', 'generation', 'validation')."
    )
    organization: Optional[str] = Field(
         None, description="Optional OpenAI organization ID. Can use ${ENV_VAR}."
    )
    # Azure specific - could be nested under provider details later
    api_version: Optional[str] = Field(None, description="API version, required for Azure OpenAI deployments.")

    @field_validator('preferred_tasks', mode='after')
    @classmethod
    def check_tasks(cls, tasks: List[str]):
        valid_tasks = {'analysis', 'generation', 'validation'}
        for task in tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task '{task}' in preferred_tasks. Must be one of {valid_tasks}")
        return tasks

    @field_validator("model")
    @classmethod
    def check_model_recommendation(cls, v: str, info: ValidationInfo): # Pydantic v2 passes ValidationInfo
        if not info.context or 'config_dict' not in info.context:
             # Cannot access other fields easily without full context propagation in Pydantic V2
             # Basic standalone check:
             if 'gpt-4' not in v and 'gpt-3.5' not in v and 'claude' not in v:
                  warnings.warn(
                      f"Remote model name '{v}' used. Ensure this model is compatible with the selected provider.",
                      UserWarning
                  )
             return v

        # If context was passed (advanced usage), check provider:
        values = info.context['config_dict'] # Assume parent passes full dict here
        provider = values.get('provider', '').lower()
        api_base = values.get('api_base', '')
        is_openai_provider = provider == 'openai' or (api_base and 'openai.com' in api_base)

        if is_openai_provider and not ('gpt-4' in v or 'gpt-3.5' in v):
             warnings.warn(
                 f"Model '{v}' specified for OpenAI provider. Consider using standard models like 'gpt-4o', 'gpt-4-turbo', or 'gpt-3.5-turbo' unless using fine-tuned models.", UserWarning
             )
        return v

    # Add validation for Azure specifics
    @model_validator(mode='after')
    def check_azure_requirements(self) -> 'RemoteServiceConfig':
         if self.provider == 'azure':
              if not self.api_base:
                   raise PydanticCustomError('value_error', "Azure provider requires 'api_base' (endpoint URL).", {})
              if not self.api_version:
                   raise PydanticCustomError('value_error', "Azure provider requires 'api_version'.", {})
         return self


class LLMServiceConfig(BaseModel):
    local: Optional[LocalServiceConfig] = Field(
        default=None, description="Configuration for local LLM services. Enable by defining this section."
    )
    remote: Optional[RemoteServiceConfig] = Field(
        # Don't default factory here, let it be None unless specified by user
        default=None, description="Configuration for remote LLM services. Enable by defining this section."
    )

    @model_validator(mode='after')
    def validate_llm_config(self) -> 'LLMServiceConfig':
        local_config = self.local
        remote_config = self.remote

        local_enabled = local_config and local_config.enabled
        remote_enabled = remote_config and remote_config.enabled

        # 1. At least one service must be defined AND enabled
        if not local_enabled and not remote_enabled:
             # If neither are defined, it's okay if user doesn't intend LLM use (rare)
             if local_config is None and remote_config is None:
                  warnings.warn("No LLM services ('local' or 'remote') are defined in the configuration. LLM-dependent features will fail.", UserWarning)
             else:
                  # At least one was defined but explicitly disabled
                   raise PydanticCustomError(
                       'value_error',
                       "Configuration Error: At least one LLM service ('local' or 'remote') must be defined and enabled for LLM features.", {})

        # 2. Check API keys if services are enabled (Warning only)
        if remote_enabled and not getattr(remote_config, 'api_key', None):
            warnings.warn("config: Remote LLM service is enabled, but no api_key is set. Ensure it's provided via environment variable or config.", UserWarning)

        # 3. Prevent task overlap in preferences (Strict)
        local_tasks = set(local_config.preferred_tasks) if local_enabled else set()
        remote_tasks = set(remote_config.preferred_tasks) if remote_enabled else set()
        overlap = local_tasks.intersection(remote_tasks)
        if overlap:
             raise PydanticCustomError(
                'value_error',
                f"Configuration Error: Tasks {overlap} cannot be preferred by both local and remote LLM services.",
                {'overlap': list(overlap)}
             )
        return self

class CategoryConfig(BaseModel):
    """Configuration for a question category."""
    min_questions: int = Field(1, ge=0, description="Minimum questions of this type per chunk.")
    weight: float = Field(1.0, gt=0, description="Weighting factor for adaptive generation.")


# --- Question Generation Configuration ---
class QuestionGenerationConfig(BaseModel):
    class CategoryConfig(BaseModel):
        min_questions: int = Field(1, ge=0, description="Minimum questions of this type per chunk.")
        weight: float = Field(1.0, gt=0, description="Weighting factor for adaptive generation.")

    categories: Dict[str, CategoryConfig] = Field(default_factory=lambda: {
        "factual": CategoryConfig(min_questions=2, weight=1.0),
        "inferential": CategoryConfig(min_questions=2, weight=1.2),
        "conceptual": CategoryConfig(min_questions=1, weight=1.5)
    }, description="Configuration for different question categories.")

    max_questions_per_chunk: int = Field(10, gt=0, description="Maximum total questions allowed per chunk.")
    adaptive_generation: bool = Field(True, description="Enable adaptive question count based on chunk analysis (density).")


# --- Validation Configuration ---
class BaseValidatorConfig(BaseModel): # Base for specific validator configs
    enabled: bool = Field(True)
    threshold: float = Field(0.7, ge=0.0, le=1.0)

class FactualValidatorConfig(BaseValidatorConfig):
    pass

class CompletenessValidatorConfig(BaseValidatorConfig):
    pass

class ClarityValidatorConfig(BaseValidatorConfig):
    pass

class DiversityValidatorConfig(BaseValidatorConfig):
    threshold: float = Field(0.85, ge=0.0, le=1.0, description="Similarity threshold (e.g., SequenceMatcher ratio) ABOTE which questions are rejected by this validator.")


class ValidationConfig(BaseModel):
    factual_accuracy: FactualValidatorConfig = Field(default_factory=FactualValidatorConfig)
    answer_completeness: CompletenessValidatorConfig = Field(default_factory=CompletenessValidatorConfig)
    question_clarity: ClarityValidatorConfig = Field(default_factory=ClarityValidatorConfig)
    diversity: DiversityValidatorConfig = Field(default_factory=DiversityValidatorConfig)

# --- Output Configuration ---
class OutputConfig(BaseModel):
    format: str = Field("json", description="Output format ('json', 'csv').")
    include_metadata: bool = Field(True, description="Include document and processing metadata in output file.")
    include_statistics: bool = Field(True, description="Include processing statistics in output file (may be separate file for CSV).")
    output_dir: DirectoryPath = Field("./output", description="Directory to save output files.")

    # Format-specific options - Consider nesting under format keys later if many options
    json_indent: Optional[int] = Field(2, description="Indentation level for JSON output (None for compact).")
    json_ensure_ascii: bool = Field(False, description="Ensure JSON output only contains ASCII characters.")
    csv_delimiter: str = Field(",", description="Delimiter character for CSV output.")
    csv_quotechar: str = Field('"', description="Quote character for CSV output.")
    csv_include_document_info: bool = Field(False, description="Include document metadata columns in CSV output (can make CSV large).")


    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str):
        valid_formats = ["json", "csv"] # Add more as implemented
        if v.lower() not in valid_formats:
            raise ValueError(f"Output format must be one of {valid_formats}")
        return v.lower()

# --- Processing Configuration ---
class ProcessingConfig(BaseModel):
    concurrency: int = Field(3, gt=0, description="Maximum concurrent asynchronous operations (e.g., LLM calls).")
    enable_checkpoints: bool = Field(True, description="Enable saving/resuming progress using checkpoints.")
    checkpoint_interval: int = Field(10, ge=1, description="Save checkpoint after every N chunks processed.")
    checkpoint_dir: DirectoryPath = Field("./checkpoints", description="Directory to store checkpoint files.")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    debug_mode: bool = Field(False, description="Enable debug mode (more verbose logging, potentially saves failed LLM inputs/outputs).")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v

# --- Root Configuration ---
class SemanticQAGenConfig(BaseModel):
    """Root configuration model for SemanticQAGen."""
    version: str = Field("1.0", description="Schema version for future compatibility.") # Keep schema version logical
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm_services: LLMServiceConfig = Field(default_factory=LLMServiceConfig)
    question_generation: QuestionGenerationConfig = Field(default_factory=QuestionGenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Pydantic V2 model_config replaces Config class
    model_config = {
        "validate_assignment": True, # Re-validate on attribute assignment
        "extra": "forbid",           # Forbid extra fields not defined in schema
        "arbitrary_types_allowed": False # Disallow arbitrary types
    }

# --- PDFLoader Configuration ---
class PDFLoaderConfig(LoaderConfig):
    extract_images: bool = Field(False)
    ocr_enabled: bool = Field(False, description="Enable OCR for scanned documents or images with text")
    ocr_language: str = Field("eng", description="OCR language code (e.g., 'eng', 'deu', 'fra', etc.)")
    ocr_dpi: int = Field(300, description="DPI for rendering pages for OCR processing")
    ocr_timeout: int = Field(30, description="Timeout in seconds for OCR processing per page")
    detect_headers_footers: bool = Field(True)
    fix_cross_page_sentences: bool = Field(True)
    preserve_page_numbers: bool = Field(True)
    use_advanced_reading_order: bool = Field(False, description="Use advanced reading order algorithm (better for complex layouts)")
    min_heading_ratio: float = Field(1.2, description="Font size ratio for heading detection")
    header_footer_threshold: float = Field(0.75, description="Similarity threshold for header/footer detection")
