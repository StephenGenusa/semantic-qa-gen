# semantic_qa_gen/llm/router.py

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Literal, Union, List

from pydantic import BaseModel  # Assume BaseModel is available

from semantic_qa_gen.config.manager import ConfigManager
# Assume config schema/models define these correctly
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.config.schema import LLMServiceConfig as ServiceDetails # Renamed class, using alias
from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.config.schema import LocalServiceConfig, RemoteServiceConfig

# Adapters
from semantic_qa_gen.llm.adapters.base import BaseLLMAdapter, ModelConfig
# Import the adapter we are actually using
from semantic_qa_gen.llm.adapters.openai_adapter import OpenAIAdapter

from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError, SemanticQAGenError


class TaskRouter:
    """
    Manages LLM service adapters and routes tasks based on configuration.

    Initializes adapters defined in the configuration's 'llm_services' section
    and maps tasks ('analysis', 'generation', 'validation') to the appropriate
    adapter instance and model configuration.
    """
    VALID_TASKS = {'analysis', 'generation', 'validation'}

    def __init__(self, config_manager: ConfigManager, prompt_manager: PromptManager):
        """
        Initialize the TaskRouter.

        Args:
            config_manager: The configuration manager instance.
            prompt_manager: The prompt manager instance.

        Raises:
            ConfigurationError: If the LLM configuration is missing or invalid.
            SemanticQAGenError: For other initialization errors.
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.prompt_manager = prompt_manager

        try:
            self.root_config: SemanticQAGenConfig = config_manager.config
            self.llm_services_config = getattr(self.root_config, 'llm_services', None)

            if not self.llm_services_config:
                raise ConfigurationError("LLM configuration section ('llm_services') is missing.")
        except AttributeError as e:
            raise ConfigurationError(f"Failed to access required configuration sections: {e}")

        self.adapters: Dict[str, BaseLLMAdapter] = {}
        self.mapped_tasks: Dict[str, Tuple[BaseLLMAdapter, ModelConfig]] = {}

        self._initialize_services_and_map_tasks()

    def _initialize_services_and_map_tasks(self) -> None:
        """Initialize adapters and map tasks based on the configuration."""
        self.logger.info("Initializing LLM adapters...")

        # --- Step 1: Initialize Adapter Instances ---
        configured_services = {'local': self.llm_services_config.local, 'remote': self.llm_services_config.remote}

        for service_name, service_cfg in configured_services.items():
            if not service_cfg or not service_cfg.enabled:
                self.logger.info(f"Service '{service_name}' is not configured or is disabled. Skipping initialization.")
                continue

            try:
                adapter_instance = self._create_adapter(service_name, service_cfg)
                if adapter_instance:
                    self.adapters[service_name] = adapter_instance
                    self.logger.info(
                        f"Initialized adapter for service '{service_name}' (Type: {type(adapter_instance).__name__})")

            except (ConfigurationError, SemanticQAGenError, ImportError) as e:
                self.logger.error(f"Fatal error initializing adapter for service '{service_name}': {e}", exc_info=True)
                raise ConfigurationError(f"Adapter initialization failed for '{service_name}': {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error initializing adapter for service '{service_name}': {e}",
                                  exc_info=True)
                raise ConfigurationError(f"Unexpected adapter initialization failure for '{service_name}': {e}")

        if not self.adapters:
            if any(cfg.enabled for cfg in configured_services.values() if cfg):
                raise ConfigurationError(
                    "No LLM adapters were successfully initialized despite services being configured and enabled.")
            else:
                self.logger.warning("No LLM services are configured or enabled.")
                return

        # --- Step 2: Map Tasks based on 'preferred_tasks' ---
        self.logger.info("Mapping tasks to initialized LLM services...")
        task_service_map: Dict[str, str] = {}

        for service_name, adapter in self.adapters.items():
            service_cfg = configured_services.get(service_name)
            if not service_cfg: continue

            preferred_tasks = getattr(service_cfg, 'preferred_tasks', [])
            for task_name in preferred_tasks:
                if task_name not in self.VALID_TASKS:
                    self.logger.warning(f"Service '{service_name}' prefers unknown task '{task_name}'. Ignoring.")
                    continue
                if task_name not in task_service_map:
                    task_service_map[task_name] = service_name
                    self.logger.debug(f"Task '{task_name}' initially mapped to service '{service_name}'.")
                else:
                    self.logger.info(
                        f"Task '{task_name}' already mapped to '{task_service_map[task_name]}'. Service '{service_name}' preference ignored for this task.")

        # Fallback assignment
        first_available_service = None
        service_preference_order = ['remote', 'local'] # Define preference for fallback
        for name in service_preference_order:
            if name in self.adapters:
                first_available_service = name
                break

        if first_available_service:
            for task_name in self.VALID_TASKS:
                if task_name not in task_service_map:
                    task_service_map[task_name] = first_available_service
                    self.logger.info(
                        f"Task '{task_name}' not explicitly preferred, assigned fallback service '{first_available_service}'.")
        else:
            # Should not happen if self.adapters check passed earlier
            self.logger.error("Fallback assignment failed: No enabled adapters found.")


        # Create the final mapped_tasks dictionary
        self.mapped_tasks.clear()
        for task_name, service_name in task_service_map.items():
            adapter_instance = self.adapters.get(service_name)
            service_cfg = configured_services.get(service_name)

            if adapter_instance and service_cfg:
                try:
                    # Create ModelConfig using the service's default model info
                    # Let adapter handle temp/max_tokens defaults later if needed
                    model_config = ModelConfig(
                        name=service_cfg.model, # Default model for the service
                        temperature=0.7, # Default temp, can be overridden
                        max_tokens=None # Default max_tokens, can be overridden
                    )
                    self.mapped_tasks[task_name] = (adapter_instance, model_config)
                    self.logger.info(f"Task '{task_name}' successfully mapped to Service '{service_name}' "
                                     f"(Adapter: {type(adapter_instance).__name__}) "
                                     f"with Default Model '{model_config.name}'")
                except Exception as e:
                    self.logger.error(
                        f"Failed to create ModelConfig for task '{task_name}' using service '{service_name}' config: {e}. Skipping task mapping.")
            else:
                self.logger.error(
                    f"Internal error: Could not find adapter/config for service '{service_name}' while mapping task '{task_name}'.")

        # Final check and logging
        mapped_tasks_count = len(self.mapped_tasks)
        unmapped_tasks = self.VALID_TASKS - set(self.mapped_tasks.keys())
        self.logger.info(
            f"LLM Task Mapping Complete: {mapped_tasks_count}/{len(self.VALID_TASKS)} standard tasks mapped.")
        if unmapped_tasks:
            self.logger.warning(f"Tasks not mapped (no suitable service found/enabled?): {', '.join(unmapped_tasks)}")


    def _create_adapter(self, service_name: str, service_cfg: Union[LocalServiceConfig, RemoteServiceConfig]) -> Optional[BaseLLMAdapter]:
        """Helper function to instantiate the correct adapter based on config."""
        config_dict = service_cfg.dict(exclude_none=True)

        # Use OpenAIAdapter for 'remote' (openai, azure, openrouter etc.) and potentially 'local' if compatible
        if service_name == 'remote' or service_name == 'local':
            # Assume local uses an OpenAI-compatible endpoint structure
            # Provider hint helps adapter internally (e.g., Azure URL structuring)
            try:
                # Pass the prompt manager instance during initialization
                return OpenAIAdapter(config_dict, self.prompt_manager)
            except ImportError as e:
                # OpenAIAdapter itself should not raise ImportError unless httpx/tiktoken missing
                self.logger.error(f"Core dependency missing for OpenAIAdapter: {e}")
                # Re-raise as ConfigurationError, as it means setup is incomplete
                raise ConfigurationError(f"Missing core dependency for LLM communication: {e}")
            except LLMServiceError as e:
                # Catch config errors specific to the adapter init (like missing keys/URL for Azure)
                self.logger.error(f"Configuration error initializing OpenAIAdapter for '{service_name}': {e}")
                raise ConfigurationError(f"Adapter config error for '{service_name}': {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error initializing OpenAIAdapter for '{service_name}': {e}", exc_info=True)
                raise ConfigurationError(f"Unexpected adapter init error for '{service_name}': {e}")
        else:
            # If other adapter types were supported, they would be handled here
            raise ConfigurationError(f"Unsupported service type '{service_name}' encountered during adapter creation.")


    def get_task_handler(self, task_name: Literal['analysis', 'generation', 'validation']) -> 'LLMTaskService':
        """
        Retrieve the configured adapter and model settings for a specific task.

        Args:
            task_name: The name of the task ('analysis', 'generation', 'validation').

        Returns:
            An LLMTaskService instance containing the appropriate adapter and model configuration.

        Raises:
            LLMServiceError: If the specified task is not mapped to an available service.
        """
        task_tuple = self.mapped_tasks.get(task_name)
        if not task_tuple:
            raise LLMServiceError(
                f"Task '{task_name}' is not mapped to an available & enabled service in the configuration.")

        adapter_instance, model_config = task_tuple
        return LLMTaskService(adapter=adapter_instance, model_config=model_config, prompt_manager=self.prompt_manager)

    async def close_adapters(self) -> None:
        """Close connections for all initialized adapters."""
        self.logger.info("Closing LLM adapter connections...")
        closed_count = 0
        tasks = []
        active_adapters = list(self.adapters.items()) # Create list to iterate over safely

        for service_name, adapter in active_adapters:
            if hasattr(adapter, 'close') and asyncio.iscoroutinefunction(adapter.close):
                # Use adapter class name and service name for clarity
                task_name = f"close_{service_name}_{adapter.__class__.__name__}"
                tasks.append(asyncio.create_task(adapter.close(), name=task_name))
                closed_count += 1
            elif hasattr(adapter, 'close') and callable(adapter.close): # Handle sync close if necessary
                try:
                    adapter.close()
                    closed_count += 1
                except Exception as e:
                    self.logger.error(f"Error closing synchronous adapter for service '{service_name}': {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                task_name = task.get_name() # Get the name we set
                if isinstance(result, Exception):
                     self.logger.error(f"Error during async close for adapter task '{task_name}': {result}")

        self.logger.info(f"Attempted to close {closed_count}/{len(active_adapters)} adapter connections.")
        self.adapters.clear() # Clear the dictionary after attempting close


class LLMTaskService:
    """
    A wrapper bundling an LLM adapter instance, its ModelConfig for a specific task,
    and the PromptManager. Facilitates executing LLM-based tasks.
    """

    def __init__(self, adapter: BaseLLMAdapter, model_config: ModelConfig, prompt_manager: PromptManager):
        self.adapter = adapter
        self.model_config = model_config
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)

    async def analyze_chunk(self, chunk: Chunk) -> AnalysisResult:
        """Analyze a chunk using the configured adapter and model."""
        self.logger.debug(
            f"Executing 'analysis' task with adapter '{type(self.adapter).__name__}' and model '{self.model_config.name}'.")
        # Correctly call the adapter's analyze_chunk method
        return await self.adapter.analyze_chunk(chunk=chunk, model_config=self.model_config)

    async def generate_questions(self,
                                 chunk: Chunk,
                                 analysis: AnalysisResult,
                                 category_counts: Dict[str, int]) -> List[Question]:
        """Generate questions using the configured adapter and model."""
        self.logger.debug(
            f"Executing 'generation' task with adapter '{type(self.adapter).__name__}' and model '{self.model_config.name}'.")
        # Correctly call the adapter's generate_questions method
        return await self.adapter.generate_questions(
            chunk=chunk,
            analysis=analysis,
            model_config=self.model_config, # Pass model config
            category_counts=category_counts # Pass counts
        )

    async def validate_question(self, question: Question, chunk: Chunk) -> Dict[str, Any]:
        """Validate a question using the configured adapter and model."""
        self.logger.debug(
            f"Executing 'validation' task with adapter '{type(self.adapter).__name__}' and model '{self.model_config.name}'.")
        # Correctly call the adapter's validate_question method
        return await self.adapter.validate_question(
            question=question,
            chunk=chunk,
            model_config=self.model_config # Pass model config
        )
