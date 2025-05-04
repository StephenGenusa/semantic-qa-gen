# filename: semantic_qa_gen/llm/router.py

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Literal, Union, List

# Use Pydantic V2 BaseModel explicitly
from pydantic import BaseModel, Field, ValidationError

from semantic_qa_gen.config.manager import ConfigManager
from semantic_qa_gen.document.models import Chunk, AnalysisResult, Question
from semantic_qa_gen.config.schema import LLMServiceConfig as ServiceDetails
from semantic_qa_gen.config.schema import SemanticQAGenConfig
from semantic_qa_gen.config.schema import LocalServiceConfig, RemoteServiceConfig

# Adapters
from semantic_qa_gen.llm.adapters.base import BaseLLMAdapter, ModelConfig
from semantic_qa_gen.llm.adapters.openai_adapter import OpenAIAdapter # Keep specific import

from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError, SemanticQAGenError

# --- Corrected LLMTaskService ---
class LLMTaskService(BaseModel):
    """
    A container bundling an LLM adapter instance, its ModelConfig for a task,
    and the PromptManager. Returned by TaskRouter.get_task_handler.
    """
    adapter: BaseLLMAdapter
    task_model_config: ModelConfig # Renamed field
    prompt_manager: PromptManager

    # Use model_config attribute for Pydantic V2 model configuration
    model_config = {
        "arbitrary_types_allowed": True
    }


# --- TaskRouter (Updated get_task_handler return type) ---
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
        self.prompt_manager = prompt_manager # Store prompt manager

        try:
            self.root_config: SemanticQAGenConfig = config_manager.config
            self.llm_services_config = getattr(self.root_config, 'llm_services', None)

            if not self.llm_services_config:
                raise ConfigurationError("LLM configuration section ('llm_services') is missing.")
        except AttributeError as e:
            raise ConfigurationError(f"Failed to access required configuration sections: {e}")

        self.adapters: Dict[str, BaseLLMAdapter] = {}
        # Store the mapped task details (Adapter, ModelConfig)
        self._mapped_task_details: Dict[str, Tuple[BaseLLMAdapter, ModelConfig]] = {}

        self._initialize_services_and_map_tasks()

    def _initialize_services_and_map_tasks(self) -> None:
        """Initialize adapters and map tasks based on the configuration."""
        self.logger.info("Initializing LLM adapters...")
        configured_services = {'local': self.llm_services_config.local, 'remote': self.llm_services_config.remote}

        for service_name, service_cfg in configured_services.items():
            if not service_cfg or not service_cfg.enabled:
                self.logger.info(f"Service '{service_name}' is not configured or is disabled. Skipping initialization.")
                continue
            try:
                adapter_instance = self._create_adapter(service_name, service_cfg)
                if adapter_instance:
                    self.adapters[service_name] = adapter_instance
                    self.logger.info(f"Initialized adapter for service '{service_name}' (Type: {type(adapter_instance).__name__})")
            except Exception as e:
                self.logger.error(f"Error initializing adapter for service '{service_name}': {e}", exc_info=True)
                raise ConfigurationError(f"Adapter initialization failed for '{service_name}': {e}") from e

        if not self.adapters:
             if any(cfg.enabled for cfg in configured_services.values() if cfg):
                 raise ConfigurationError("No LLM adapters were successfully initialized despite services being configured and enabled.")
             else:
                 self.logger.warning("No LLM services are configured or enabled.")
                 return

        self.logger.info("Mapping tasks to initialized LLM services...")
        task_service_map: Dict[str, str] = {}
        for service_name, adapter in self.adapters.items():
            service_cfg = configured_services.get(service_name)
            if not service_cfg: continue
            preferred_tasks = getattr(service_cfg, 'preferred_tasks', [])
            for task_name in preferred_tasks:
                if task_name not in self.VALID_TASKS: continue
                if task_name not in task_service_map:
                    task_service_map[task_name] = service_name
                    self.logger.debug(f"Task '{task_name}' initially mapped to service '{service_name}'.")
                # else: Task already mapped

        first_available_service = next((name for name in ['remote', 'local'] if name in self.adapters), None)
        if first_available_service:
            for task_name in self.VALID_TASKS:
                if task_name not in task_service_map:
                    task_service_map[task_name] = first_available_service
                    self.logger.info(f"Task '{task_name}' assigned fallback service '{first_available_service}'.")

        self._mapped_task_details.clear()
        for task_name, service_name in task_service_map.items():
             adapter_instance = self.adapters.get(service_name)
             service_cfg = configured_services.get(service_name)
             if adapter_instance and service_cfg:
                 try:
                     if not self.prompt_manager: raise ConfigurationError("PromptManager needed for task mapping.")
                     system_prompt_text = None
                     prompt_key_map = {'analysis': 'chunk_analysis', 'generation': 'question_generation', 'validation': 'question_validation'}
                     prompt_key = prompt_key_map.get(task_name)
                     if prompt_key:
                         try: system_prompt_text = self.prompt_manager.get_system_prompt(prompt_key)
                         except (LLMServiceError, Exception) as sp_err: self.logger.debug(f"Could not get system prompt for {task_name}: {sp_err}")

                     model_config_data = {"name": service_cfg.model, "temperature": 0.7, "max_tokens": None, "system_prompt": system_prompt_text}
                     model_config = ModelConfig(**{k:v for k,v in model_config_data.items() if v is not None})
                     self._mapped_task_details[task_name] = (adapter_instance, model_config)
                     self.logger.info(f"Task '{task_name}' mapped to Service '{service_name}' (Adapter: {type(adapter_instance).__name__}, Model: '{model_config.name}')")
                 except Exception as e:
                     self.logger.error(f"Failed ModelConfig creation for task '{task_name}', service '{service_name}': {e}. Skipping.")

        mapped_tasks_count = len(self._mapped_task_details)
        unmapped_tasks = self.VALID_TASKS - set(self._mapped_task_details.keys())
        self.logger.info(f"LLM Task Mapping Complete: {mapped_tasks_count}/{len(self.VALID_TASKS)} tasks mapped.")
        if unmapped_tasks: self.logger.warning(f"Tasks not mapped: {', '.join(unmapped_tasks)}")

    def _create_adapter(self, service_name: str, service_cfg: Union[LocalServiceConfig, RemoteServiceConfig]) -> Optional[BaseLLMAdapter]:
        """Helper function to instantiate the correct adapter based on config."""
        config_dict = service_cfg.model_dump(exclude_none=True)
        if service_name == 'remote' or service_name == 'local':
            try:
                return OpenAIAdapter(config_dict, self.prompt_manager)
            except ImportError as e:
                raise ConfigurationError(f"Missing core dependency for LLM communication: {e}") from e
            except ConfigurationError as e:
                self.logger.error(f"Configuration error initializing OpenAIAdapter for '{service_name}': {e}")
                raise
            except Exception as e:
                raise ConfigurationError(f"Unexpected adapter init error for '{service_name}': {e}") from e
        else:
            raise ConfigurationError(f"Unsupported service type '{service_name}' in adapter creation.")

    def get_task_handler(self, task_name: Literal['analysis', 'generation', 'validation']) -> LLMTaskService:
        """
        Retrieve the bundled service details (adapter, model config, prompt manager) for a task.

        Args:
            task_name: The name of the task ('analysis', 'generation', 'validation').

        Returns:
            An LLMTaskService instance containing the adapter, model config, and prompt manager.

        Raises:
            LLMServiceError: If the specified task is not mapped to an available service.
        """
        task_details = self._mapped_task_details.get(task_name)
        if not task_details:
            raise LLMServiceError(
                f"Task '{task_name}' is not mapped to an available & enabled service in the configuration.")

        adapter_instance, model_config = task_details
        # Use the renamed field 'task_model_config' here
        return LLMTaskService(
            adapter=adapter_instance,
            task_model_config=model_config, # Pass the ModelConfig object here
            prompt_manager=self.prompt_manager
        )

    async def close_adapters(self) -> None:
        """Close connections for all initialized adapters."""
        self.logger.info("Closing LLM adapter connections...")
        closed_count = 0
        tasks = []
        active_adapters = list(self.adapters.items())

        for service_name, adapter in active_adapters:
            if hasattr(adapter, 'close') and asyncio.iscoroutinefunction(adapter.close):
                tasks.append(asyncio.create_task(adapter.close(), name=f"close_{service_name}_{adapter.__class__.__name__}"))
                closed_count += 1
            elif hasattr(adapter, 'close') and callable(adapter.close):
                 try:
                       adapter.close()
                       closed_count += 1
                 except Exception as e:
                       self.logger.error(f"Error closing sync adapter '{service_name}': {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                     self.logger.error(f"Error during async close for '{task.get_name()}': {result}")

        self.logger.info(f"Attempted to close {closed_count}/{len(active_adapters)} adapter connections.")
        self.adapters.clear()
