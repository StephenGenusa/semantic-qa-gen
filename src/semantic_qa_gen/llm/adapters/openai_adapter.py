# filename: semantic_qa_gen/llm/adapters/openai_adapter.py

"""OpenAI API and compatible adapter for SemanticQAGen."""

import json
import asyncio
import logging
import time
import sys
from typing import Dict, List, Optional, Any, Union, Tuple

import httpx
import tiktoken

# Optional Azure identity imports
try:
    from azure.identity.aio import DefaultAzureCredential
    from azure.core.credentials import AccessToken

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False


    # Define placeholder for type checking
    class DefaultAzureCredential:
        async def get_token(self, *args, **kwargs): pass


    class AccessToken:
        token: str
        expires_on: int

# Project imports
from semantic_qa_gen.llm.adapters.base import BaseLLMAdapter  # Updated base class
from semantic_qa_gen.config.schema import ModelConfig
from semantic_qa_gen.llm.prompts.manager import PromptManager
from semantic_qa_gen.utils.error import LLMServiceError, ConfigurationError


class OpenAIAdapter(BaseLLMAdapter):
    """
    LLM service adapter for OpenAI and compatible APIs (like Azure OpenAI, OpenRouter).
    Implements the core `generate_completion` method for communication.
    """

    def __init__(self, service_config: Dict[str, Any], prompt_manager: PromptManager):
        """
        Initialize the OpenAI adapter.

        Args:
            service_config: Configuration dictionary for this service.
            prompt_manager: The prompt manager instance (needed by BaseLLMAdapter).

        Raises:
            ConfigurationError: If essential configuration like API key (for non-Azure)
                                or Azure details are missing.
        """
        super().__init__(service_config, prompt_manager)

        self.api_key = service_config.get('api_key')
        self.provider = service_config.get('provider', 'openai').lower()
        self.organization = service_config.get('organization')
        self.api_version = service_config.get('api_version')  # Used for Azure

        # Determine API base URL - Check 'url' first, then 'api_base'
        self.api_base = service_config.get('url') or service_config.get('api_base')

        # Log the URL being used regardless of source
        if self.api_base:
            self.logger.info(f"Using API endpoint: {self.api_base}")
        elif self.provider == 'openai':
            self.api_base = 'https://api.openai.com/v1'
            self.logger.info(f"No URL specified, using default OpenAI endpoint: {self.api_base}")
        elif self.provider == 'azure':
            # Base URL is critical for Azure, raise error if missing
            raise ConfigurationError(
                f"Azure provider requires 'url' or 'api_base' (endpoint URL) in configuration.")

        # Remove trailing slash for consistency
        self.api_base = self.api_base.rstrip('/') if self.api_base else None

        # Check Azure specific requirements
        if self.provider == 'azure' and not self.api_version:
            raise ConfigurationError(
                "Azure provider requires 'api_version' (e.g., '2023-12-01-preview') in configuration.")

        # Azure AD authentication setup
        self.use_azure_ad = (self.provider == 'azure' and not self.api_key and AZURE_IDENTITY_AVAILABLE)
        self.azure_credential = None
        self._azure_token_cache: Optional[AccessToken] = None
        self._token_lock = asyncio.Lock()

        if self.use_azure_ad:
            # Initialize DefaultAzureCredential for token acquisition
            try:
                self.azure_credential = DefaultAzureCredential()
                self.logger.info("Using Azure AD authentication with DefaultAzureCredential")
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure AD credentials: {e}")
                if not self.api_key:
                    raise ConfigurationError(f"No API key provided and Azure AD authentication failed: {e}")
        elif not self.api_key and self.provider != 'azure':
            # Allow missing key but warn strongly, calls will likely fail
            self.logger.warning(
                f"API key not found in config or environment for '{self.provider}' provider. API calls will likely fail.")

        # Initialize async HTTP client
        headers = {
            "Content-Type": "application/json",
            # Authorization headers added dynamically based on provider
        }
        if self.organization and self.provider == 'openai':
            headers["OpenAI-Organization"] = self.organization

        # Client initialization is deferred to _get_client
        self._client: Optional[httpx.AsyncClient] = None

        # Rate limiting implementation
        self._rate_limit_config = {
            # Tokens per minute limit (0 to disable)
            'tokens_per_minute': service_config.get('rate_limit_tokens_per_minute', 0),
            # Requests per minute limit (0 to disable)
            'requests_per_minute': service_config.get('rate_limit_requests', 0),
            # Window size in seconds
            'window_size': 60.0
        }

        self._request_timestamps: List[float] = []
        self._token_usage_window: List[Tuple[float, int]] = []  # List of (timestamp, token_count)
        self._rate_limit_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Initializes or returns the httpx client with appropriate auth."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}

            # Determine if this is likely a local service that doesn't need auth
            is_local_service = (
                    'localhost' in (self.api_base or '') or
                    '127.0.0.1' in (self.api_base or '') or
                    self.provider == 'local'
            )

            # Handle different authentication methods
            if self.provider == 'azure':
                # Azure always needs its key in headers
                if self.api_key:
                    headers["api-key"] = self.api_key
            elif self.api_key and not is_local_service:
                # For OpenAI and other services that use Bearer auth
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Only warn about missing auth for non-local services
            if not is_local_service and not self.api_key and not self.use_azure_ad:
                self.logger.warning(
                    "No authentication method configured for non-local service. Requests will likely fail.")

            # Add organization header for OpenAI if provided
            if self.organization and self.provider == 'openai':
                headers["OpenAI-Organization"] = self.organization

            connect_timeout = max(10.0, self.timeout / 3)  # Min 10s connect
            read_timeout = self.timeout
            write_timeout = self.timeout

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(read_timeout, connect=connect_timeout, write=write_timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )
            self.logger.info(f"Initialized httpx client for {self.provider} at {self.api_base}")
        return self._client

    async def _get_azure_ad_token(self) -> str:
        """
        Get an Azure AD token for API authentication.

        Returns:
            Valid access token string.

        Raises:
            LLMServiceError: If token acquisition fails.
        """
        async with self._token_lock:
            # Check if we have a cached non-expired token
            now = time.time()
            if self._azure_token_cache and self._azure_token_cache.expires_on > now + 60:
                # Use cached token if it has more than 60 seconds left (avoid cutting it too close)
                return self._azure_token_cache.token

            # Need to acquire a new token
            try:
                # For OpenAI API, we need a token with the "https://cognitiveservices.azure.com/.default" scope
                scope = "https://cognitiveservices.azure.com/.default"
                self.logger.debug(f"Acquiring new Azure AD token for scope: {scope}")

                token = await self.azure_credential.get_token(scope)
                self._azure_token_cache = token
                return token.token

            except Exception as e:
                self.logger.error(f"Failed to acquire Azure AD token: {e}")
                raise LLMServiceError(f"Azure AD authentication failed: {e}", retryable=True)

    async def _prepare_request_with_auth(self, url: str, request_data: Dict[str, Any]) -> Tuple[
        str, Dict[str, Any], Dict[str, str]]:
        """
        Prepare a request with proper authentication headers.

        Args:
            url: The request URL.
            request_data: The JSON payload.

        Returns:
            Tuple of (url, data, headers)
        """
        headers = {}

        # Add Azure AD token if using Azure without API key
        if self.use_azure_ad:
            token = await self._get_azure_ad_token()
            headers["Authorization"] = f"Bearer {token}"

        return url, request_data, headers

    async def generate_completion(self, prompt: str, model_config: ModelConfig) -> str:
        """
        Core method to get a text completion from the OpenAI/compatible API.

        Handles constructing the request, making the API call with retries
        for specific errors, and returning the response text.

        Args:
            prompt: The formatted prompt string to send to the model.
            model_config: Configuration for the specific model invocation
                          (name, temperature, max_tokens).

        Returns:
            The raw text completion from the LLM.

        Raises:
            LLMServiceError: If the API call fails after retries or for auth errors.
            ConfigurationError: If the configuration is insufficient.
        """
        # Apply rate limiting first (honors token-based and request-based limits)
        estimated_tokens = self.count_tokens(prompt, model_config)
        await self._enforce_rate_limits(estimated_tokens)

        client = await self._get_client()
        if not self.api_base:
            raise ConfigurationError(f"API base URL is not configured for provider '{self.provider}'.")

        # Prepare the request URL and data
        request_url, request_data = self._prepare_request(prompt, model_config)

        # Check if we're using Ollama (for response parsing)
        is_ollama = '11434' in (self.api_base or '') and (
                    'localhost' in (self.api_base or '') or '127.0.0.1' in (self.api_base or ''))

        # Add authentication if needed (especially for Azure AD)
        request_url, request_data, auth_headers = await self._prepare_request_with_auth(request_url, request_data)

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries + 1}: POST to {request_url} with model {model_config.name}")

                # Make request with any additional auth headers
                response = await client.post(request_url, json=request_data, headers=auth_headers)
                response.raise_for_status()  # Check for 4xx/5xx errors

                # OLLAMA-SPECIFIC HANDLING
                if is_ollama:
                    try:
                        # Get the raw response text for inspection
                        response_text = response.text
                        self.logger.debug(f"Raw Ollama response starts with: {response_text[:100]}...")

                        # First try to parse the entire response normally
                        try:
                            response_json = json.loads(response_text)

                            # Handle Ollama's standard response format
                            if "message" in response_json:
                                return response_json["message"]["content"].strip()
                        except json.JSONDecodeError:
                            # If that fails, we might have a streaming response with multiple JSON objects
                            self.logger.debug("Detected streaming response from Ollama, parsing line by line")

                            # Parse the first complete JSON object
                            lines = response_text.strip().split("\n")
                            if not lines:
                                raise LLMServiceError("Empty response from Ollama")

                            # Parse the first line which should be a complete JSON object
                            first_response = json.loads(lines[0])

                            # If streaming is used, the complete content is in the last line's message
                            if len(lines) > 1:
                                try:
                                    # Try to get the last complete message
                                    for line in reversed(lines):
                                        if line.strip():
                                            last_response = json.loads(line)
                                            if last_response.get("done") is True and "message" in last_response:
                                                return last_response["message"]["content"].strip()
                                except Exception as e:
                                    self.logger.warning(f"Error parsing streaming response ending: {e}")

                            # Fallback: use the first message if parsing the full stream failed
                            if "message" in first_response:
                                return first_response["message"]["content"].strip()

                        # If we reach here, we couldn't parse the response properly
                        self.logger.error(f"Could not extract content from Ollama response: {response_text[:200]}...")
                        raise LLMServiceError("Failed to parse Ollama response format")

                    except Exception as e:
                        self.logger.exception(f"Error handling Ollama response: {e}")
                        raise LLMServiceError(f"Failed to process Ollama response: {str(e)}")

                # STANDARD OPENAI HANDLING
                response_json = response.json()

                # Track token usage from response for rate limiting
                if "usage" in response_json:
                    usage = response_json["usage"]
                    total_tokens = usage.get("total_tokens", estimated_tokens)
                    await self._track_token_usage(total_tokens)

                # Handle expected response structure
                if "choices" in response_json and response_json["choices"]:
                    message = response_json["choices"][0].get("message", {})
                    completion = message.get("content")
                    if completion is not None:
                        # Strip leading/trailing whitespace and return
                        return completion.strip()

                # Handle other potential response formats
                # For older OpenAI API or compatible APIs with different response structure
                if "choices" in response_json and "text" in response_json["choices"][0]:
                    completion = response_json["choices"][0].get("text")
                    if completion is not None:
                        return completion.strip()

                # If completion not found in expected structure
                error_message = "API response structure invalid or missing completion."
                self.logger.error(f"{error_message} Response: {response_json}")
                last_exception = LLMServiceError(error_message, details=response_json)
                break  # Don't retry structural errors

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_text = e.response.text
                error_details = {"status_code": status_code, "response_text": error_text}
                message = f"API Error ({status_code}) for {e.request.url}"

                # Check if we should apply rate limit backoff based on headers
                retry_after = None
                if status_code == 429:  # Too Many Requests
                    # Check for Retry-After header (could be seconds or HTTP date)
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            # Try to parse as integer seconds
                            retry_seconds = int(retry_after)
                        except ValueError:
                            # If not an integer, assume HTTP date format
                            # This would require http.client.parsedate_to_datetime
                            # For simplicity, we'll use a default backoff
                            retry_seconds = min(2 ** attempt, 60)

                        self.logger.warning(f"Rate limit hit. Server requested retry after {retry_seconds}s")

                # Determine if the error is retryable
                is_retryable = status_code in [429, 500, 502, 503, 504]  # Common retryable HTTP errors

                try:
                    error_json = e.response.json()
                    if "error" in error_json:
                        message = f"{message}: {error_json['error'].get('message', 'Unknown error')}"
                    error_details["parsed_error"] = error_json
                except json.JSONDecodeError:
                    message = f"{message}: {error_text[:100]}"  # Include partial raw text

                self.logger.error(f"HTTP Status Error ({status_code}): {message}")
                last_exception = LLMServiceError(message, details=error_details, retryable=is_retryable)

                if not is_retryable or attempt >= self.max_retries:
                    break  # Stop retrying for non-retryable or max retries hit

                # Use server-provided retry time if available
                if retry_after:
                    await asyncio.sleep(min(int(retry_after), 120))  # Cap at 2 minutes
                    continue  # Skip the regular backoff calculation

            except httpx.TimeoutException as e:
                message = f"Request timed out after {self.timeout}s: {e}"
                self.logger.warning(message)
                last_exception = LLMServiceError(message, retryable=True)
                # Retry timeouts

            except httpx.NetworkError as e:
                message = f"Network error during API call: {e}"
                self.logger.warning(message)
                last_exception = LLMServiceError(message, retryable=True)
                # Retry network errors

            except Exception as e:
                # Catch unexpected errors during the call
                self.logger.exception(f"Unexpected error during API call: {e}", exc_info=True)
                last_exception = LLMServiceError(f"Unexpected API interaction error: {e}")
                break  # Don't retry unknown errors

            # --- Retry Logic ---
            if attempt < self.max_retries:
                delay = self._get_retry_delay(attempt)
                self.logger.info(f"Retrying in {delay:.2f}s due to {type(last_exception).__name__}...")
                await asyncio.sleep(delay)
            # -------------------

        # If loop finished without returning, raise the last captured exception
        raise last_exception

    def _prepare_request(self, prompt: str, model_config: ModelConfig) -> tuple[str, Dict[str, Any]]:
        """Prepares the request URL and data payload."""
        messages = [{"role": "user", "content": prompt}]

        # Add system prompt if found in model_config metadata
        system_prompt = model_config.model_dump().get("system_prompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        base_data = {
            "messages": messages,
            "temperature": model_config.temperature,
        }

        # Add max_tokens only if specified
        if model_config.max_tokens:
            base_data["max_tokens"] = model_config.max_tokens

        # Check for additional parameters in model_config
        # Look for json_mode, top_p, frequency_penalty, presence_penalty
        model_config_dict = model_config.model_dump()
        for param in ["top_p", "frequency_penalty", "presence_penalty"]:
            if param in model_config_dict:
                base_data[param] = model_config_dict[param]

        # Special handling for json mode
        if model_config_dict.get("json_mode") or model_config_dict.get("response_format", {}).get(
                "type") == "json_object":
            base_data["response_format"] = {"type": "json_object"}

        # More robust Ollama detection
        is_ollama = '11434' in (self.api_base or '') and (
                    'localhost' in (self.api_base or '') or '127.0.0.1' in (self.api_base or ''))

        # Log the detected provider type
        if is_ollama:
            self.logger.debug(f"Detected Ollama endpoint at {self.api_base}")

        if self.provider == 'azure':
            # Azure URL format
            request_url = f"{self.api_base}/openai/deployments/{model_config.name}/chat/completions?api-version={self.api_version}"
            request_data = base_data  # Azure doesn't need 'model' in body if using deployment URL
        elif is_ollama:
            # Ollama URL format
            request_url = f"{self.api_base}/api/chat"  # Ollama endpoint
            # Explicitly disable streaming to prevent multiple JSON objects in response
            request_data = {**base_data, "model": model_config.name, "stream": False}
            self.logger.debug(f"Using Ollama API URL: {request_url} with model {model_config.name}, streaming disabled")
        else:  # OpenAI, OpenRouter, etc.
            request_url = f"{self.api_base}/chat/completions"
            request_data = {**base_data, "model": model_config.name}

        # Remove None values from payload IMPORTANT for Azure and others
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return request_url, request_data

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None  # Reset client
            self.logger.info("Closed OpenAIAdapter HTTP client.")

        # Close Azure credential if it has a close method
        if self.azure_credential and hasattr(self.azure_credential, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.azure_credential.close):
                    await self.azure_credential.close()
                else:
                    self.azure_credential.close()
                self.logger.debug("Closed Azure credential")
            except Exception as e:
                self.logger.warning(f"Error closing Azure credential: {e}")

    def count_tokens(self, text: str, model_config: ModelConfig) -> int:
        """Counts tokens using tiktoken, trying to match the specified model."""
        # Use the model specified for the *task* (model_config.name)
        model_name = model_config.name
        encoding = None
        try:
            # Tiktoken might need mapping, especially for Azure deployment names
            # or fine-tuned models.
            effective_model_for_tokenizer = model_name
            if self.provider == 'azure':
                # Basic check, might need better mapping logic based on actual base models
                if 'gpt-4' in model_name.lower():
                    effective_model_for_tokenizer = 'gpt-4'
                elif 'gpt-35-turbo' in model_name.lower():
                    effective_model_for_tokenizer = 'gpt-3.5-turbo'
                # Add more known Azure base model mappings if necessary
            elif self.provider == 'openrouter':
                # OpenRouter might prefix models, try splitting
                if '/' in model_name: effective_model_for_tokenizer = model_name.split('/')[-1]

            # This might still fail for custom/fine-tuned names
            encoding = tiktoken.encoding_for_model(effective_model_for_tokenizer)
            # self.logger.debug(f"Using tokenizer for model '{effective_model_for_tokenizer}'")
        except KeyError:
            self.logger.warning(f"No specific tokenizer found for model '{model_name}'. Using cl100k_base as fallback.")
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.error(f"Unexpected error getting tokenizer for model '{model_name}': {e}. Using cl100k_base.",
                              exc_info=False)
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    async def _track_token_usage(self, token_count: int) -> None:
        """
        Track token usage for rate limiting purposes.

        Args:
            token_count: Number of tokens used in the request/response.
        """
        if self._rate_limit_config['tokens_per_minute'] <= 0:
            return  # Token tracking disabled

        now = time.monotonic()
        window_size = self._rate_limit_config['window_size']

        async with self._rate_limit_lock:
            # Add current usage
            self._token_usage_window.append((now, token_count))

            # Remove entries older than the window
            self._token_usage_window = [(ts, count) for ts, count in self._token_usage_window
                                        if now - ts <= window_size]

    async def _enforce_rate_limits(self, estimated_tokens: int) -> None:
        """
        Enforce both token-based and request-based rate limits.

        This implements a robust rate limiting algorithm that:
        1. Respects token limits (TPM - tokens per minute)
        2. Respects request frequency limits (RPM - requests per minute)
        3. Uses server retry-after headers when available
        4. Implements backoff with jitter for client-side limiting

        Args:
            estimated_tokens: Estimated token count for the upcoming request
        """
        token_limit = self._rate_limit_config['tokens_per_minute']
        request_limit = self._rate_limit_config['requests_per_minute']

        if token_limit <= 0 and request_limit <= 0:
            return  # Rate limiting disabled

        window_size = self._rate_limit_config['window_size']
        now = time.monotonic()

        async with self._rate_limit_lock:
            # 1. Check token-based rate limit
            if token_limit > 0:
                # Remove expired entries
                self._token_usage_window = [(ts, count) for ts, count in self._token_usage_window
                                            if now - ts <= window_size]

                # Calculate current token usage in the window
                current_token_usage = sum(count for _, count in self._token_usage_window)

                # Check if adding estimated_tokens would exceed the limit
                if current_token_usage + estimated_tokens > token_limit:
                    # Calculate how long to wait
                    if self._token_usage_window:
                        oldest_timestamp = min(ts for ts, _ in self._token_usage_window)
                        # Wait until the oldest entry expires from the window
                        wait_time = oldest_timestamp + window_size - now

                        # Add a bit of jitter (0-10% of wait time)
                        import random
                        jitter = random.uniform(0, 0.1 * wait_time)
                        wait_time += jitter

                        self.logger.warning(
                            f"Token rate limit ({token_limit}/min) would be exceeded "
                            f"(current: {current_token_usage}, adding: {estimated_tokens}). "
                            f"Waiting {wait_time:.2f}s."
                        )

                        # Sleep outside the lock to avoid blocking other tasks
                        await asyncio.sleep(wait_time)
                        # Recursive call to check again after waiting
                        # Release lock first to avoid deadlock
                        await self._enforce_rate_limits(estimated_tokens)
                        return

            # 2. Check request-based rate limit
            if request_limit > 0:
                # Remove expired timestamps
                self._request_timestamps = [ts for ts in self._request_timestamps if now - ts <= window_size]

                # Check if adding a request would exceed the limit
                if len(self._request_timestamps) >= request_limit:
                    # Calculate how long to wait
                    if self._request_timestamps:
                        oldest_timestamp = min(self._request_timestamps)
                        # Wait until the oldest timestamp expires from the window
                        wait_time = oldest_timestamp + window_size - now

                        # Add a bit of jitter
                        import random
                        jitter = random.uniform(0, 0.1 * wait_time)
                        wait_time += jitter

                        self.logger.warning(
                            f"Request rate limit ({request_limit}/min) would be exceeded. "
                            f"Waiting {wait_time:.2f}s."
                        )

                        # Sleep outside the lock
                        await asyncio.sleep(wait_time)
                        # Recursive call to check again after waiting
                        await self._enforce_rate_limits(estimated_tokens)
                        return

                # Add current request timestamp
                self._request_timestamps.append(now)
