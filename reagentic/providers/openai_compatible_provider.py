import abc
from typing import Optional
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from .base_provider import BaseProvider
from .common import ModelInfo
from ..env_manager import get_env_var
from ..logging import get_logger, provider_context


class OpenAICompatibleProvider(BaseProvider):
    """
    Base class for providers that are compatible with the OpenAI API format.
    
    This class provides common functionality for providers that use the OpenAI client
    and OpenAIChatCompletionsModel, such as OpenRouter, Together.ai, and others.
    """
    
    def __init__(
        self, 
        model: ModelInfo | str, 
        base_url: str,
        api_key_env_var: str,
        provider_name: str,
        key: Optional[str] = None, 
        disable_tracing: bool = True
    ):
        """
        Initialize OpenAI-compatible provider.

        Args:
            model: ModelInfo instance or string identifier (e.g., 'deepseek/deepseek-chat')
            base_url: Base URL for the API endpoint
            api_key_env_var: Environment variable name for the API key
            provider_name: Name of the provider for logging purposes
            key: Optional API key, if not provided will use environment variable
            disable_tracing: Whether to disable tracing to prevent OpenAI API key messages.
                           Defaults to True since most OpenAI-compatible providers 
                           don't need OpenAI tracing.
        """
        # Convert string to ModelInfo if needed
        if isinstance(model, str):
            model = ModelInfo.from_string(model)
        self._model = model
        self._url = base_url
        self._api_key = key or get_env_var(api_key_env_var)
        self._provider_name = provider_name

        # Initialize logger
        self._logger = get_logger(f'providers.{provider_name}')

        # Disable tracing by default for OpenAI-compatible providers
        if disable_tracing:
            set_tracing_disabled(True)

        # Log provider initialization
        with provider_context(provider_name, model.str_identifier):
            self._logger.log_provider_call(provider_name, 'initialize', model=model.str_identifier)

    @property
    def url(self) -> str:
        """Get the API base URL."""
        return self._url

    @property
    def model(self) -> ModelInfo:
        """Get the model information."""
        return self._model
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    def get_client(self) -> AsyncOpenAI:
        """
        Creates an OpenAI client configured for this provider.
        
        Returns:
            AsyncOpenAI client instance configured with the provider's base URL and API key
        """
        with provider_context(self._provider_name, self._model.str_identifier):
            self._logger.log_provider_call(self._provider_name, 'get_client', model=self._model.str_identifier)

            return AsyncOpenAI(
                base_url=self.url,
                api_key=self._api_key,
            )

    def get_openai_model(self) -> OpenAIChatCompletionsModel:
        """
        Creates an OpenAIChatCompletionsModel for use with agents.
        
        Returns:
            OpenAIChatCompletionsModel instance configured with this provider's client
        """
        with provider_context(self._provider_name, self._model.str_identifier):
            self._logger.log_provider_call(self._provider_name, 'get_openai_model', model=self._model.str_identifier)

            openai_client = self.get_client()
            model = OpenAIChatCompletionsModel(model=self.model.str_identifier, openai_client=openai_client)

            return model

    # Abstract methods that subclasses can override if needed
    def get_headers(self) -> dict:
        """
        Get additional headers for API requests.
        Subclasses can override this to add provider-specific headers.
        
        Returns:
            Dictionary of additional headers
        """
        return {}

    def get_client_kwargs(self) -> dict:
        """
        Get additional keyword arguments for the OpenAI client.
        Subclasses can override this to add provider-specific client configuration.
        
        Returns:
            Dictionary of additional client kwargs
        """
        return {} 