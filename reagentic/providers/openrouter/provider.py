import os
from enum import Enum
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from ...env_manager import get_env_var
from ..base_provider import BaseProvider
from ..common import ModelInfo


class ConfigConstants:
    URL = 'https://openrouter.ai/api/v1'
    API_KEY_VAR = 'OPENROUTER_API_KEY'


class OpenrouterProvider(BaseProvider):
    def __init__(self, model: ModelInfo, key: str = None, disable_tracing: bool = True):
        """
        Initialize OpenRouter provider.

        Args:
            model: ModelInfo instance containing model details
            key: Optional API key, if not provided will use environment variable
            disable_tracing: Whether to disable tracing to prevent OpenAI API key messages.
                           Defaults to True since OpenRouter doesn't need OpenAI tracing.
        """
        self._model = model
        self._url = ConfigConstants.URL
        self._api_key = key or get_env_var(ConfigConstants.API_KEY_VAR)

        # Disable tracing by default for OpenRouter to prevent OpenAI API key messages
        if disable_tracing:
            set_tracing_disabled(True)

    @property
    def url(self) -> str:
        return self._url

    @property
    def model(self) -> ModelInfo:
        return self._model

    def get_client(self) -> AsyncOpenAI:
        """
        Creates an OpenAI client configured for OpenRouter.
        """
        return AsyncOpenAI(
            base_url=self.url,
            api_key=self._api_key,
        )

    def get_openai_model(self) -> OpenAIChatCompletionsModel:
        """
        Creates a run_config for an Agent using the OpenRouter provider.
        """
        openai_client = self.get_client()
        return OpenAIChatCompletionsModel(model=self.model.str_identifier, openai_client=openai_client)
