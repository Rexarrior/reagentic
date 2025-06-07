import os
from enum import Enum
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from ...env_manager import get_env_var
from ..base_provider import BaseProvider
from ..common import ModelInfo

    

class ConfigConstants:
    URL = "https://openrouter.ai/api/v1"
    API_KEY_VAR = "OPENROUTER_API_KEY"

class OpenrouterProvider(BaseProvider):
    def __init__(self, model: ModelInfo, key: str = None):
        self._model = model
        self._url = ConfigConstants.URL
        self._api_key = key or  get_env_var(ConfigConstants.API_KEY_VAR)

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
        return OpenAIChatCompletionsModel(
            model=self.model.str_identifier,
            openai_client=openai_client
        )