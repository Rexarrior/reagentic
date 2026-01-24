from typing import Optional
from ..openai_compatible_provider import OpenAICompatibleProvider
from ..common import ModelInfo


class ConfigConstants:
    URL = 'https://openrouter.ai/api/v1'
    API_KEY_VAR = 'OPENROUTER_API_KEY'


class OpenrouterProvider(OpenAICompatibleProvider):
    """
    OpenRouter provider implementation.
    
    OpenRouter provides access to various AI models through a unified OpenAI-compatible API.
    """
    
    def __init__(self, model: ModelInfo | str, key: Optional[str] = None, disable_tracing: bool = True):
        """
        Initialize OpenRouter provider.

        Args:
            model: ModelInfo instance or string identifier (e.g., 'deepseek/deepseek-chat')
            key: Optional API key, if not provided will use environment variable
            disable_tracing: Whether to disable tracing to prevent OpenAI API key messages.
                           Defaults to True since OpenRouter doesn't need OpenAI tracing.
        """
        super().__init__(
            model=model,
            base_url=ConfigConstants.URL,
            api_key_env_var=ConfigConstants.API_KEY_VAR,
            provider_name='openrouter',
            key=key,
            disable_tracing=disable_tracing
        )
