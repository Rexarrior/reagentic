from .available_models import *
from .provider import OpenrouterProvider

def auto(requirements: str | None = None, disable_tracing: bool = True):
    """
    Automatically choose and create an OpenRouter provider with the best free model.
    
    Args:
        requirements: Optional requirements for the model.
        disable_tracing: Whether to disable tracing to prevent OpenAI API key messages.
                        Defaults to True since OpenRouter doesn't need OpenAI tracing.
    
    Returns:
        OpenrouterProvider instance configured with the best free model
    """
    from ...agents.model_chooser import choose_model
    model_name = choose_model("openrouter", requirements or "best free model", False)
    info = ALL_MODELS_DICT[model_name]
    return OpenrouterProvider(info, disable_tracing=disable_tracing)