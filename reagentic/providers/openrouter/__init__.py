from .available_models import *
from .provider import OpenrouterProvider

def auto():
    from ...agents.model_chooser import choose_model
    model_name = choose_model("openrouter", "best free model", False)
    info =  ALL_MODELS_DICT[model_name]
    return OpenrouterProvider(info)