from typing import List
from pydantic import BaseModel
from ...providers.common import ModelInfo


class AllOpenRouterModels(BaseModel): 
    'container with information about all models'
    all_models: List[ModelInfo]

