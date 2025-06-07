
from pydantic import BaseModel
class ModelInfo(BaseModel):
    'Info about some model'
    str_identifier: str
    'an str that identify this model in provider api'
    price_in: float
    'price per 1mln input token. If free, then 0'
    price_out: float
    'price per 1mln output token. If free, then 0'
    description: str
    "description of the model"
    
    def __str__(self):
        return self.str_identifier