import abc
from typing import Union
from .common import ModelInfo


class BaseProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def url(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> Union[str, ModelInfo]:
        pass

    @abc.abstractmethod
    def get_client(self):
        pass

    @abc.abstractmethod
    def get_openai_model(self):
        pass
