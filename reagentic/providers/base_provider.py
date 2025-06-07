import abc

class BaseProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def url(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model(self) -> str:
        pass

    @abc.abstractmethod
    def get_client(self):
        pass

    @abc.abstractmethod
    def get_run_config(self):
        pass