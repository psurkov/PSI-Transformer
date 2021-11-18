from abc import ABC, abstractmethod

from flccpsi.connectors.settings import GenerationSettings


class Connector(ABC):
    @abstractmethod
    def get_suggestions(self, prime: str, filename: str, language: str, settings: GenerationSettings):
        pass

    @abstractmethod
    def cancel(self):
        pass
