from abc import ABC, abstractmethod


class Stateful(ABC):
    @abstractmethod
    def save_pretrained(self, path: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def from_pretrained(path: str, inference_mode: bool) -> "Stateful":
        pass

    @staticmethod
    @abstractmethod
    def pretrained_exists(path: str) -> bool:
        pass

    @abstractmethod
    def train(self, data):
        pass
