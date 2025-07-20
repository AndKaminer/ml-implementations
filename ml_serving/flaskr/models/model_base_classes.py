from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def predict(self, x):
        pass

