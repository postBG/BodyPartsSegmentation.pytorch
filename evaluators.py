from abc import ABCMeta, abstractmethod


class AbstractBaseEvaluator(metaclass=ABCMeta):

    @abstractmethod
    @property
    def worst_score(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def mode(self):
        raise NotImplementedError

    @abstractmethod
    def score(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        raise NotImplementedError
