from abc import ABCMeta, abstractmethod


class ITLEParser:
    __metaclass__ = ABCMeta

    @classmethod
    def version(self): return "1.0"

    @abstractmethod
    def parse(self, tle): raise NotImplementedError

    @abstractmethod
    def __extract(self, tle): raise NotImplementedError

    @abstractmethod
    def __validate(self, tle): raise NotImplementedError
