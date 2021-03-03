from threading import Lock
from typing import Dict, Any

from .statements import GlobalStatement

ScopeDict = Dict[str, GlobalStatement]


class Thunk:
    value: Any
    ready: bool


class ValueThunk(Thunk):
    def __init__(self, value):
        assert not isinstance(value, Thunk)
        self.value = value
        self.ready = True


class NodeThunk(Thunk):
    def __init__(self, statement):
        self.lock = Lock()
        self.statement = statement
        self.ready = False
        self._value = None

    def set(self, value):
        assert not self.ready
        self._value = value
        self.ready = True

    @property
    def value(self):
        assert self.ready
        return self._value
