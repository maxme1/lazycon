from threading import Lock
from typing import Any

from .statements import GlobalStatement


class Thunk:
    value: Any
    ready: bool


class ValueThunk(Thunk):
    def __init__(self, value: Any):
        assert not isinstance(value, Thunk)
        self.value = value
        self.ready = True


class NodeThunk(Thunk):
    def __init__(self, statement: GlobalStatement, lock: Lock):
        self.lock = lock
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
