from threading import Lock
from typing import Any

from .statements import GlobalStatement, LiveObject


class Thunk:
    value: Any
    ready: bool


class ValueThunk(Thunk):
    def __init__(self, statement: LiveObject):
        self.statement = statement
        self.value = statement.value
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
