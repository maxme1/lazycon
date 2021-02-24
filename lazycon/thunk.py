from threading import Lock
from typing import Dict

from .wrappers import Wrapper, PatternAssignment

ScopeDict = Dict[str, Wrapper]


class Thunk:
    def match(self, name):
        raise NotImplementedError


class ValueThunk(Thunk):
    def __init__(self, value):
        assert not isinstance(value, Thunk)
        self._value = value
        self.ready = True

    def match(self, name):
        return self._value


class NodeThunk(Thunk):
    def __init__(self, statement):
        self.lock = Lock()
        self.statement = statement
        self.ready = False
        self._value = None

    @staticmethod
    def _match(name, pattern):
        if isinstance(pattern, str):
            yield name == pattern, []
            return

        assert isinstance(pattern, tuple)
        min_size = max_size = len(pattern)
        for idx, entry in enumerate(pattern):
            level = idx, min_size, max_size
            for match, levels in NodeThunk._match(name, entry):
                yield match, [level] + levels

    def set(self, value):
        assert not self.ready
        self._value = value
        self.ready = True

    def match(self, name):
        assert self.ready
        value = self._value
        # TODO: probably need a subclass
        if not isinstance(self.statement, PatternAssignment):
            return value

        pattern = self.statement.pattern
        if isinstance(pattern, str):
            return value

        for match, levels in self._match(name, pattern):
            if match:
                for idx, min_size, max_size in levels:
                    size = len(value)
                    if size < min_size:
                        raise ValueError('not enough values to unpack (expected %d)' % max_size)
                    if size > max_size:
                        raise ValueError('too many values to unpack (expected %d)' % max_size)

                    value = value[idx]

                return value

        # unreachable code
        assert False
