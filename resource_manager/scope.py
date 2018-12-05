import builtins
import importlib
import sys
from threading import Lock
from typing import Dict, Any

from .exceptions import BadSyntaxError
from .statements import Statement, ExpressionStatement, UnifiedImport


def add_if_missing(target: dict, name, node):
    if name in target:
        raise BadSyntaxError('Duplicate definition of resource "%s" in %s' % (name, node.source()))
    target[name] = node


class Thunk:
    pass


class ValueThunk(Thunk):
    def __init__(self, value):
        assert not isinstance(value, Thunk)
        self.value = value
        self.ready = True


class NodeThunk(Thunk):
    def __init__(self, statement: Statement):
        self.lock = Lock()
        self.statement = statement
        self.ready = False
        self.value = None


class Scope(Dict[str, Any]):
    def __init__(self):
        super().__init__()
        self._parent = vars(builtins)
        self._thunks = {}

    def add_statement(self, name, statement):
        assert name not in self
        if statement not in self._thunks:
            self._thunks[statement] = NodeThunk(statement)
        super().__setitem__(name, self._thunks[statement])

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __getitem__(self, name: str):
        if name not in self:
            return self._parent[name]

        thunk = super().__getitem__(name)
        if thunk.ready:
            return thunk.value

        assert isinstance(thunk, NodeThunk)
        with thunk.lock:
            if not thunk.ready:
                thunk.value = render(thunk.statement, self)
                thunk.ready = True

            return thunk.value


def render(statement: Statement, scope: Scope):
    if isinstance(statement, ExpressionStatement):
        code = compile(statement.body, statement.source, 'eval')
        return eval(code, {}, scope)

    if isinstance(statement, UnifiedImport):
        from_ = '.'.join(statement.root)
        what = '.'.join(statement.what)
        if not from_:
            result = importlib.import_module(what)
            packages = statement.what
            if len(packages) > 1 and not statement.as_:
                # import a.b.c
                return sys.modules[packages[0]]
            return result
        try:
            return getattr(importlib.import_module(from_), what)
        except AttributeError:
            pass
        try:
            return importlib.import_module(what, from_)
        except ModuleNotFoundError:
            pass
        return importlib.import_module(from_ + '.' + what)

    raise NotImplementedError
