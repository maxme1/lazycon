import builtins
from collections import defaultdict
from threading import Lock
from typing import Dict, Any

from .wrappers import Wrapper, UnifiedImport
from .renderer import Renderer
from .exceptions import BadSyntaxError, ResourceError

ScopeDict = Dict[str, Wrapper]


def add_if_missing(target: dict, name, node):
    if name in target:
        raise BadSyntaxError('Duplicate definition of resource "%s" in %s' % (name, node.source_path))
    target[name] = node


class Thunk:
    pass


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
        self.value = None


class Builtins(dict):
    def __init__(self):
        super().__init__(vars(builtins))

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise ResourceError('"%s" is not defined.' % name) from None


class Scope(Dict[str, Any]):
    def __init__(self):
        super().__init__()
        self.parent = Builtins()
        self._statement_to_thunk = {}

    def render(self):
        statements = {v: k for k, v in self._statement_to_thunk.items()}
        names = {name: statements[thunk] for name, thunk in self.items()}
        groups = defaultdict(list)
        for name, statement in names.items():
            groups[statement].append(name)

        imports, definitions = [], []
        for statement, names in groups.items():
            pair = sorted(names), statement
            if isinstance(statement, UnifiedImport):
                imports.append(pair)
            else:
                definitions.append(pair)

        # TODO: smarter arrangement
        for names, statement in sorted(imports) + sorted(definitions):
            yield statement.to_str(names)

    def add_statement(self, name, statement):
        assert name not in self
        if statement not in self._statement_to_thunk:
            self._statement_to_thunk[statement] = NodeThunk(statement)

        super().__setitem__(name, self._statement_to_thunk[statement])

    def __setitem__(self, key, value):
        # TODO: move add_statement here
        raise NotImplementedError

    def __getitem__(self, name: str):
        if name not in self:
            return self.parent[name]

        thunk = super().__getitem__(name)
        if thunk.ready:
            return thunk.value

        assert isinstance(thunk, NodeThunk)
        with thunk.lock:
            if not thunk.ready:
                thunk.value = Renderer.render(thunk.statement, self)
                thunk.ready = True

            return thunk.value


class ScopeWrapper(Dict[str, Any]):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def __getitem__(self, name):
        try:
            return self.scope[name]
        except ResourceError:
            pass

        assert name in self
        return super().__getitem__(name)
