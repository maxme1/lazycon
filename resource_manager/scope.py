import builtins
from ast import NameConstant
from collections import defaultdict, OrderedDict
from threading import Lock
from typing import Dict, Any

from .wrappers import Wrapper, UnifiedImport
from .renderer import Renderer
from .exceptions import ResourceError, SemanticError, ExceptionWrapper

ScopeDict = Dict[str, Wrapper]


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
    def __init__(self, injections: dict):
        base = dict(vars(builtins))
        common = set(base) & set(injections)
        if common:
            raise SemanticError('Some injections clash with builtins: ' + str(common))
        base.update(injections)
        super().__init__(base)

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise ResourceError('"%s" is not defined.' % name) from None


class Scope(OrderedDict):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._statement_to_thunk = {}
        self._populated = False
        self._updated = False

    def check_populated(self):
        if self._populated:
            raise RuntimeError('The scope has already been populated with live objects. Overwriting them might cause '
                               'undefined behaviour. Please, create another instance of ResourceManager.')

    def get_name_to_statement(self):
        statements = {v: k for k, v in self._statement_to_thunk.items()}
        return {name: statements[thunk] for name, thunk in self.items()}

    def render(self, order: dict):
        if self._updated:
            raise RuntimeError('The scope has already been updated by live objects that cannot be rendered properly.')

        names = self.get_name_to_statement()
        groups = defaultdict(list)
        for name, statement in names.items():
            groups[statement].append(name)

        import_groups, imports, definitions = defaultdict(list), [], []
        for statement, names in sorted(groups.items(), key=lambda x: min(order[name] for name in x[1])):
            pair = sorted(names), statement
            if isinstance(statement, UnifiedImport):
                if statement.root:
                    import_groups[statement.root, statement.dots].append(pair)
                else:
                    imports.append(pair)
            else:
                definitions.append(pair)

        for names, statement in imports:
            yield statement.to_str(names)

        if imports:
            yield ''

        for group in import_groups.values():
            names, statement = group[0]
            result = statement.to_str(names)
            for names, statement in group[1:]:
                assert len(names) == 1
                result += ', ' + statement.import_what(names[0])

            yield result

        if import_groups or imports:
            yield '\n'

        for names, statement in definitions:
            yield statement.to_str(names)

    def _set_thunk(self, name, thunk):
        super().__setitem__(name, thunk)

    def add_value(self, name, value):
        assert name not in self
        self._set_thunk(name, ValueThunk(value))

    # TODO: unify these functions
    def update_values(self, values: dict):
        self._updated = True
        self.check_populated()

        for name, value in values.items():
            statement = NameConstant(value)
            if statement not in self._statement_to_thunk:
                self._statement_to_thunk[statement] = ValueThunk(value)

            self._set_thunk(name, self._statement_to_thunk[statement])

    def update_statements(self, items):
        self.check_populated()

        for name, statement in items:
            if statement not in self._statement_to_thunk:
                self._statement_to_thunk[statement] = NodeThunk(statement)

            self._set_thunk(name, self._statement_to_thunk[statement])

    def __setitem__(self, key, value):
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
                self._populated = True
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
        except KeyError as e:
            # this is needed because KeyError is converted to NameError by `eval`
            raise ExceptionWrapper(e) from e
        except ResourceError:
            pass

        assert name in self, name
        return super().__getitem__(name)
