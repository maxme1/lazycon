import builtins
from ast import NameConstant
from collections import defaultdict, OrderedDict
from threading import Lock
from typing import Dict, Any, Set, Sequence

from .utils import reverse_mapping
from .wrappers import Wrapper, UnifiedImport, PatternAssignment
from .renderer import Renderer
from .exceptions import ResourceError, SemanticError, ExceptionWrapper

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

    def _get_leave_time(self, parents: Dict[Wrapper, Set[Wrapper]], entry_points: Sequence[str]):
        def mark_name(name):
            nonlocal current
            if name not in leave_time:
                leave_time[name] = current
                current += 1

        def visit_parents(node):
            visited.add(node)
            for parent in parents[node]:
                find_leave_time(parent)

        def find_leave_time(node):
            if node in visited:
                return

            visit_parents(node)
            for name in statements[node]:
                mark_name(name)

        names = self.get_name_to_statement()
        statements = reverse_mapping(names)
        if entry_points is None:
            entry_points = list(names)
        else:
            delta = set(entry_points) - set(names)
            if delta:
                raise ValueError(f'The names {delta} are not defined, and cannot be used as entry points.')

        leave_time = {}
        visited = set()
        current = 0
        # we can't just visit the first-level nodes because some of them may have several names
        #  we need to drop such cases
        for n in entry_points:
            visit_parents(names[n])
            mark_name(n)

        names = {n: names[n] for n in leave_time}
        return names, leave_time

    def render(self, parents: Dict[Wrapper, Set[Wrapper]], entry_points: Sequence[str] = None):
        if self._updated:
            raise RuntimeError('The scope has already been updated by live objects that cannot be rendered properly.')

        # grouping imports
        names, order = self._get_leave_time(parents, entry_points)
        groups = defaultdict(list)
        for name, statement in names.items():
            groups[statement].append(name)

        import_groups, imports, definitions = defaultdict(list), [], []
        for statement, names in sorted(groups.items(), key=lambda x: min(order[n] for n in x[1])):
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
            return thunk.match(name)

        assert isinstance(thunk, NodeThunk)
        with thunk.lock:
            if not thunk.ready:
                self._populated = True
                thunk.set(Renderer.render(thunk.statement, self))

            return thunk.match(name)


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

        if name not in self:
            raise NameError(f'The name "{name}" is not defined.')
        return super().__getitem__(name)

    def __contains__(self, name):
        return name in self.scope or super().__contains__(name)
