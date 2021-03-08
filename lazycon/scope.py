import ast
import builtins
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Sequence

from .semantics.analyzer import NodeParents
from .thunk import ValueThunk, NodeThunk, Thunk
from .utils import reverse_mapping
from .statements import GlobalStatement, GlobalImport, GlobalImportFrom
from .renderer import Renderer
from .exceptions import EntryError, ExceptionWrapper, SemanticError

ScopeDict = Dict[str, GlobalStatement]


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
            raise EntryError(f'"{name}" is not defined.') from None


class Scope:
    """
    A lazy global scope.

    Internal logic:
        - each name is bound to a single statement
        - each statement has a target that can be rendered
        - multiple targets can map to the same statement: `a = b = f(...)`
    """

    def __init__(self, statements: Sequence[GlobalStatement], parent, parents: NodeParents):
        super().__init__()
        self.statements = statements
        self._parent = parent
        # TODO: simplify this by exec-ing assignments?
        self._node_to_thunk: Dict[ast.AST, Thunk] = {}
        self._name_to_thunk: OrderedDict[str, Thunk] = OrderedDict()
        self._parents = parents

        self._populated = False
        self._updated = False

        for statement in statements:
            target = statement.target
            if target not in self._node_to_thunk:
                self._node_to_thunk[target] = NodeThunk(statement)

            self._name_to_thunk[statement.name] = self._node_to_thunk[target]

    def __getitem__(self, name: str):
        """ Render a name """
        if name not in self._name_to_thunk:
            return self._parent[name]

        thunk = self._name_to_thunk[name]
        if thunk.ready:
            return thunk.value

        assert isinstance(thunk, NodeThunk)
        with thunk.lock:
            if not thunk.ready:
                self._populated = True
                thunk.set(Renderer.render(thunk.statement, self))

            return thunk.value

    def __contains__(self, name):
        return name in self._name_to_thunk or name in self._parent

    def check_populated(self):
        if self._populated:
            raise RuntimeError('The scope has already been populated with live objects. Overwriting them might cause '
                               'undefined behaviour. Please, create another instance of Config.')

    def keys(self):
        return self._name_to_thunk.keys()

    def update_values(self, values: Dict[str, Any]):
        self.check_populated()
        self._updated = True

        for name, value in values.items():
            self._name_to_thunk[name] = ValueThunk(value)

    def _get_name_to_statement(self):
        return {x.name: x for x in self.statements}

    def _get_leave_time(self, entry_points: Sequence[str]):
        def mark_name(name):
            nonlocal current
            if name not in leave_time:
                leave_time[name] = current
                current += 1

        def visit_parents(node):
            visited.add(node)
            for parent in self._parents[node]:
                find_leave_time(parent)

        def find_leave_time(node):
            if node in visited:
                return

            visit_parents(node)
            for name in statements[node]:
                mark_name(name)

        names = self._get_name_to_statement()
        statements = reverse_mapping(names)
        if entry_points is None:
            entry_points = list(names)
        else:
            delta = set(entry_points) - set(names)
            if delta:
                raise ValueError('The names %s are not defined, and cannot be used as entry points.' % delta)

        leave_time = {}
        visited = set()
        current = 0
        # we can't just visit the first-level nodes because some of them may have several names
        #  we need to drop such cases
        for n in entry_points:
            visit_parents(names[n])
            mark_name(n)

        names = [names[n] for n in leave_time]
        return names, leave_time

    def render(self, entry_points: Sequence[str] = None):
        if self._updated:
            raise RuntimeError('The scope has already been updated by live objects that cannot be rendered properly.')

        # grouping imports
        statements, order = self._get_leave_time(entry_points)
        # here we want a statement -> names mapping, obtained through target
        groups = defaultdict(list)
        target_to_statement = {}
        for statement in statements:
            target_to_statement[statement.target] = statement
            groups[statement.target].append(statement.name)
        groups = {target_to_statement[target]: names for target, names in groups.items()}

        import_groups, imports, definitions = defaultdict(list), [], []
        for statement, names in sorted(groups.items(), key=lambda x: min(order[n] for n in x[1])):
            pair = sorted(names), statement
            if isinstance(statement, GlobalImportFrom):
                import_groups[statement.root].append(pair)
            elif isinstance(statement, GlobalImport):
                imports.append(pair)
            else:
                definitions.append(pair)

        for names, statement in imports:
            yield statement.to_str(names)

        if imports:
            yield ''

        for group in import_groups.values():
            yield GlobalImportFrom.group_to_str(group)

        if import_groups or imports:
            yield '\n'

        for names, statement in definitions:
            yield statement.to_str(names)

    def __setitem__(self, key, value):
        raise NotImplementedError


class ScopeWrapper(Dict[str, Any]):
    def __init__(self, scope: Scope):
        super().__init__()
        self.scope = scope

    def __contains__(self, name):
        return name in self.scope or super().__contains__(name)

    def keys(self):
        return list(set(super().keys()) | set(self.scope.keys()))

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError
        # yield from self.scope.items()
        # for key in set(super().keys()) - set(self.scope):
        #     yield key, super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.keys())

    def popitem(self):
        raise NotImplementedError

    def setdefault(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError

    def __delitem__(self, v):
        raise NotImplementedError


class ScopeEval(ScopeWrapper):
    def __getitem__(self, name):
        try:
            return self.scope[name]
        except KeyError as e:
            # this is needed because KeyError is converted to NameError by `eval`
            raise ExceptionWrapper(e) from e
        except EntryError:
            pass

        if name not in self:
            raise NameError(f'The name "{name}" is not defined.')
        return super().__getitem__(name)

    def __setitem__(self, k, v):
        raise NotImplementedError


class ScopeExec(ScopeWrapper):
    def __init__(self, scope: Scope, name: str = None):
        super().__init__(scope)
        self.name = name

    def __setitem__(self, name, value):
        assert self.name is not None and name == self.name
        super().__setitem__(name, value)

    def get_result(self):
        return super().pop(self.name)

    def __getitem__(self, name):
        if name in self.scope:
            return self.scope[name]
        return super().__getitem__(name)
