import ast
import builtins
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Sequence

from .semantics.analyzer import NodeParents
from .thunk import ValueThunk, NodeThunk, Thunk
from .utils import reverse_mapping
from .statements import GlobalStatement, GlobalImport, GlobalImportFrom
from .exceptions import EntryError, SemanticError

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
                # TODO: maybe the thunk should hold the target?
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
                thunk.set(thunk.statement.render(self))

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

        return [names[n] for n in leave_time], {names[n]: t for n, t in leave_time.items()}

    def render(self, entry_points: Sequence[str] = None):
        if self._updated:
            raise RuntimeError('The scope has already been updated by live objects that cannot be rendered properly.')

        statements, order = self._get_leave_time(entry_points)

        # imports
        import_groups, imports, definitions = defaultdict(list), [], []
        for statement in sorted(statements, key=lambda s: order[s]):
            if isinstance(statement, GlobalImportFrom):
                import_groups[statement.root].append(statement)
            elif isinstance(statement, GlobalImport):
                imports.append(statement)
            else:
                definitions.append(statement)

        for statement in imports:
            yield statement.to_str()

        if imports:
            yield ''

        for group in import_groups.values():
            yield GlobalImportFrom.group_to_str(group)

        if import_groups or imports:
            yield '\n'

        # definitions
        groups = defaultdict(list)
        for statement in definitions:
            groups[statement.target].append(statement)

        for _, group in sorted(groups.items(), key=lambda x: min(order[s] for s in x[1])):
            assert len(set(map(type, group))) == 1
            yield group[0].group_to_str(group)

    def __setitem__(self, key, value):
        raise NotImplementedError
