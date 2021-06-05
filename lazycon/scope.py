import builtins
from collections import defaultdict, OrderedDict
from threading import Lock
from typing import Dict, Any, Sequence, List

from .semantics.analyzer import NodeParents
from .thunk import ValueThunk, NodeThunk, Thunk
from .statements import GlobalStatement, GlobalImport, GlobalImportFrom, Definitions
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

    def __init__(self, definitions: Definitions, parent, parents: NodeParents):
        super().__init__()
        self.definitions = definitions
        self._parent = parent

        self._name_to_thunk: OrderedDict[str, Thunk] = OrderedDict()
        self._statement_to_names: Dict[GlobalStatement, List[str]] = {}
        self._parents = parents

        self._populated = False
        self._updated = False

        locks = {}
        for definition in definitions:
            statement = definition.statement
            if statement not in locks:
                locks[statement] = Lock()

            self._name_to_thunk[definition.name] = NodeThunk(statement, locks[statement])

        self._update_names()

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

                statement = thunk.statement
                names = self._statement_to_names[statement]

                for name, value in zip(names, statement.render(self, names)):
                    local = self._name_to_thunk[name]
                    assert isinstance(local, NodeThunk)
                    assert local.lock == thunk.lock
                    local.set(value)

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

        self._update_names()

    def _update_names(self):
        result = defaultdict(list)
        for name, thunk in self._name_to_thunk.items():
            if isinstance(thunk, NodeThunk):
                result[thunk.statement].append(name)
        self._statement_to_names = dict(result)

    def _get_leave_time(self, entry_points: Sequence[str]):
        def mark_name(name):
            nonlocal current
            if name not in leave_time:
                leave_time[name] = current
                current += 1

        def visit_parents(name):
            visited.add(name)
            for parent in self._parents[name]:
                find_leave_time(parent)

        def find_leave_time(name):
            if name in visited:
                return

            visit_parents(name)
            mark_name(name)

        names = {d.name: d for d in self.definitions}
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
            visit_parents(n)
            mark_name(n)

        return [names[n] for n in leave_time], {names[n]: t for n, t in leave_time.items()}

    def render(self, entry_points: Sequence[str] = None):
        if self._updated:
            raise RuntimeError('The scope has already been updated by live objects that cannot be rendered properly.')

        definitions, order = self._get_leave_time(entry_points)
        return render_scope(definitions, order)

    def __setitem__(self, key, value):
        raise NotImplementedError


def render_scope(definitions, order):
    # imports
    import_groups, imports, rest = defaultdict(list), [], []
    for definition in sorted(definitions, key=lambda s: order[s]):
        statement = definition.statement
        if isinstance(statement, GlobalImportFrom):
            import_groups[statement.root].append(statement)
        elif isinstance(statement, GlobalImport):
            imports.append(statement)
        else:
            rest.append(definition)

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
    for definition in rest:
        groups[definition.statement].append(definition)

    for _, group in sorted(groups.items(), key=lambda x: min(order[s] for s in x[1])):
        assert len(set(map(type, group))) == 1
        yield group[0].statement.group_to_str(group)
