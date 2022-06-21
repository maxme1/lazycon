import ast
import os
import sys
from typing import Sequence, NamedTuple, Any, Tuple

from .exceptions import ConfigImportError
from .render import ScopeExec, execute

IGNORE_NAME = '_'


class Wrapper:
    def __init__(self, position):
        self.line, self.column, self.source_path = self.position = position


class GlobalStatement(Wrapper):
    """
    Wraps ast nodes used to define names in configs and maps a single name to a ast node.
    """

    def __init__(self, node: ast.AST, position):
        super().__init__(position)
        self.node = node

    def render(self, global_scope, names: Sequence[str]) -> Sequence[Any]:
        wrapper = ScopeExec(global_scope)
        execute(self.node, wrapper, self.source_path)
        return wrapper.get_results(names)


class Definition(NamedTuple):
    name: str
    statement: GlobalStatement


Definitions = Sequence[Definition]


class GlobalAssign(GlobalStatement):
    """ A single `target = value` statement. """

    def __init__(self, node: ast.Assign, body: str, position):
        super().__init__(node, position)
        self.body = body

    @staticmethod
    def _group_to_str(pattern, names, level) -> Tuple[str, bool]:
        if isinstance(pattern, ast.Name):
            name = pattern.id if pattern.id in names else IGNORE_NAME
            # the name could already have been IGNORE_NAME
            return name, name != IGNORE_NAME

        if isinstance(pattern, ast.Starred):
            child, contains = GlobalAssign._group_to_str(pattern.value, names, level + 1)
            return '*' + child, contains

        assert isinstance(pattern, (ast.Tuple, ast.List))
        joined, contains = [], False
        for elt in pattern.elts:
            child, local = GlobalAssign._group_to_str(elt, names, level + 1)
            contains = contains or local
            joined.append(child)
        joined = ', '.join(joined)

        if isinstance(pattern, ast.Tuple):
            if len(pattern.elts) == 1:
                joined += ','
            if level > 0:
                joined = f'({joined})'

        else:
            joined = f'[{joined}]'

        return joined, contains

    @staticmethod
    def group_to_str(definitions: Definitions):
        assert len({d.statement for d in definitions}) == 1
        statement: GlobalAssign = definitions[0].statement
        names = {d.name for d in definitions}

        result = ''
        for target in statement.node.targets:
            target, contains = GlobalAssign._group_to_str(target, names, 0)
            if contains:
                result += target + ' = '

        return result + statement.body


class GlobalFunction(GlobalStatement):
    def __init__(self, node: ast.FunctionDef, body: str, position):
        super().__init__(node, position)
        self.body = body

    def to_str(self):
        return '\n' + self.body.strip() + '\n\n'

    @staticmethod
    def group_to_str(definitions: Definitions):
        assert len(definitions) == 1
        return definitions[0].statement.to_str()

    def render(self, global_scope, names: Sequence[str]) -> Sequence[Any]:
        assert len(names) == 1 and self.node.name == names[0]
        return super().render(global_scope, names)


class GlobalImport(GlobalStatement):
    def __init__(self, node: ast.Import, position):
        super().__init__(node, position)
        alias, = node.names
        self.alias = alias
        self.name = alias.asname or alias.name.split('.', 1)[0]

    def _import_what(self):
        alias = self.alias
        result = alias.name
        if alias.asname is not None:
            result += ' as ' + alias.asname

        return result

    def to_str(self):
        return f'import {self._import_what()}'

    def render(self, global_scope, names: Sequence[str]) -> Sequence[Any]:
        assert len(names) == 1 and self.name == names[0]
        return super().render(global_scope, names)


class GlobalImportFrom(GlobalImport):
    def __init__(self, node: ast.ImportFrom, position):
        super().__init__(node, position)
        self.root = node.module

    def to_str(self):
        return f'from {self.root} {super().to_str()}'

    @staticmethod
    def group_to_str(statements: Sequence['GlobalImportFrom']):
        base, *statements = statements
        result = base.to_str()
        for statement in statements:
            result += ', ' + statement._import_what()

        return result


class ImportConfig(Wrapper):
    def __init__(self, root: Sequence[str], dots: int, extension: str, position):
        super().__init__(position)
        self.dots = dots
        self.root = tuple(root)
        self.position = position
        self.extension = extension

    def get_path(self, shortcuts):
        # relative import
        if self.dots > 0:
            root = (os.pardir,) * (self.dots - 1) + self.root
            prefix = os.path.dirname(self.source_path)
            return os.path.join(prefix, *root) + self.extension

        # import by shortcut
        shortcut, *root = self.root
        if shortcut in shortcuts:
            return os.path.join(shortcuts[shortcut], *root) + self.extension

        # import by sys.path
        visited = set()
        for prefix in sys.path:
            # optimizing disk access
            if prefix in visited:
                continue
            visited.add(prefix)

            path = os.path.join(prefix, *self.root) + self.extension
            if os.path.exists(path):
                return path

        root = '.'.join(self.root)
        raise ConfigImportError(f'Parent config "{root}" not found while parsing "{self.source_path}".')


class LiveObject(Wrapper):
    def __init__(self, name: str, value: Any):
        super().__init__((1, 1, '<code>'))
        self._name = name
        self.value = value

    @staticmethod
    def group_to_str(definitions: Definitions):
        assert len(definitions) == 1, definitions
        name, value = definitions[0].statement._name, definitions[0].statement.value
        return f'{name} = {LiveObject._stringify(value)}'

    @staticmethod
    def _stringify(value):
        if isinstance(value, (bool, int, float, complex, str, bytes)) or value is None:
            return repr(value)
        if isinstance(value, list):
            return '[' + ', '.join(map(LiveObject._stringify, value)) + ']'
        if isinstance(value, tuple):
            return '(' + ', '.join(map(LiveObject._stringify, value)) + ')'
        if isinstance(value, set):
            return '{' + ', '.join(map(LiveObject._stringify, value)) + '}'
        if isinstance(value, dict):
            return '{' + ', '.join(
                f'{LiveObject._stringify(k)}: {LiveObject._stringify(v)}' for k, v in value.items()
            ) + '}'

        raise ValueError(f"Don't know how to stringify '{value}'")
