import ast
import os
import sys
from typing import Sequence

from .exceptions import ConfigImportError


class Wrapper:
    def __init__(self, position):
        self.line, self.column, self.source_path = self.position = position


class GlobalStatement(Wrapper):
    """
    Wraps ast nodes used to define names in configs and maps a single name to a ast node.
    """

    def __init__(self, name: str, target: ast.AST, position):
        super().__init__(position)
        self.name = name
        self.target = target

    def to_str(self, names: Sequence[str]):
        raise NotImplementedError


class GlobalAssign(GlobalStatement):
    """ A single `name = value` statement. """

    def __init__(self, name: str, expression: ast.AST, body: str, position):
        super().__init__(name, expression, position)
        self.expression = expression
        self.body = body

    def to_str(self, names):
        return ' = '.join(names) + ' = ' + self.body


class ImportBase(GlobalStatement):
    pass


class GlobalImport(ImportBase):
    def __init__(self, alias: ast.alias, position):
        # TODO: smarter name inference
        super().__init__(alias.asname or alias.name, alias, position)
        self.alias = alias

    def to_str(self, names):
        # FIXME
        assert len(names) == 1
        name, = names
        assert name == self.name

        result = 'import ' + self.alias.name
        what = self.alias.name.split('.')
        if len(what) > 1 or what[0] != name:
            result += ' as ' + name

        return result


class GlobalImportFrom(ImportBase):
    def __init__(self, alias: ast.alias, root: str, position):
        # TODO: smarter name inference
        super().__init__(alias.asname or alias.name, alias, position)
        self.root = root
        self.alias = alias

    def to_str(self, names):
        # FIXME
        assert len(names) == 1, names
        name, = names
        assert name == self.name

        result = 'from ' + self.root + ' import ' + self.alias.name
        what = self.alias.name.split('.')
        if len(what) > 1 or what[0] != name:
            result += ' as ' + name

        return result

    def _import_what(self, name):
        result = self.alias.name
        what = result.split('.')
        if len(what) > 1 or what[0] != name:
            result += ' as ' + name

        return result

    @staticmethod
    def group_to_str(statements: Sequence['GlobalImportFrom']):
        base, *statements = statements
        result = base[1].to_str(base[0])
        for statement in statements:
            result += ', ' + statement[1]._import_what(statement[0])

        return result


class GlobalFunction(GlobalStatement):
    def __init__(self, node: ast.FunctionDef, body: str, position):
        super().__init__(node.name, node, position)
        self.node = node
        self.body = body

    def to_str(self, names):
        assert len(names) == 1 and names[0] == self.name
        return '\n' + self.body.strip() + '\n\n'


class ImportConfig(Wrapper):
    def __init__(self, root: Sequence[str], dots: int, position):
        super().__init__(position)
        self.dots = dots
        self.root = tuple(root)
        self.position = position

    def get_path(self, shortcuts):
        # relative import
        if self.dots > 0:
            root = (os.pardir,) * (self.dots - 1) + self.root
            prefix = os.path.dirname(self.source_path)
            return os.path.join(prefix, *root) + '.config'

        # import by shortcut
        shortcut, *root = self.root
        if shortcut in shortcuts:
            return os.path.join(shortcuts[shortcut], *root) + '.config'

        # import by sys.path
        visited = set()
        for prefix in sys.path:
            # optimizing disk access
            if prefix in visited:
                continue
            visited.add(prefix)

            path = os.path.join(prefix, *self.root) + '.config'
            if os.path.exists(path):
                return path

        raise ConfigImportError(f'Parent config not found while parsing "{self.source_path}".')


def dotted(x):
    return '.'.join(x)