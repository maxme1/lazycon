import ast
import os
import sys
from typing import Sequence

from .exceptions import ConfigImportError
from .render import ScopeEval, ScopeExec, execute


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

    def to_str(self):
        raise NotImplementedError

    def render(self, global_scope):
        raise NotImplementedError


class GlobalAssign(GlobalStatement):
    """ A single `name = value` statement. """

    def __init__(self, name: str, expression: ast.AST, body: str, position):
        super().__init__(name, expression, position)
        self.expression = expression
        self.body = body

    def to_str(self):
        return f'{self.name} = {self.body}'

    @staticmethod
    def group_to_str(statements: Sequence['GlobalAssign']):
        assert len({statement.target for statement in statements}) == 1

        *statements, base = statements
        result = ''
        for statement in statements:
            result += f'{statement.name} = '
        return result + base.to_str()

    def render(self, global_scope):
        code = compile(ast.Expression(self.expression), self.source_path, 'eval')
        return eval(code, ScopeEval(global_scope))


class GlobalFunction(GlobalStatement):
    def __init__(self, node: ast.FunctionDef, body: str, position):
        super().__init__(node.name, node, position)
        self.body = body
        self.node = node

    def to_str(self):
        return '\n' + self.body.strip() + '\n\n'

    @staticmethod
    def group_to_str(statements: Sequence['GlobalFunction']):
        assert len(statements) == 1
        return statements[0].to_str()

    def render(self, global_scope):
        wrapper = ScopeExec(global_scope, self.name)
        execute(self.target, wrapper, self.source_path)
        return wrapper.get_result()


class ImportBase(GlobalStatement):
    pass


class GlobalImport(ImportBase):
    def __init__(self, node: ast.Import, position):
        alias, = node.names
        super().__init__(alias.asname or alias.name.split('.', 1)[0], node, position)
        self.alias = alias

    def _import_what(self):
        alias = self.alias
        result = alias.name
        if alias.asname is not None:
            result += ' as ' + alias.asname

        return result

    def to_str(self):
        return f'import {self._import_what()}'

    def render(self, global_scope):
        wrapper = ScopeExec(global_scope, self.name)
        execute(self.target, wrapper, self.source_path)
        return wrapper.get_result()


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
