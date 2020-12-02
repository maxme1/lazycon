import ast
import inspect
import os
import sys
from typing import Iterable, Sequence, Tuple

from .exceptions import ConfigImportError


class Wrapper(ast.AST):
    """Wraps ast nodes used to define names in configs."""

    def __init__(self, position):
        super().__init__()
        self.line, self.column, self.source_path = self.position = position

    def to_str(self, *args):
        raise NotImplementedError


class ExpressionStatement(Wrapper):
    """Wraps the right part of an `=` assignment."""

    def __init__(self, expression: ast.AST, body, position):
        super().__init__(position)
        self.body = body
        self.expression = expression

    def to_str(self, names, level: int = 0):
        return '    ' * level + ' = '.join(names) + ' = ' + self.body


class ExpressionWrapper(Wrapper):
    """
    Wraps expressions inside functions, such as decorators, return value, defaults.
    """

    def __init__(self, expression: ast.AST, position):
        super().__init__(position)
        self.expression = expression


class PatternAssignment(ExpressionWrapper):
    """Wraps an unpacked assignment: a, b = c."""

    def __init__(self, expression: ast.AST, pattern, position):
        super().__init__(expression, position)
        self.pattern = pattern


class AssertionWrapper(Wrapper):
    def __init__(self, assertion: ast.Assert, position):
        super().__init__(position)
        self.assertion = assertion


def dotted(x):
    return '.'.join(x)


class BaseImport(Wrapper):
    """Wrapper for all import statements."""

    def __init__(self, root: Iterable[str], dots: int, position):
        super().__init__(position)
        self.root = tuple(root)
        self.dots = dots

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

        raise ConfigImportError('Shortcut "%s" is not found while parsing "%s".' % (shortcut, self.source_path))

    def _to_str(self):
        result = ''
        if self.root:
            result = 'from ' + '.' * self.dots + '%s ' % dotted(self.root)
        return result + 'import '


class ImportStarred(BaseImport):
    def __init__(self, root: Iterable[str], dots: int, position):
        super().__init__(root, dots, position)

    def to_str(self, names=()):
        assert not names
        return self._to_str() + '*\n'


class UnifiedImport(BaseImport):
    def __init__(self, root: Iterable[str], dots: int, what: Iterable[str], as_: bool, position):
        super().__init__(root, dots, position)
        self.what = tuple(what)
        self.as_ = as_

    def potentially_config(self):
        return bool(self.root)

    def import_what(self, name):
        result = dotted(self.what)
        if len(self.what) > 1 or self.what[0] != name:
            result += ' as ' + name

        return result

    def to_str(self, names, level=0):
        assert len(names) == 1
        return self._to_str() + self.import_what(names[0])


class Function(Wrapper):
    def __init__(self, signature: inspect.Signature, docstring: str, bindings: Sequence[Tuple[str, Wrapper]],
                 expression: ExpressionWrapper, decorators: Sequence[ExpressionWrapper],
                 assertions: Sequence[AssertionWrapper], original_name: str, position):
        super().__init__(position)
        self.docstring = docstring
        self.assertions = assertions
        self.body = None
        self.decorators = decorators
        self.original_name = original_name
        self.bindings = bindings
        self.expression = expression
        self.signature = signature

    def _to_str(self, name):
        return self.body[0] + ' ' + name + self.body[1]

    def to_str(self, names):
        return '\n' + '\n\n\n'.join(self._to_str(name) for name in names).strip() + '\n\n'
