import ast
import inspect
import os
from typing import Iterable, Sequence, Tuple


class Wrapper(ast.AST):
    def __init__(self, position):
        super().__init__()
        self.line, self.column, self.source_path = self.position = position

    def to_str(self, *args):
        raise NotImplementedError


class ExpressionStatement(Wrapper):
    def __init__(self, expression: ast.AST, body, position):
        super().__init__(position)
        self.body = body
        self.expression = expression

    def to_str(self, names, level: int = 0):
        # TODO: keep information about newline
        return '    ' * level + ' = '.join(names) + ' = ' + self.body + '\n'


class ExpressionWrapper(Wrapper):
    def __init__(self, expression: ast.AST, position):
        super().__init__(position)
        self.expression = expression


def dotted(x):
    return '.'.join(x)


class BaseImport(Wrapper):
    def __init__(self, root: Iterable[str], dots: int, position):
        super().__init__(position)
        self.root = tuple(root)
        self.dots = dots

    def get_path(self, shortcuts):
        if self.dots == 0:
            shortcut, *root = self.root
            if shortcut not in shortcuts:
                raise ImportError('Shortcut "%s" is not found while parsing "%s".' % (shortcut, self.source_path))

            prefix = shortcuts[shortcut]
        else:
            root = (os.pardir,) * (self.dots - 1) + self.root
            prefix = os.path.dirname(self.source_path)

        return os.path.join(prefix, *root) + '.config'

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

    def is_config_import(self, shortcuts):
        return self.root and (self.dots > 0 or self.root in shortcuts)

    def import_what(self, name):
        result = dotted(self.what)
        if len(self.what) > 1 or self.what[0] != name:
            result += ' as ' + name

        return result

    def to_str(self, names, level=0):
        assert len(names) == 1
        return self._to_str() + self.import_what(names[0])


class Function(Wrapper):
    def __init__(self, signature: inspect.Signature, bindings: Sequence[Tuple[str, Wrapper]],
                 expression: ExpressionWrapper, decorators: Sequence[ExpressionWrapper],
                 original_name: str, body: Tuple[str, str], position):
        super().__init__(position)
        self.body = body
        self.decorators = decorators
        self.original_name = original_name
        self.bindings = bindings
        self.expression = expression
        self.signature = signature

    def _to_str(self, name):
        return self.body[0] + ' ' + name + self.body[1]

    def to_str(self, names):
        return '\n' + '\n'.join(self._to_str(name) for name in names).strip() + '\n\n'
