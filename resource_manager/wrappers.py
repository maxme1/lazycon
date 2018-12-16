import ast
import inspect
import os
from collections import namedtuple
from typing import List, Iterable, Tuple


class Wrapper(ast.AST):
    def __init__(self, position):
        super().__init__()
        self.position = position
        self.line, self.column, self.source_path = position

    def to_str(self, *args):
        raise NotImplementedError


ScopeItem = namedtuple('ScopeItem', 'name value')


class ExpressionWrapper(Wrapper):
    def __init__(self, node: ast.AST, body, position):
        super().__init__(position)
        self.body = body
        self.node = node

    def to_str(self, names, level: int = 0):
        return '    ' * level + ' = '.join(names) + ' = ' + self.body + '\n'


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
            assert shortcut in shortcuts
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

    def to_str(self, names, level=0):
        assert len(names) == 1
        result = self._to_str() + dotted(self.what)

        if len(self.what) > 1 or self.what[0] != names[0]:
            result += ' as ' + names[0]

        return result + '\n'


class Function(Wrapper):
    def __init__(self, signature: inspect.Signature, bindings: [(str, Wrapper)],
                 expression: ExpressionWrapper, position):
        super().__init__(position)
        self.bindings = bindings
        self.expression = expression
        self.signature = signature

    def _to_str(self, name, level):
        result = '\ndef ' + name + str(self.signature) + ':\n'
        for local_name, binding in self.bindings:
            result += binding.to_str([local_name], level + 1)
        return result + '    ' * (level + 1) + 'return ' + self.expression.body + '\n'

    def to_str(self, names, level=0):
        return '\n'.join(self._to_str(name, level) for name in names)
