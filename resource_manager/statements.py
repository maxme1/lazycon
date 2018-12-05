import os
from typing import List

from .structure import TokenWrapper


class Statement:
    def __init__(self, main_token: TokenWrapper):
        self.main_token = main_token

    def position(self):
        return self.line, self.main_token.column, self.source

    @property
    def source(self):
        return self.main_token.source or '<string input>'

    @property
    def line(self):
        return self.main_token.line

    def to_str(self, names: List[str], level: int):
        raise NotImplementedError


class ExpressionStatement(Statement):
    def __init__(self, expression, body: str, main_token):
        super().__init__(main_token)
        self.body = body
        self.expression = expression

    def to_str(self, names, level):
        return '    ' * level + ' = '.join(names) + ' = ' + self.body + '\n'


def dotted(x):
    return '.'.join(x)


class BaseImport(Statement):
    def __init__(self, root: List[TokenWrapper], dots: int, main_token):
        super().__init__(main_token)
        self.root = tuple(r.body for r in root)
        self.dots = dots

    def get_path(self, shortcuts):
        if self.dots == 0:
            shortcut, *root = self.root
            assert shortcut in shortcuts
            prefix = shortcuts[shortcut]
        else:
            root = (os.pardir,) * (self.dots - 1) + self.root
            prefix = os.path.dirname(self.main_token.source)

        return os.path.join(prefix, *root) + '.config'

    def _to_str(self):
        result = ''
        if self.root:
            result = 'from ' + '.' * self.dots + '%s ' % dotted(self.root)
        return result + 'import '


class ImportStarred(BaseImport):
    def __init__(self, root: List[TokenWrapper], dots: int, main_token: TokenWrapper):
        super().__init__(root, dots, main_token)

    def to_str(self, names=(), level=0):
        assert not names and level == 0
        return self._to_str() + '*\n'


class UnifiedImport(BaseImport):
    def __init__(self, root: List[TokenWrapper], what: List[TokenWrapper], as_: bool, dots: int,
                 main_token: TokenWrapper):
        super().__init__(root, dots, main_token)
        self.as_ = as_
        self.what = tuple(w.body for w in what)

    def is_config_import(self, shortcuts):
        return self.root and (self.dots > 0 or self.root in shortcuts)

    def to_str(self, names, level=0):
        assert level == 0 and len(names) == 1
        result = self._to_str() + dotted(self.what)

        if len(self.what) > 1 or self.what[0] != names[0]:
            result += ' as ' + names[0]

        return result + '\n'
