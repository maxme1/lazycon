import os
from typing import List

from .structure import Structure, TokenWrapper


class Statement:
    def __init__(self, main_token: TokenWrapper):
        self.main_token = main_token

    def position(self):
        return self.main_token.line, self.main_token.column, self.source()

    def source(self):
        return self.main_token.source or '<string input>'


class ExpressionStatement(Statement):
    def __init__(self, expression, body: str, main_token):
        super().__init__(main_token)
        self.body = body
        self.expression = expression

        # super().__init__(names[0])
        # self.names = names
        # self.value = value

    def to_str(self, level):
        return '    ' * level + ' = '.join(name.body for name in self.names) + ' = %s\n' % self.value.to_str(level)


# import a
# from a import b
# from a import *


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
            prefix = self.main_token.source

        return os.path.join(os.path.dirname(prefix), *root) + '.config'

    def to_str(self, level):
        result = ''
        if self._from:
            result = 'from ' + '.' * self._prefix_dots + '%s ' % make_dotted(self._from)
        return result + 'import '


class ImportStarred(BaseImport):
    def __init__(self, root: List[TokenWrapper], dots: int, main_token: TokenWrapper):
        super().__init__(root, dots, main_token)


class UnifiedImport(BaseImport):
    def __init__(self, root: List[TokenWrapper], what: List[TokenWrapper], dots: int, main_token: TokenWrapper):
        super().__init__(root, dots, main_token)
        self.what = tuple(w.body for w in what)

    def is_config_import(self, shortcuts):
        return self.root and (self.dots > 0 or self.root in shortcuts)

    def to_str(self, level):
        result = super().to_str(level)
        for value, name in self.iterate_values():
            result += value
            if name is not None:
                result += ' as ' + name.body
            result += ', '

        return result[:-2] + '\n'


class LazyImport(Statement):
    def __init__(self, from_, what, as_, main_token: TokenWrapper):
        super().__init__(main_token)
        self.from_, self.what, self.as_ = from_, what, as_

    def from_to_str(self):
        if self.from_:
            return 'from ' + self.from_ + ' '
        return ''

    def what_to_str(self):
        result = '%s' % self.what
        if self.as_:
            result += ' as %s' % self.as_.body
        return result

    def to_str(self, level):
        return self.from_to_str() + 'import ' + self.what_to_str() + '\n'

    def error_message(self):
        result = 'importing '
        if self.from_:
            result += self.from_ + '.'
        return result + self.what
