import os
from typing import List

from .structures import Structure, TokenWrapper


class Definition(Structure):
    def __init__(self, name: TokenWrapper, value: Structure):
        super().__init__(name)
        self.name = name
        self.value = value

    def to_str(self, level):
        return '{} = {}'.format(self.name.body, self.value.to_str(level))


class ImportPython(Structure):
    def __init__(self, root: List[TokenWrapper], values: list, relative: bool, main_token):
        super().__init__(main_token)
        self.from_ = root
        self.root = '.'.join(x.body for x in root)
        self._values = values
        self.values = [('.'.join(x.body for x in value), name) for value, name in values]
        self.relative = relative

    def to_str(self, level):
        result = ''
        if self.root:
            result += 'from %s ' % self.root
        result += 'import '
        for value, name in self.values:
            result += value + ' '
            if name is not None:
                result += 'as ' + name.body
            result += ', '
        return result[:-2] + '\n'


class ImportStarred(Structure):
    def __init__(self, root: List[TokenWrapper], relative: bool):
        super().__init__(root[0])
        root = [x.body for x in root]
        if relative:
            self.shortcut = ''
        else:
            self.shortcut = root.pop(0)
        self.root = root
        self.relative = relative

    def get_paths(self):
        return [(self.shortcut, os.path.join(*self.root) + '.config')]

    def to_str(self, level):
        return 'from ' + self.shortcut + '.' + '.'.join(self.root) + ' import *'


class ImportPath(Structure):
    def __init__(self, root, paths, main_token):
        super().__init__(main_token)
        self.from_ = root
        self.paths = [x.body for x in paths]

        root = eval(root.body) if root else ''
        parts = root.split(':', 1)
        assert len(parts) <= 2
        if len(parts) == 2:
            self.shortcut = parts.pop(0)
        else:
            self.shortcut = ''
        self.root = parts[0]

    def get_paths(self):
        return [(self.shortcut, os.path.join(self.root, path)) for path in map(eval, self.paths)]

    def to_str(self, level):
        result = ''
        if self.from_:
            result = 'from ' + self.from_.body
        return result + 'import ' + ','.join(self.paths)


class LazyImport(Structure):
    def __init__(self, from_, what, as_, relative: bool, main_token: TokenWrapper):
        super().__init__(main_token)
        self.from_, self.what, self.as_ = from_, what, as_
        self.relative = relative

    def to_str(self, level):
        result = ''
        if self.from_:
            result += 'from '
            if self.relative:
                result += '.'
            result += self.from_ + ' '

        result += 'import %s' % self.what
        if self.as_:
            result += ' as %s' % self.as_.body
        return result + '\n'

    def error_message(self):
        result = 'importing '
        if self.from_:
            result += self.from_ + '.'
        return result + self.what
