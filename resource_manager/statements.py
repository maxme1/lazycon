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


def get_imported_name(what, as_):
    if as_ is not None:
        return as_.body
    name = what
    packages = name.split('.')
    if len(packages) > 1:
        name = packages[0]
    return name


class ImportPython(Structure):
    def __init__(self, root: List[TokenWrapper], values: list, main_token):
        super().__init__(main_token)
        self.from_ = root
        self.root = '.'.join(x.body for x in root)
        self._values = values
        self.values = [('.'.join(x.body for x in value), name) for value, name in values]

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
    def __init__(self, root: List[TokenWrapper], prefix_dots: int):
        super().__init__(root[0])
        assert 0 <= prefix_dots <= 2
        root = [x.body for x in root]
        if prefix_dots > 0:
            self.shortcut = ''
        else:
            self.shortcut = root.pop(0)
        self.root = root
        self.prefix_dots = prefix_dots

    def get_paths(self):
        root = self.root
        if self.prefix_dots > 1:
            root = [os.pardir] + root
        return [(self.shortcut, os.path.join(*root) + '.config')]

    def to_str(self, level):
        if self.prefix_dots > 1:
            prefix = '.'
        else:
            prefix = self.shortcut
        return 'from ' + prefix + '.' + '.'.join(self.root) + ' import *'


class ImportPartial(ImportStarred):
    def __init__(self, root: List[TokenWrapper], prefix_dots: int, values: list):
        super().__init__(root, prefix_dots)
        self.values = [('.'.join(x.body for x in value), name) for value, name in values]

    def to_str(self, level):
        if self.prefix_dots > 1:
            prefix = '.'
        else:
            prefix = self.shortcut
        what = ''
        for value, name in self.values:
            what += value + ' '
            if name is not None:
                what += 'as ' + name.body
            what += ', '
        return 'from ' + prefix + '.' + '.'.join(self.root) + ' import ' + what[:-2] + '\n'


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
    def __init__(self, from_, what, as_, main_token: TokenWrapper):
        super().__init__(main_token)
        self.from_, self.what, self.as_ = from_, what, as_

    def to_str(self, level):
        result = ''
        if self.from_:
            result += 'from ' + self.from_ + ' '

        result += 'import %s' % self.what
        if self.as_:
            result += ' as %s' % self.as_.body
        return result + '\n'

    def error_message(self):
        result = 'importing '
        if self.from_:
            result += self.from_ + '.'
        return result + self.what
