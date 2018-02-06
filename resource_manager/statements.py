import os
from typing import List

from .structures import Structure, Token


class Definition(Structure):
    def __init__(self, name: Token, value: Structure):
        super().__init__(name)
        self.name = name
        self.value = value

    def to_str(self, level):
        return '{} = {}'.format(self.name.body, self.value.to_str(level))


class ImportPython(Structure):
    def __init__(self, root: List[Token], values: list, relative: bool, main_token):
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
    def __init__(self, root: List[Token], relative: bool):
        super().__init__(root[0])

        self.root = root
        self.relative = relative

    def get_paths(self):
        parent = ''
        root = self.root
        if not self.relative:
            parent = root[0].body + ':'
            root = root[1:]
        return [parent + os.sep.join(x.body for x in root) + '.config']

    def to_str(self, level):
        result = 'from '
        if self.relative:
            result += '.'
        return result + '.'.join(x.body for x in self.root) + ' import *'


class ImportPath(Structure):
    def __init__(self, root, paths, main_token):
        super().__init__(main_token)
        self.root = root
        self.paths = paths

    def get_paths(self):
        paths = [eval(x.body) for x in self.paths]
        if not self.root:
            return paths
        return [os.path.join(self.root.body, x) for x in paths]

    def to_str(self, level):
        result = ''
        if self.root:
            result = 'from ' + self.root.body
        return result + 'import ' + ','.join(x.body for x in self.paths)


class LazyImport(Structure):
    def __init__(self, from_, what, as_, relative: bool, main_token: Token):
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
