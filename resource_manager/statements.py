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


def make_dotted(ids):
    return '.'.join(ids)


class BaseImport(Structure):
    def __init__(self, root: List[TokenWrapper], prefix_dots: int, main_token):
        super().__init__(main_token)
        self._from = tuple(x.body for x in root)
        self._prefix_dots = prefix_dots

    def get_path(self):
        if self._prefix_dots == 0:
            shortcut, *root = self._from
        else:
            shortcut, root = '', self._from
        if self._prefix_dots > 1:
            root = (os.pardir,) + root

        return shortcut, os.path.join(*root) + '.config'

    def to_str(self, level):
        result = ''
        if self._from:
            result = 'from ' + '.' * self._prefix_dots + '%s ' % make_dotted(self._from)
        return result + 'import '


class UnifiedImport(BaseImport):
    def __init__(self, root: List[TokenWrapper], values: list, prefix_dots: int, main_token):
        super().__init__(root, prefix_dots, main_token)
        self._what = tuple((x.body for x in value) for value, name in values)
        self._as = tuple(name for value, name in values)

    def is_config_import(self, shortcuts):
        return self._from and (self._prefix_dots > 0 or self._from[0] in shortcuts)

    def iterate_values(self):
        for value, name in zip(self._what, self._as):
            yield make_dotted(value), name

    def get_root(self):
        return make_dotted(self._from)

    def to_str(self, level):
        result = super().to_str(level)
        for value, name in self.iterate_values():
            result += value
            if name is not None:
                result += ' as ' + name.body
            result += ', '

        return result[:-2] + '\n'


class ImportStarred(BaseImport):
    def to_str(self, level):
        return super().to_str(level) + '*\n'


class ImportPath(Structure):
    def __init__(self, path, main_token):
        super().__init__(main_token)
        self.path = path.body

        parts = eval(path.body).split(':', 1)
        if len(parts) == 2:
            self.shortcut = parts.pop(0)
        else:
            self.shortcut = ''
        self.root = parts[0]

    def get_path(self):
        return self.shortcut, self.root

    def to_str(self, level):
        return 'import %s \n' % self.path


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
