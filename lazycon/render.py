import ast
import sys
from typing import Dict, Any

from .exceptions import ExceptionWrapper, EntryError


class ScopeWrapper(Dict[str, Any]):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def __contains__(self, name):
        return name in self.scope or super().__contains__(name)

    def keys(self):
        return list(set(super().keys()) | set(self.scope.keys()))

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError
        # yield from self.scope.items()
        # for key in set(super().keys()) - set(self.scope):
        #     yield key, super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.keys())

    def popitem(self):
        raise NotImplementedError

    def setdefault(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError

    def __delitem__(self, v):
        raise NotImplementedError


class ScopeEval(ScopeWrapper):
    def __getitem__(self, name):
        try:
            return self.scope[name]
        except KeyError as e:
            # this is needed because KeyError is converted to NameError by `eval`
            raise ExceptionWrapper(e) from e
        except EntryError:
            pass

        if name not in self:
            raise NameError(f'The name "{name}" is not defined.')
        return super().__getitem__(name)

    def __setitem__(self, k, v):
        raise NotImplementedError


class ScopeExec(ScopeWrapper):
    def __init__(self, scope, name: str = None):
        super().__init__(scope)
        self.name = name

    def __setitem__(self, name, value):
        assert self.name is not None and name == self.name
        super().__setitem__(name, value)

    def get_result(self):
        return super().pop(self.name)

    def __getitem__(self, name):
        if name in self.scope:
            return self.scope[name]
        return super().__getitem__(name)


def execute(statement, global_scope, source):
    exec(compile(make_module([statement]), source, 'exec'), global_scope)


if sys.version_info[:2] >= (3, 8):
    def make_module(body):
        # handling `type_ignores`
        return ast.Module(body, [])
else:
    def make_module(body):
        return ast.Module(body)
