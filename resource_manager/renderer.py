import importlib
import sys
from types import CodeType

from .statements import Statement, ExpressionStatement, UnifiedImport
from . import scope

NAMES = ('argcount kwonlyargcount nlocals stacksize flags codestring '
         'consts names varnames filename name firstlineno lnotab freevars cellvars').split()


def shift_code(code: CodeType, shift: int):
    args = {arg: getattr(code, 'co_' + arg, None) for arg in NAMES}
    args['codestring'] = code.co_code
    args['firstlineno'] += shift - 1
    args['consts'] = tuple(shift_code(const, shift) if isinstance(const, CodeType) else const
                           for const in code.co_consts)
    return CodeType(*(args[name] for name in NAMES))


def render(statement: Statement, global_scope, local_scope=None):
    if isinstance(statement, ExpressionStatement):
        code = shift_code(compile(statement.body, statement.source, 'eval'), statement.line)
        return eval(code, scope.ScopeWrapper(global_scope), local_scope)

    if isinstance(statement, UnifiedImport):
        from_ = '.'.join(statement.root)
        what = '.'.join(statement.what)
        if not from_:
            result = importlib.import_module(what)
            packages = statement.what
            if len(packages) > 1 and not statement.as_:
                # import a.b.c
                return sys.modules[packages[0]]
            return result
        try:
            return getattr(importlib.import_module(from_), what)
        except AttributeError:
            pass
        try:
            return importlib.import_module(what, from_)
        except ModuleNotFoundError:
            pass
        return importlib.import_module(from_ + '.' + what)

    raise NotImplementedError
