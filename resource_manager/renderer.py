import ast
import importlib
import sys
from types import CodeType

from .visitor import Visitor
from .wrappers import ExpressionWrapper, UnifiedImport
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


class Renderer(Visitor):
    def __init__(self, global_scope, local_scope):
        self.local_scope = local_scope
        self.global_scope = global_scope

    @staticmethod
    def render(node, global_scope, local_scope=None):
        return Renderer(global_scope, local_scope).visit(node)

    def visit_expression_wrapper(self, node: ExpressionWrapper):
        code = shift_code(compile(ast.Expression(node.node), node.source_path, 'eval'), node.line)
        return eval(code, scope.ScopeWrapper(self.global_scope), self.local_scope)

    def visit_unified_import(self, node: ExpressionWrapper):
        from_ = '.'.join(node.root)
        what = '.'.join(node.what)
        if not from_:
            result = importlib.import_module(what)
            packages = node.what
            if len(packages) > 1 and not node.as_:
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


render = Renderer.render
