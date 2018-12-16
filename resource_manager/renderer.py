import ast
import importlib
import sys

from .visitor import Visitor
from .wrappers import ExpressionWrapper, UnifiedImport
from . import scope


class Renderer(Visitor):
    def __init__(self, global_scope, local_scope):
        self.local_scope = local_scope
        self.global_scope = global_scope

    @staticmethod
    def render(node, global_scope, local_scope=None):
        return Renderer(global_scope, local_scope).visit(node)

    def visit_expression_wrapper(self, node: ExpressionWrapper):
        code = compile(ast.Expression(node.expression), node.source_path, 'eval')
        return eval(code, scope.ScopeWrapper(self.global_scope), self.local_scope)

    def visit_unified_import(self, node: UnifiedImport):
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
