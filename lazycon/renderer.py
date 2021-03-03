import ast
import sys

from .visitor import Visitor
from .statements import GlobalFunction, GlobalAssign, GlobalImport
from . import scope


# TODO: move to node
class Renderer(Visitor):
    def __init__(self, global_scope):
        self.global_scope = global_scope

    @staticmethod
    def render(node, global_scope):
        return Renderer(global_scope).visit(node)

    def visit_global_assign(self, node: GlobalAssign):
        code = compile(ast.Expression(node.expression), node.source_path, 'eval')
        return eval(code, scope.ScopeEval(self.global_scope))

    def visit_global_import(self, node: GlobalImport):
        name = node.name
        wrapper = scope.ScopeExec(self.global_scope, name)
        self._exec(node.to_str([name]), wrapper, node.source_path)
        return wrapper.get_result()

    visit_global_import_from = visit_global_import

    def visit_global_function(self, node: GlobalFunction):
        wrapper = scope.ScopeExec(self.global_scope, node.name)
        self._exec(self._module([node.node]), wrapper, node.source_path)
        return wrapper.get_result()

    @staticmethod
    def _exec(code, global_scope, source):
        exec(compile(code, source, 'exec'), global_scope)

    if sys.version_info[:2] >= (3, 8):
        @staticmethod
        def _module(body):
            # handling `type_ignores`
            return ast.Module(body, [])
    else:
        @staticmethod
        def _module(body):
            return ast.Module(body)
