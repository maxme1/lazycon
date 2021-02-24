import ast
import sys

from .visitor import Visitor
from .wrappers import ExpressionWrapper, UnifiedImport, Function, AssertionWrapper, PatternAssignment
from . import scope


class Renderer(Visitor):
    def __init__(self, global_scope):
        self.global_scope = global_scope

    @staticmethod
    def render(node, global_scope):
        return Renderer(global_scope).visit(node)

    def visit_expression_wrapper(self, node: ExpressionWrapper):
        code = compile(ast.Expression(node.expression), node.source_path, 'eval')
        return eval(code, scope.ScopeEval(self.global_scope))

    def visit_pattern_assignment(self, node: PatternAssignment):
        value = self.visit_expression_wrapper(node)
        if not isinstance(node.pattern, str):
            value = tuple(value)
        return value

    visit_expression_statement = visit_expression_wrapper

    def visit_unified_import(self, node: UnifiedImport):
        name = 'alias'
        wrapper = scope.ScopeExec(self.global_scope, name)
        self._exec(node.to_str([name]), wrapper, node.source_path)
        return wrapper.get_result()

    def visit_assertion_wrapper(self, node: AssertionWrapper):
        self._exec(self._module([node.assertion]), scope.ScopeExec(self.global_scope), node.source_path)

    def visit_function(self, node: Function):
        wrapper = scope.ScopeExec(self.global_scope, node.original_name)
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
