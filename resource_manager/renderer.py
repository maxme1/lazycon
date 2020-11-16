import ast
import importlib
import sys
from inspect import Parameter

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
        return eval(code, scope.ScopeWrapper(self.global_scope))

    def visit_pattern_assignment(self, node: PatternAssignment):
        value = self.visit_expression_wrapper(node)
        if not isinstance(node.pattern, str):
            value = tuple(value)
        return value

    visit_expression_statement = visit_expression_wrapper

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

    def visit_assertion_wrapper(self, node: AssertionWrapper):
        if sys.version_info[:2] >= (3, 8):
            # handling `type_ignores`
            args = [node.assertion], []
        else:
            args = [node.assertion],

        code = compile(ast.Module(*args), node.source_path, 'exec')
        return exec(code, scope.ScopeWrapper(self.global_scope))

    def visit_function(self, node: Function):
        def function_(*args, **kwargs):
            arguments = signature.bind_partial(*args, **kwargs)
            arguments.apply_defaults()

            not_defined = set(signature.parameters.keys()) - set(arguments.arguments)
            if not_defined:
                raise TypeError('Undefined argument(s): ' + ', '.join(not_defined))

            local_scope = scope.Scope(self.global_scope)
            local_scope.update_statements(node.bindings)
            for name, value in arguments.arguments.items():
                local_scope.add_value(name, value)

            for assertion in node.assertions:
                Renderer.render(assertion, local_scope)
            return Renderer.render(node.expression, local_scope)

        parameters = []
        for parameter in node.signature.parameters.values():
            if parameter.default is not Parameter.empty:
                parameter = parameter.replace(default=self.visit(parameter.default))

            parameters.append(parameter)

        signature = node.signature.replace(parameters=parameters)
        function_.__signature__ = signature
        function_.__doc__ = node.docstring
        function_.__name__ = function_.__qualname__ = node.original_name
        for decorator in reversed(node.decorators):
            function_ = self.visit(decorator)(function_)

        return function_
