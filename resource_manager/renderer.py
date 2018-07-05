import functools
import importlib
import sys

from .token import BINARY_OPERATORS, UNARY_OPERATORS, TokenType
from . import scopes
from .structures import *
from .exceptions import RenderError, custom_raise, LambdaArgumentsError


class Renderer:
    def __init__(self, scope, node_levels):
        self.scope = scope
        self._definitions_stack = []
        self._node_levels = node_levels

    @staticmethod
    def render(node: Structure, scope, node_levels):
        return Renderer(scope, node_levels)._render(node)

    @staticmethod
    def make_renderer(scope, node_levels):
        return Renderer(scope, node_levels)._render

    def update_levels(self, levels: dict):
        self._node_levels = levels

    def _render(self, node: Structure):
        self._definitions_stack.append(node)
        try:
            value = node.render(self)
        except BaseException as e:
            if not self._definitions_stack:
                raise
            definitions, self._definitions_stack = self._definitions_stack, []
            custom_raise(RenderError(definitions), e)

        self._definitions_stack.pop()
        return value

    def _render_lambda(self, node):
        return self._render_func_def(node)

    def _render_func_def(self, node: FuncDef):
        upper_scope = self.scope

        def function_(*args, **kwargs):
            scope = scopes.LocalScope(upper_scope)
            if node.vararg:
                scope.set_resource(node.vararg.name.body, args[len(node.positional):])
            else:
                if len(node.arguments) < len(args):
                    custom_raise(LambdaArgumentsError('Function requires %d argument(s), but %d provided' %
                                                      (len(node.arguments), len(args))))

            for name, arg in zip(node.positional, args):
                scope.set_resource(name, arg)

            for name, arg in kwargs.items():
                if name not in node.keyword:
                    custom_raise(LambdaArgumentsError("Function doesn't take argument: " + name))
                scope.set_resource(name, arg)

            not_defined = []
            for argument in node.arguments:
                name = argument.name.body
                if name not in scope:
                    if argument.has_default_value:
                        scope.set_resource(name, argument.default_value(self._render))
                    else:
                        not_defined.append(name)

            if not_defined:
                custom_raise(LambdaArgumentsError('Undefined argument(s): ' + ', '.join(not_defined)))

            for binding in node.bindings:
                scope.set_node(binding.names[0].body, binding.value)

            return Renderer.render(node.expression, scope, self._node_levels)

        function_.__qualname__ = function_.__name__ = node.name
        return function_

    def _render_resource(self, node: Resource):
        levels = self._node_levels[node]
        old_scope = self.scope
        for _ in range(levels):
            self.scope = self.scope._upper
        try:
            value = self.scope.get_resource(node.name.body, self._render)
        finally:
            self.scope = old_scope
        return value

    def _render_get_attribute(self, node: GetAttribute):
        data = self._render(node.target)
        return getattr(data, node.name.body)

    def _render_slice(self, node: Slice):
        return slice(*(x if x is None else self._render(x) for x in node.args))

    def _render_get_item(self, node: GetItem):
        target = self._render(node.target)
        args = tuple(self._render(arg) for arg in node.args)
        if not node.trailing_coma and len(args) == 1:
            args = args[0]
        return target[args]

    def _render_call(self, node: Call):
        target = self._render(node.target)
        args = []
        for arg in node.args:
            temp = self._render(arg.value)
            if arg.vararg:
                args.extend(temp)
            else:
                args.append(temp)
        kwargs = {arg.name.body: self._render(arg.value) for arg in node.kwargs}
        if node.lazy:
            return functools.partial(target, *args, **kwargs)
        return target(*args, **kwargs)

    def _render_literal(self, node: Literal):
        return eval(node.value.body, {}, {})

    def _render_binary(self, node: Binary):
        left = self._render(node.left)
        # shortcut logic
        if node.key == TokenType.AND:
            return left and self._render(node.right)
        if node.key == TokenType.OR:
            return left or self._render(node.right)

        return BINARY_OPERATORS[node.key](left, self._render(node.right))

    def _render_unary(self, node: Unary):
        return UNARY_OPERATORS[node.key](self._render(node.argument))

    def _render_inline_if(self, node: InlineIf):
        if self._render(node.condition):
            return self._render(node.left)
        return self._render(node.right)

    def _render_array(self, node: Array):
        result = []
        for value in node.entries:
            if type(value) is Starred:
                result.extend(list(self._render(value.expression)))
            else:
                result.append(self._render(value))
        return result

    def _render_tuple(self, node):
        return tuple(self._render_array(node))

    def _render_set(self, node):
        return set(self._render_array(node))

    def _render_dictionary(self, node: Dictionary):
        return {self._render(key): self._render(value) for key, value in node.entries}

    def _render_parenthesis(self, node: Parenthesis):
        return self._render(node.expression)

    def _render_lazy_import(self, node: LazyImport):
        if not node.from_:
            result = importlib.import_module(node.what)
            packages = node.what.split('.')
            if len(packages) > 1 and not node.as_:
                # import a.b.c
                return sys.modules[packages[0]]
            return result
        try:
            return getattr(importlib.import_module(node.from_), node.what)
        except AttributeError:
            pass
        try:
            return importlib.import_module(node.what, node.from_)
        except ModuleNotFoundError:
            return importlib.import_module(node.from_ + '.' + node.what)
