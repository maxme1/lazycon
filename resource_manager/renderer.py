import functools
import importlib
import sys

from .structures import *
from .token import BINARY_OPERATORS, UNARY_OPERATORS, TokenType
from . import scopes


class Renderer:
    def __init__(self, scope):
        self.scope = scope
        self._definitions_stack = []

    @classmethod
    def render(cls, node: Structure, scope):
        return cls(scope)._render(node)

    def _render(self, node: Structure):
        self._definitions_stack.append(node)
        try:
            value = node.render(self)
        except BaseException as e:
            if not self._definitions_stack:
                raise

            stack = []
            last_position = None
            for definition in reversed(self._definitions_stack):
                position = definition.position()
                position = position[0], position[2]

                if position != last_position:
                    line = definition.line()
                    if line[-1] == '\n':
                        line = line[:-1]
                    stack.append('\n  at %d:%d in %s\n    ' % definition.position() + line)
                last_position = position
            message = ''.join(reversed(stack))

            definition = self._definitions_stack[-1]
            self._definitions_stack = []
            raise RuntimeError('An exception occurred while ' + definition.error_message() + message) from e
        self._definitions_stack.pop()
        return value

    def _render_lambda(self, node: Lambda):
        err = 'Function requires %s%d argument(s), but %d provided'

        def f(*args):
            if len(args) != len(node.params):
                if not node.vararg:
                    raise ValueError(err % (' ', len(node.params), len(args)))
                elif len(args) < len(node.params):
                    raise ValueError(err % ('at least ', len(node.params), len(args)))

            scope = scopes.LocalScope(self.scope)
            for x, y in zip(node.params, args):
                scope.set_resource(x.body, y)
            if node.vararg:
                scope.set_resource(node.vararg.body, args[len(node.params):])

            return Renderer.render(node.expression, scope)

        return f

    def _render_resource(self, node: Resource):
        return self.scope.get_resource(node.name.body)

    def _render_get_attribute(self, node: GetAttribute):
        data = self._render(node.target)
        return getattr(data, node.name.body)

    def _render_get_item(self, node: GetItem):
        target = self._render(node.target)
        args = tuple(self._render(arg) for arg in node.args)
        if not node.trailing_coma and len(args) == 1:
            args = args[0]
        return target[args]

    def _render_call(self, node: Call):
        target = self._render(node.target)
        args = []
        for vararg, arg in zip(node.varargs, node.args):
            temp = self._render(arg)
            if vararg:
                args.extend(temp)
            else:
                args.append(temp)
        kwargs = {param.name.body: self._render(param.value) for param in node.params}
        if node.lazy:
            return functools.partial(target, *args, **kwargs)
        return target(*args, **kwargs)

    def _render_literal(self, node: Literal):
        return eval(node.value.body)

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
        for value in node.values:
            if type(value) is Starred:
                result.extend(list(self._render(value.expression)))
            else:
                result.append(self._render(value))
        return result

    def _render_tuple(self, node: Tuple):
        return tuple(self._render_array(node))

    def _render_dictionary(self, node: Dictionary):
        return {self._render(key): self._render(value) for key, value in node.pairs}

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
