from collections import defaultdict
from typing import Iterable, Dict

from .exceptions import custom_raise, BuildConfigError
from .token import TokenType, INVALID_STRING_PREFIXES
from .scopes import GlobalScope
from .structures import *


class SyntaxTree:
    def __init__(self, name_to_node: dict, node_to_names: dict, builtins: Iterable):
        self.messages = defaultdict(lambda: defaultdict(set))
        self._scopes = []
        self._builtins = builtins
        self.node_to_names = node_to_names
        self.node_levels = {}

        self.enter_scope(name_to_node)
        self.visit_current_scope()

    def add_message(self, message, node, content):
        self.messages[message][node.source()].add(content)

    @staticmethod
    def format(message, elements):
        message += ':\n'
        for source, item in elements.items():
            message += '  in %s\n    ' % source
            message += ', '.join(item)
            message += '\n'
        return message

    @staticmethod
    def analyze(scope: GlobalScope):
        tree = SyntaxTree(scope._name_to_node, scope._node_to_names, scope.builtins)
        message = ''
        for msg, elements in tree.messages.items():
            message += tree.format(msg, elements)
        if message:
            custom_raise(BuildConfigError(message))
        return tree.node_levels

    def enter_scope(self, names: Dict[str, Structure], visited=()):
        scope = {name: [value, None] for name, value in names.items()}
        for name in visited:
            scope[name][1] = True
        self._scopes.append(scope)

    def leave_scope(self):
        self._scopes.pop()

    def not_visited(self, value):
        return value[1] is None

    def entered(self, value):
        return value[1] is False

    def set_node_level(self, node, level):
        assert node not in self.node_levels, node
        self.node_levels[node] = level

    def visit(self, value, level=0):
        assert value[1] is None
        n = len(self._scopes) - level
        self._scopes, tail = self._scopes[:n], self._scopes[n:]
        value[1] = False
        node = value[0]

        node.render(self)
        if node in self.node_to_names:
            assert len(self._scopes) == 1
            scope = self._scopes[0]
            for name in self.node_to_names[node]:
                scope[name][1] = True
            assert value[1]
        else:
            value[1] = True
        self._scopes.extend(tail)

    def visit_current_scope(self):
        for value in self._scopes[-1].values():
            if self.not_visited(value):
                self.visit(value)

    def _render_sequence(self, sequence):
        for item in sequence:
            item.render(self)

    def _render_resource(self, node: Resource):
        name = node.name.body
        for level, scope in enumerate(reversed(self._scopes)):
            if name in scope:
                if self.not_visited(scope[name]):
                    self.visit(scope[name], level)
                elif self.entered(scope[name]):
                    # TODO: more information
                    self.add_message('Resources are referenced before being completely defined',
                                     node, '"' + name + '" at %d:%d' % node.position()[:2])

                return self.set_node_level(node, level)

        if name in self._builtins:
            return self.set_node_level(node, len(self._scopes) - 1)

        # undefined resource:
        self.add_message('Undefined resources found, but are required', node, name)

    def _render_get_attribute(self, node: GetAttribute):
        node.target.render(self)

    def _render_slice(self, node: Slice):
        for arg in node.args:
            if arg is not None:
                arg.render(self)

    def _render_get_item(self, node: GetItem):
        node.target.render(self)
        self._render_sequence(node.args)

    def _render_call(self, node: Call):
        names = set(arg.name.body for arg in node.kwargs)
        if len(names) < len(node.kwargs):
            self.add_message('Duplicate keyword arguments', node, 'at %d:%d' % node.position()[:2])

        node.target.render(self)
        for param in node.args + node.kwargs:
            param.value.render(self)

    def _render_starred(self, node: Starred):
        node.expression.render(self)

    def _render_array(self, node: Array):
        self._render_sequence(node.entries)

    def _render_tuple(self, node):
        self._render_array(node)

    def _render_set(self, node):
        self._render_array(node)

    def _render_dictionary(self, node: Dictionary):
        for key, value in node.entries:
            key.render(self)
            value.render(self)

    def _render_parenthesis(self, node: Parenthesis):
        node.expression.render(self)

    def _render_lambda(self, node):
        self._render_func_def(node)

    def _render_func_def(self, node: FuncDef):
        names = {x.name.body: None for x in node.arguments}
        if len(names) != len(node.arguments):
            self.add_message('Duplicate arguments in lambda definition', node, 'at %d:%d' % node.position()[:2])

        assert all(len(x.names) == 1 for x in node.bindings)
        bindings = {x.names[0].body: x.value for x in node.bindings}
        if len(bindings) != len(node.bindings):
            self.add_message('Duplicate binding names in function definition', node, 'at %d:%d' % node.position()[:2])

        if set(names) & set(bindings):
            self.add_message('Binding names clash with argument names in function definition',
                             node, 'at %d:%d' % node.position()[:2])

        for argument in node.arguments:
            if argument.has_default_value:
                argument.default_expression.render(self)

        bindings.update(names)
        self.enter_scope(bindings, names)
        node.expression.render(self)
        self.visit_current_scope()
        self.leave_scope()

    def _render_lazy_import(self, node: LazyImport):
        pass

    def _render_literal(self, node: Literal):
        if node.value.type == TokenType.STRING and node.value.body.startswith(INVALID_STRING_PREFIXES):
            self.add_message('Inline string formatting is not supported', node, 'at %d:%d' % node.position()[:2])

    def _render_binary(self, node: Binary):
        node.left.render(self)
        node.right.render(self)

    def _render_unary(self, node: Unary):
        node.argument.render(self)

    def _render_inline_if(self, node: InlineIf):
        self._render_sequence([node.condition, node.left, node.right])
