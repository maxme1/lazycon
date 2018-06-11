from collections import defaultdict
from typing import Iterable

from resource_manager.exceptions import custom_raise, BuildConfigError
from .token import TokenType, INVALID_STRING_PREFIXES
from .scopes import GlobalScope
from .structures import *


class SyntaxTree:
    def __init__(self, resources: dict, builtins: Iterable):
        self.resources = resources
        self._request_stack = []
        self.messages = defaultdict(lambda: defaultdict(set))

        self._scopes = []
        self._builtins = builtins
        self._global = {x: False for x in resources}
        self._structure_types = []
        for name, node in resources.items():
            self._analyze_tree(name)

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
        tree = SyntaxTree(scope._undefined_resources, scope.builtins)
        message = ''
        for msg, elements in tree.messages.items():
            message += tree.format(msg, elements)
        if message:
            custom_raise(BuildConfigError(message))

    def _analyze_tree(self, name):
        self._request_stack.append(name)
        self.resources[name].render(self)
        self._global[name] = True
        self._request_stack.pop()

    def _render_sequence(self, sequence):
        for item in sequence:
            item.render(self)

    def _render_resource(self, node: Resource):
        name = node.name.body
        # is it an argument?
        for scope in reversed(self._scopes):
            if name in scope:
                return
        # undefined variable:
        if name not in self._global:
            if name not in self._builtins:
                self.add_message('Undefined resources found, but are required', node, name)
            return
        # cycle
        if name in self._request_stack:
            prefix = " -> ".join(self._request_stack)
            self.add_message('Cyclic dependencies found', node, '{} -> {}'.format(prefix, name))
            return

        if not self._global[name]:
            self._analyze_tree(name)

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

    def _render_lambda(self, node: Lambda):
        names = {x.name.body for x in node.arguments}
        if len(names) != len(node.arguments):
            self.add_message('Duplicate arguments in lambda definition', node, 'at %d:%d' % node.position()[:2])
        self._scopes.append(names)
        node.expression.render(self)
        self._scopes.pop()

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
