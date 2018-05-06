from collections import defaultdict
from typing import Iterable

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
            raise RuntimeError(message)

    def _analyze_tree(self, name):
        self._request_stack.append(name)
        self.resources[name].render(self)
        self._global[name] = True
        self._request_stack.pop()

    def _render_resource(self, node: Resource):
        name = node.name.body
        # is it an argument?
        for scope in reversed(self._scopes):
            if name in scope:
                return
        # undefined variable:
        if name not in self._global:
            if name not in self._builtins:
                self.add_message('Undefined resources found', node, name)
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
        for arg in node.args:
            arg.render(self)

    def _render_call(self, node: Call):
        names = set(param.name.body for param in node.params)
        if len(names) < len(node.params):
            self.add_message('Duplicate keyword arguments', node, 'at %d:%d' % node.position()[:2])

        node.target.render(self)
        for arg in node.args:
            arg.render(self)
        for param in node.params:
            param.value.render(self)

    def _render_starred(self, node: Starred):
        node.expression.render(self)

    def _render_array(self, node: Array):
        for x in node.values:
            x.render(self)

    def _render_tuple(self, node: Array):
        for x in node.values:
            x.render(self)

    def _render_dictionary(self, node: Dictionary):
        for key, value in node.pairs:
            key.render(self)
            value.render(self)

    def _render_parenthesis(self, node: Parenthesis):
        node.expression.render(self)

    def _render_lambda(self, node: Lambda):
        params = node.params
        if node.vararg:
            params = params + (node.vararg,)

        names = {x.body for x in params}
        if len(names) != len(params):
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
        node.condition.render(self)
        node.left.render(self)
        node.right.render(self)
