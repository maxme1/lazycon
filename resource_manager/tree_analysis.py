from collections import defaultdict

from resource_manager.helpers import Scope
from .structures import *


class SyntaxTree:
    def __init__(self, resources: dict):
        self.resources = resources
        self._request_stack = []
        self.cycles = defaultdict(set)
        self.undefined = defaultdict(set)
        self.duplicate = defaultdict(list)

        self._scopes = []
        self._global = {x: False for x in resources}
        self._structure_types = []
        for name, node in resources.items():
            self._analyze_tree(name)

    def _format(self, message, elements):
        message += ':\n'
        for source, item in elements.items():
            message += '  in %s\n    ' % source
            message += ', '.join(item)
            message += '\n'
        return message

    @staticmethod
    def analyze(scope: Scope):
        tree = SyntaxTree(scope._undefined_resources)
        message = ''
        if tree.cycles:
            message += tree._format('Cyclic dependencies found', tree.cycles)
        if tree.undefined:
            message += tree._format('Undefined resources found', tree.undefined)
        if tree.duplicate:
            message += tree._format('Duplicate arguments in lambda definition', tree.duplicate)
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

        source = node.source()
        # undefined variable:
        if name not in self._global:
            self.undefined[source].add(name)
            return
        # cycle
        if name in self._request_stack:
            prefix = " -> ".join(self._request_stack)
            self.cycles[source].add('{} -> {}'.format(prefix, name))
            return

        if not self._global[name]:
            self._analyze_tree(name)

    def _render_get_attribute(self, node: GetAttribute):
        node.target.render(self)

    def _render_get_item(self, node: GetItem):
        node.target.render(self)
        for arg in node.args:
            arg.render(self)

    def _render_call(self, node: Call):
        node.target.render(self)
        for arg in node.args:
            arg.render(self)
        for param in node.params:
            param.value.render(self)

    def _render_array(self, node: Array):
        for x in node.values:
            x.render(self)

    def _render_dictionary(self, node: Dictionary):
        for key, value in node.pairs:
            key.render(self)
            value.render(self)

    def _render_lambda(self, node: Lambda):
        names = {x.body for x in node.params}
        if len(names) != len(node.params):
            self.duplicate[node.source()].append('at %d:%d' % node.position()[:2])
        self._scopes.append(names)
        node.expression.render(self)
        self._scopes.pop()

    def _render_lazy_import(self, node: LazyImport):
        pass

    def _render_literal(self, node: Literal):
        pass

    def _render_module(self, node: Module):
        pass
