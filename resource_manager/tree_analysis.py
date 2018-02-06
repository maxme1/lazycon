from collections import defaultdict

from .structures import *


class SyntaxTree:
    def __init__(self, resources: dict):
        self.resources = resources
        self._request_stack = []
        self.cycles = defaultdict(set)
        self.undefined = defaultdict(set)

        self._scopes = []
        self._global = {}
        self._structure_types = []
        for name, node in resources.items():
            self._analyze_tree(name)

    def _analyze_tree(self, name):
        self._request_stack.append(name)
        self._global[name] = False
        self._analyze_node(self.resources[name])
        self._global[name] = True
        self._request_stack.pop()

    def _analyze_node(self, node: Structure):
        node.render(self)

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
        if not self._global[name]:
            prefix = " -> ".join(self._request_stack)
            self.cycles[source].add('{} -> {}'.format(prefix, name))

    def _render_get_attribute(self, node: GetAttribute):
        self._analyze_node(node.target)

    def _render_get_item(self, node: GetItem):
        self._analyze_node(node.target)
        for arg in node.args:
            self._analyze_node(arg)

    def _render_call(self, node: Call):
        for arg in node.args:
            self._analyze_node(arg)
        for param in node.params:
            self._analyze_node(param.value)

    def _render_array(self, node: Array):
        for x in node.values:
            self._analyze_node(x)

    def _render_dictionary(self, node: Dictionary):
        for key, value in node.pairs:
            self._analyze_node(key)
            self._analyze_node(value)

    def _render_lambda(self, node: Lambda):
        # TODO: checks
        self._scopes.append({x.body for x in node.params})
        self._analyze_node(node.expression)
        self._scopes.pop()

    def _render_lazy_import(self, node: LazyImport):
        pass

    def _render_literal(self, node: Literal):
        pass

    def _render_module(self, node: Module):
        pass
