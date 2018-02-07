from collections import defaultdict

from resource_manager.helpers import Scope
from .structures import *


class SyntaxTree:
    def __init__(self, resources: dict):
        self.resources = resources
        self._request_stack = []
        self.cycles = defaultdict(set)
        self.undefined = defaultdict(set)
        self._duplicate_arguments = defaultdict(list)

        self._scopes = []
        self._global = {x: False for x in resources}
        self._structure_types = []
        for name, node in resources.items():
            self._analyze_tree(name)

    @staticmethod
    def analyze(scope: Scope):
        tree = SyntaxTree(scope._undefined_resources)
        message = ''
        if tree.cycles:
            message += 'Cyclic dependencies found in the following resources:\n'
            for source, cycles in tree.cycles.items():
                message += '  in %s\n    ' % source
                message += '\n    '.join(cycles)
                message += '\n'
        if tree.undefined:
            if message:
                message += '\n'
            message += 'Undefined resources found:\n'
            for source, undefined in tree.undefined.items():
                message += '  in %s\n    ' % source
                message += ', '.join(undefined)
                message += '\n'
        if tree._duplicate_arguments:
            if message:
                message += '\n'
            message += 'Duplicate arguments in lambda definition:\n'
            for source, nodes in tree._duplicate_arguments.items():
                message += '  in %s\n    at ' % source
                message += ', '.join('%d:%d' % node.position()[:2] for node in nodes)
                message += '\n'
        if message:
            raise RuntimeError(message)

    def _analyze_tree(self, name):
        self._request_stack.append(name)
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
        if name in self._request_stack:
            prefix = " -> ".join(self._request_stack)
            self.cycles[source].add('{} -> {}'.format(prefix, name))

        if not self._global[name]:
            self._analyze_tree(name)

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
        names = {x.body for x in node.params}
        if len(names) != len(node.params):
            self._duplicate_arguments[node.source()].append(node)
        self._scopes.append(names)
        self._analyze_node(node.expression)
        self._scopes.pop()

    def _render_lazy_import(self, node: LazyImport):
        pass

    def _render_literal(self, node: Literal):
        pass

    def _render_module(self, node: Module):
        pass
