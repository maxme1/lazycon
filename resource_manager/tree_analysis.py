from collections import defaultdict

from .structures import *


class SyntaxTree:
    def __init__(self, resources: dict):
        self.resources = resources
        self._request_stack = []
        self.cycles = defaultdict(set)
        self.undefined = defaultdict(set)

        self._visited = {name: False for name in resources}
        for name, node in resources.items():
            self._analyze_tree(name, node.position()[-1])

    def _analyze_tree(self, name, source):
        # undefined variable:
        if name not in self.resources:
            self.undefined[source].add(name)
            return
        if self._visited[name]:
            return
        # cycle
        if name in self._request_stack:
            prefix = " -> ".join(self._request_stack)
            self.cycles[source].add('{} -> {}'.format(prefix, name))
            return

        self._request_stack.append(name)
        self._analyze_node(self.resources[name])
        self._request_stack.pop()

        self._visited[name] = True

    def _analyze_node(self, node):
        if type(node) is Resource:
            self._analyze_tree(node.name.body, node.position()[-1])
        if type(node) is GetAttribute:
            self._analyze_node(node.target)
        if type(node) is GetItem:
            for arg in node.args:
                self._analyze_node(arg)
        if type(node) is Call:
            for arg in node.args:
                self._analyze_node(arg)
            for param in node.params:
                self._analyze_node(param.value)
        if type(node) is Array:
            for x in node.values:
                self._analyze_node(x)
        if type(node) is Dictionary:
            for key, value in node.dictionary.items():
                self._analyze_node(value)
