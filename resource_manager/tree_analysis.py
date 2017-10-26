from .structures import *


class SyntaxTree:
    def __init__(self, resources_dict):
        self.resources = resources_dict
        self._request_stack = []
        self.cycles = []
        self.undefined = []

        self._visited = {name: False for name in resources_dict}
        for name in resources_dict:
            self._analyze_tree(name)

    def _analyze_tree(self, name):
        # undefined variable:
        if name not in self.resources:
            self.undefined.append(name)
            return
        if self._visited[name]:
            return
        # cycle
        if name in self._request_stack:
            prefix = " -> ".join(self._request_stack)
            self.cycles.append('{} -> {}'.format(prefix, name))
            return

        self._request_stack.append(name)
        self._analyze_node(self.resources[name])
        self._request_stack.pop()

        self._visited[name] = True

    def _analyze_node(self, node):
        if type(node) is Resource:
            self._analyze_tree(node.name.body)
        if type(node) is GetAttribute:
            self._analyze_node(node.data)
        if type(node) is Module:
            for param in node.params:
                self._analyze_node(param.value)
        if type(node) is Array:
            for x in node.values:
                self._analyze_node(x)
        if type(node) is Dictionary:
            for key, value in node.dictionary.items():
                self._analyze_node(value)
