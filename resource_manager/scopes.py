import builtins
from collections import OrderedDict, defaultdict
from contextlib import suppress
from threading import Lock

from .exceptions import custom_raise, BuildConfigError, BadSyntaxError, LambdaArgumentsError
from .renderer import Renderer
from .structures import Structure, LazyImport


class Scope:
    def __init__(self):
        self._defined_resources = OrderedDict()
        self._name_to_node = {}
        self._node_to_names = defaultdict(set)
        self._node_locks = {}

    def _update_node_to_names(self):
        for name, node in self._name_to_node.items():
            self._node_to_names[node].add(name)

    def set_node(self, name: str, node: Structure):
        if name in self._name_to_node:
            custom_raise(BadSyntaxError('Duplicate definition of resource "%s" in %s' % (name, node.source())))
        self._name_to_node[name] = node
        self._update_node_to_names()
        self._node_locks[node] = Lock()

    def render_resource(self, name: str, renderer=None):
        node = self._name_to_node[name]
        with self._node_locks[node]:
            with suppress(KeyError):
                return self._defined_resources[name]

            if renderer is None:
                resource = Renderer.render(node, self)
            else:
                resource = renderer(node)
            for name_ in self._node_to_names[node]:
                self._defined_resources[name_] = resource
            return resource


class GlobalScope(Scope):
    def __init__(self):
        super().__init__()
        self.builtins = {x: getattr(builtins, x) for x in dir(builtins) if not x.startswith('_')}

    def render_config(self):
        groups = defaultdict(list)
        plain = []
        for node, names in self._node_to_names.items():
            if type(node) is LazyImport:
                assert len(names) == 1
                if node.from_:
                    groups[node.from_].append(node)
                else:
                    plain.append(node)

        result = ''
        for node in plain:
            result += node.to_str(0)
        if plain:
            result += '\n'

        for group in groups.values():
            result += group[0].from_to_str() + 'import ' + ', '.join(node.what_to_str() for node in group) + '\n'
        if groups:
            result += '\n'

        for node, names in self._node_to_names.items():
            if type(node) is not LazyImport:
                result += ' = '.join(sorted(names)) + ' = %s\n\n' % node.to_str(0)

        return result[:-1]

    def overwrite(self, scope: Scope):
        for name in scope._name_to_node.keys():
            with suppress(KeyError):
                node = self._name_to_node[name]
                if (node in self._node_locks and self._node_locks[node].locked()) or name in self._defined_resources:
                    custom_raise(BuildConfigError('The resource "%s" is already rendered. '
                                                  'Overwriting it may lead to undefined behaviour.' % name))

        for name, node in scope._name_to_node.items():
            self._name_to_node[name] = node
            if node not in self._node_locks:
                self._node_locks[node] = Lock()
        self._update_node_to_names()

    def get_resource(self, name: str, renderer=None):
        with suppress(KeyError):
            return self._defined_resources[name]
        with suppress(KeyError):
            return self.builtins[name]

        if name not in self._name_to_node:
            raise AttributeError('Resource "{}" is not defined'.format(name))

        return self.render_resource(name, renderer)


class LocalScope(Scope):
    def __init__(self, upper_scope: Scope):
        super().__init__()
        self._upper = upper_scope

    def set_resource(self, name: str, value):
        if name in self._defined_resources:
            custom_raise(LambdaArgumentsError('Duplicate argument: ' + name))
        self._defined_resources[name] = value

    def get_resource(self, name: str, renderer=None):
        with suppress(KeyError):
            return self._defined_resources[name]
        return self.render_resource(name, renderer)

    def __contains__(self, item):
        return item in self._defined_resources
