from collections import OrderedDict
from threading import Lock

from .renderer import Renderer
from .structures import Structure


class GlobalScope:
    def __init__(self):
        self._defined_resources = OrderedDict()
        self._undefined_resources = OrderedDict()
        self._local_locks = {}

    def overwrite(self, scope):
        if self._defined_resources and scope._undefined_resources:
            raise RuntimeError("The resource manager's scope already contains rendered resources. "
                               "Overwriting them may lead to undefined behaviour.")
        for name, value in scope._undefined_resources.items():
            self._undefined_resources[name] = value
            if name not in self._local_locks:
                self._local_locks[name] = Lock()

    def set_node(self, name: str, value: Structure):
        if name in self._undefined_resources:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        self._undefined_resources[name] = value
        self._local_locks[name] = Lock()

    def set_resource(self, name: str, value):
        assert name not in self._defined_resources and name in self._undefined_resources
        self._defined_resources[name] = value

    def get_resource(self, name: str):
        if name not in self._undefined_resources:
            raise AttributeError('Resource "{}" is not defined'.format(name))

        with self._local_locks[name]:
            if name in self._defined_resources:
                return self._defined_resources[name]

            node = self._undefined_resources[name]
            resource = Renderer.render(node, self)
            self._defined_resources[name] = resource
            return resource


class LocalScope:
    def __init__(self, upper_scope):
        self._defined_resources = {}
        self._upper = upper_scope

    def set_resource(self, name: str, value):
        assert name not in self._defined_resources
        self._defined_resources[name] = value

    def get_resource(self, name: str):
        try:
            return self._defined_resources[name]
        except KeyError:
            return self._upper.get_resource(name)
