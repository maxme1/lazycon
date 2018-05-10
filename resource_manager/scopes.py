import builtins
from collections import OrderedDict
from contextlib import suppress
from threading import Lock

from .renderer import Renderer
from .structures import Structure


class GlobalScope:
    def __init__(self):
        self._defined_resources = OrderedDict()
        self._undefined_resources = OrderedDict()
        self._local_locks = {}
        self.builtins = {x: getattr(builtins, x) for x in dir(builtins) if not x.startswith('_')}

    def overwrite(self, scope):
        # TODO: check if really something will be overwritten. (thread-safe)
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

    def get_resource(self, name: str, renderer=None):
        with suppress(KeyError):
            return self._defined_resources[name]
        with suppress(KeyError):
            return self.builtins[name]

        if name not in self._undefined_resources:
            raise AttributeError('Resource "{}" is not defined'.format(name))

        # render the resource
        with self._local_locks[name]:
            with suppress(KeyError):
                return self._defined_resources[name]

            node = self._undefined_resources[name]
            if renderer is None:
                resource = Renderer.render(node, self)
            else:
                resource = renderer(node)
            self._defined_resources[name] = resource
            return resource


class LocalScope:
    def __init__(self, upper_scope):
        self._defined_resources = {}
        self._upper = upper_scope

    def set_resource(self, name: str, value):
        assert name not in self._defined_resources
        self._defined_resources[name] = value

    def get_resource(self, name: str, renderer=None):
        with suppress(KeyError):
            return self._defined_resources[name]
        return self._upper.get_resource(name, renderer)
