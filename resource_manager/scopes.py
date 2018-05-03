from collections import OrderedDict

from resource_manager.renderer import Renderer
from .structures import Structure


class GlobalScope:
    def __init__(self):
        self._defined_resources = OrderedDict()
        self._undefined_resources = OrderedDict()

    def overwrite(self, scope):
        assert not self._defined_resources and not scope._defined_resources
        for name, value in scope._undefined_resources.items():
            self._undefined_resources[name] = value

    # TODO: better names
    def set_resource(self, name: str, value: Structure):
        if name in self._undefined_resources:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        self._undefined_resources[name] = value

    def define_resource(self, name: str, value):
        if name in self._defined_resources:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        self._defined_resources[name] = value

    def get_resource(self, name: str):
        if name in self._defined_resources:
            return self._defined_resources[name]

        if name not in self._undefined_resources:
            raise AttributeError('Resource "{}" is not defined'.format(name))

        node = self._undefined_resources[name]
        resource = Renderer.render(node, self)
        self._defined_resources[name] = resource
        return resource


class LocalScope:
    def __init__(self, upper_scope):
        self._defined_resources = {}
        self._upper = upper_scope

    def define_resource(self, name: str, value):
        assert name not in self._defined_resources
        self._defined_resources[name] = value

    def get_resource(self, name: str):
        if name in self._defined_resources:
            return self._defined_resources[name]

        return self._upper.get_resource(name)
