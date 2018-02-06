from collections import OrderedDict

from .structures import Structure


class Scope:
    def __init__(self):
        self._defined_resources = OrderedDict()
        self._undefined_resources = OrderedDict()
        self._upper = None

    def set_upper(self, upper):
        assert self._upper is None
        self._upper = upper

    def overwrite(self, scope):
        assert not self._defined_resources and not scope._defined_resources
        for name, value in scope._undefined_resources.items():
            self._undefined_resources[name] = value

    def set_resource(self, name: str, value: Structure):
        if name in self._undefined_resources:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        self._undefined_resources[name] = value

    def define_resource(self, name: str, value):
        if name in self._defined_resources:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        self._defined_resources[name] = value

    def get_resource(self, name: str, renderer):
        if name in self._defined_resources:
            return self._defined_resources[name]

        if name not in self._undefined_resources:
            if self._upper is None:
                raise AttributeError('Resource "{}" is not defined'.format(name))
            return self._upper.get_resource(name, renderer)

        node = self._undefined_resources[name]
        resource = renderer(node)
        self._defined_resources[name] = resource
        return resource


class LambdaFunction:
    def __init__(self, node, scope, interpreter):
        self.node = node
        self.scope = scope
        self.interpreter = interpreter

    def __call__(self, *args):
        scope = Scope()
        for x, y in zip(self.node.params, args):
            scope.define_resource(x.body, y)
        scope.set_upper(self.scope)

        self.interpreter._scopes.append(scope)
        try:
            result = self.interpreter._render(self.node.expression)
        finally:
            self.interpreter._scopes.pop()
        return result
