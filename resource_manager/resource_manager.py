import json
import functools

from .parser import parse_file
from .structures import *


class ResourceManager:
    def __init__(self, source_path, get_module):
        self._get_module = get_module
        definitions = parse_file(source_path)
        self._undefined_resources = self._get_resources_dict(definitions)
        self._defined_resources = {}

    def __getattr__(self, name):
        return self._get_resource(name)

    def _get_resources_dict(self, definitions):
        result = {}
        for definition in definitions:
            result[definition.name.body] = definition.value
        return result

    def _get_resource(self, name: str):
        try:
            return self._defined_resources[name]
        except KeyError:
            if name not in self._undefined_resources:
                raise AttributeError(f'Resource {repr(name)} is not defined')
            node = self._undefined_resources[name]
            resource = self._define_resource(node)
            self._defined_resources[name] = resource
            return resource

    def _define_resource(self, node):
        if type(node) is Value:
            return json.loads(node.value.body)
        if type(node) is Array:
            return [self._define_resource(x) for x in node.values]
        if type(node) is Dictionary:
            return {json.loads(name.body): self._define_resource(value) for name, value in node.dictionary.items()}
        if type(node) is Resource:
            return self._get_resource(node.name.body)
        if type(node) is Module:
            constructor = self._get_module(node.module_type.body, node.module_name.body)
            kwargs = {param.name.body: self._define_resource(param.value) for param in node.params}
            # by default init is True
            if node.init is None or json.loads(node.init.value.body):
                return constructor(**kwargs)
            else:
                return functools.partial(constructor, **kwargs)

        raise TypeError(f'Undefined resource description of type {type(node)}')
