import functools
import json
import os
import traceback
from collections import OrderedDict

from .parser import parse_file
from .structures import *


class ResourceManager:
    def __init__(self, source_path: str, get_module: callable):
        self.get_module = get_module
        source_path = os.path.realpath(source_path)
        self._imported = OrderedDict()
        self._defined_resources = {}
        self._request_stack = []

        self._undefined_resources = self._get_resources_dict(source_path)

    def __getattr__(self, name: str):
        return self._get_resource(name)

    def get(self, name: str, default=None):
        try:
            return self._get_resource(name)
        except AttributeError:
            return default

    def set(self, name, value, override=False):
        if name in self._defined_resources and not override:
            raise RuntimeError(f'Attempt to overwrite resource {name}')
        self._defined_resources[name] = value

    def save_whole_config(self, path):
        with open(path, 'w') as file:
            file.write(self._get_whole_config())

    def _get_whole_config(self):
        added = set()
        result = ''
        for name, value in self._undefined_resources.items():
            added.add(name)
            result += f'{name} = {value.to_str(0)}\n\n'

        for config in self._imported.values():
            for name, value in config.items():
                if name not in added:
                    added.add(name)
                    result += f'{name} = {value.to_str(0)}\n\n'
        return result[:-1]

    def _get_resources_dict(self, absolute_path):
        definitions, parents = parse_file(absolute_path)
        result = {}
        for definition in definitions:
            result[definition.name.body] = definition.value
        for parent in parents:
            parent = os.path.join(os.path.dirname(absolute_path), parent)
            if parent not in self._imported:
                # avoiding cycles
                self._imported[parent] = None
                self._imported[parent] = self._get_resources_dict(parent)
        return result

    def _get_resource(self, name: str):
        if name in self._defined_resources:
            return self._defined_resources[name]
        # avoiding cycles
        if name in self._request_stack:
            self._request_stack = []
            raise RuntimeError('Cyclic dependency found in the following resource:\n  '
                               f'{" -> ".join(self._request_stack)} -> {name}')
        self._request_stack.append(name)

        resource = self._define_resource(self._get_node(name))
        self._defined_resources[name] = resource

        self._request_stack.pop()
        return resource

    def _get_node(self, name):
        try:
            return self._undefined_resources[name]
        except KeyError:
            for config in self._imported.values():
                if name in config:
                    result = config[name]
                    self._undefined_resources[name] = result
                    return result

        raise AttributeError(f'Resource {repr(name)} is not defined') from None

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
            try:
                constructor = self.get_module(node.module_type.body, node.module_name.body)
                kwargs = {param.name.body: self._define_resource(param.value) for param in node.params}
                # by default init is True
                if node.init is None or json.loads(node.init.value.body):
                    return constructor(**kwargs)
                else:
                    return functools.partial(constructor, **kwargs)
            except BaseException as e:
                raise RuntimeError(f'An exception occured while building resource '
                                   f'"{node.module_name.body}" of type "{node.module_type.body}"') from e

        raise TypeError(f'Undefined resource description of type {type(node)}')
