import functools
import importlib
import os
import warnings
from collections import OrderedDict

import sys

from resource_manager.utils import put_in_stack
from .parser import parse_file
from .structures import *
from .tree_analysis import SyntaxTree


class ResourceManager:
    def __init__(self, source_path: str, get_module: callable, path_map: dict = None):
        """
        A config interpreter

        Parameters
        ----------
        source_path: str
            path to the config file
        get_module: callable(module_type, module_name) -> object
            a callable that loads external modules
        path_map: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        """
        # TODO: rename path_map to shortcuts
        self.get_module = get_module
        self._imported_configs = OrderedDict()
        self._defined_resources = {}
        self._undefined_resources = {}
        self.path_map = path_map or {}

        self._import_config(self._resolve_path(source_path))

        # TODO: probably move to a method
        tree = SyntaxTree(self._undefined_resources)
        message = ''
        if tree.cycles:
            message += 'Cyclic dependencies found in the following resources:\n'
            for source, cycles in tree.cycles.items():
                message += '  in %s\n    ' % source
                message += '\n    '.join(cycles)
                message += '\n'
        if tree.undefined:
            message += '\nUndefined resources found:\n'
            for source, undefined in tree.undefined.items():
                message += '  in %s\n    ' % source
                message += ', '.join(undefined)
                message += '\n'
        if message:
            raise RuntimeError(message)

    def __getattribute__(self, name: str):
        # TODO: looks kinda ugly. not sure if it's worth it
        try:
            value = super().__getattribute__(name)
            if value is not TempPlaceholder:
                return value
        except AttributeError:
            pass
        # a whole new request, so clear the stack
        self._request_stack = []
        self._definitions_stack = []
        try:
            return self._get_resource(name)
        except BaseException as e:
            if not self._definitions_stack:
                raise

            definition = self._definitions_stack[-1]
            message = 'An exception occurred while ' + definition.error_message()
            message += '\n    at %d:%d in %s' % definition.position()
            raise RuntimeError(message) from e

    def get(self, name: str, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def set(self, name, value, override=False):
        warnings.warn("Manually modifying the ResourceManager's state is highly not recommended", RuntimeWarning)
        if name in self._defined_resources and not override:
            raise RuntimeError('Attempt to overwrite resource {}'.format(name))
        self._defined_resources[name] = value
        self._add_to_dict(name, value)

    def save_config(self, path):
        with open(path, 'w') as file:
            file.write(self._get_whole_config())

    def _get_whole_config(self):
        result = ''
        for name, value in self._undefined_resources.items():
            if type(value) is LazyImport:
                result += value.to_str(0)

        for name, value in self._undefined_resources.items():
            if type(value) is not LazyImport:
                result += '{} = {}\n\n'.format(name, value.to_str(0))

        return result[:-1]

    def _add_to_dict(self, name, value):
        # adding support for interactive shells and notebooks:
        # TODO: this may be ugly, but is the only way I found to avoid triggering getattr
        if name not in self.__dict__:
            setattr(self, name, value)

    def _set_definition(self, registry, name, value, absolute_path):
        if name in registry:
            raise SyntaxError('Duplicate definition of resource "{}" '
                              'in config file {}'.format(name, absolute_path))
        registry[name] = value
        if name not in self._undefined_resources:
            self._undefined_resources[name] = value
            self._add_to_dict(name, TempPlaceholder)

    def _resolve_path(self, path: str, source: str = None):
        parts = path.split(':', 1)
        if len(parts) > 1:
            shortcut = parts[0]
            if shortcut not in self.path_map:
                message = 'Shortcut %s not in recognized' % shortcut
                if source:
                    message = 'Error while processing file {}:\n ' % source + message
                raise ValueError(message)
            path = os.path.join(self.path_map[shortcut], parts[1])

        path = os.path.expanduser(path)
        return os.path.realpath(path)

    def _import_config(self, source_path):
        if source_path in self._imported_configs:
            return
        self._imported_configs[source_path] = None

        definitions, parents, imports = parse_file(source_path)
        result = {}
        for imp in imports:
            for what, as_ in imp.values.items():
                if as_ is not None:
                    name = as_.body
                else:
                    name = what
                    packages = name.split('.')
                    if len(packages) > 1:
                        name = packages[0]
                self._set_definition(result, name, LazyImport(imp.root, what, as_, imp.main_token), source_path)

        for definition in definitions:
            self._set_definition(result, definition.name.body, definition.value, source_path)

        for parent in reversed(parents):
            if ':' not in parent:
                parent = os.path.join(os.path.dirname(source_path), parent)
            parent = self._resolve_path(parent, source_path)
            self._import_config(parent)

        self._imported_configs[source_path] = result

    def _get_resource(self, name: str):
        if name in self._defined_resources:
            return self._defined_resources[name]

        try:
            node = self._undefined_resources[name]
        except KeyError:
            raise AttributeError('Resource "{}" is not defined'.format(name)) from None

        resource = self._define_resource(node)
        self._defined_resources[name] = resource
        return resource

    @put_in_stack
    def _define_resource(self, node):
        if type(node) is Literal:
            return eval(node.value.body)
        if type(node) is Array:
            return [self._define_resource(x) for x in node.values]
        if type(node) is Dictionary:
            return {eval(name.body): self._define_resource(value) for name, value in node.dictionary.items()}
        if type(node) is Resource:
            return self._get_resource(node.name.body)
        if type(node) is GetAttribute:
            data = self._define_resource(node.data)
            return getattr(data, node.name.body)
        if type(node) is Module:
            return self.get_module(node.module_type.body, node.module_name.body)
        if type(node) is Partial:
            target = self._define_resource(node.target)
            kwargs = {param.name.body: self._define_resource(param.value) for param in node.params}
            if node.lazy:
                return functools.partial(target, **kwargs)
            else:
                return target(**kwargs)
        if type(node) is LazyImport:
            if not node.from_:
                result = importlib.import_module(node.what)
                packages = node.what.split('.')
                if len(packages) > 1 and not node.as_:
                    # import a.b.c
                    return sys.modules[packages[0]]
                return result
            try:
                return importlib.import_module(node.what, node.from_)
            except ModuleNotFoundError:
                return getattr(importlib.import_module(node.from_), node.what)

        raise TypeError('Undefined resource description of type {}'.format(type(node)))


class TempPlaceholder:
    pass
