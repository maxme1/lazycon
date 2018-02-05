import os
import sys
import functools
import importlib
from typing import Callable
from collections import OrderedDict

from .structures import *
from .parser import parse_file, parse_string
from .tree_analysis import SyntaxTree


class ResourceManager:
    """
    A config interpreter.

    Parameters
    ----------
    shortcuts: dict, optional
        a dict that maps keywords to paths. It is used to resolve paths during import.
    get_module: callable(module_type, module_name) -> object, optional
        a callable that loads external modules
    """

    def __init__(self, shortcuts: dict = None, get_module: Callable = None):
        self.get_module = get_module
        self._shortcuts = shortcuts or {}

        self._imported_configs = {}
        self._defined_resources = OrderedDict()
        self._undefined_resources = OrderedDict()
        self._request_stack = []
        self._definitions_stack = []

    @classmethod
    def read_config(cls, source_path: str, shortcuts: dict = None, get_module: Callable = None):
        """
        Import the config located at `source_path` and return a ResourceManager instance.

        Parameters
        ----------
        source_path: str
            path to the config to import
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        get_module: callable(module_type, module_name) -> object, optional
            a callable that loads external modules

        Returns
        -------
        resource_manager: ResourceManager
        """
        rm = cls(shortcuts, get_module)
        rm.import_config(source_path)
        return rm

    def import_config(self, path: str):
        """Import the config located at `path`."""
        path = self._resolve_path(path)
        result = self._import(path)
        self._update_resources(result)

    def string_input(self, source: str):
        """Interpret the `source`."""
        definitions, parents, imports = parse_string(source)
        result = self._get_resources(definitions, imports, parents, '')
        self._update_resources(result)

    def render_config(self) -> str:
        """
        Generate a string containing definitions of all the resources in the current scope.

        Returns
        -------
        config: str
        """
        result = ''
        for name, value in self._undefined_resources.items():
            if type(value) is LazyImport:
                result += value.to_str(0)
        if result:
            result += '\n'

        for name, value in self._undefined_resources.items():
            if type(value) is not LazyImport:
                result += '{} = {}\n\n'.format(name, value.to_str(0))

        return result[:-1]

    def save_config(self, path: str):
        """Render the config and save it to `path`."""
        with open(path, 'w') as file:
            file.write(self.render_config())

    def __getattr__(self, item):
        return self.get_resource(item)

    def get_resource(self, name: str):
        self._request_stack = []
        self._definitions_stack = []
        return self._get_resource(name)

    def _get_resource(self, name: str):
        if name in self._defined_resources:
            return self._defined_resources[name]

        if name not in self._undefined_resources:
            raise AttributeError('Resource "{}" is not defined'.format(name))

        node = self._undefined_resources[name]
        try:
            resource = self._define_resource(node)
        except BaseException as e:
            if not self._definitions_stack:
                raise
            # TODO: should all the traceback be printed?
            definition = self._definitions_stack[0]
            self._definitions_stack = []

            raise RuntimeError('An exception occurred while ' + definition.error_message() +
                               '\n    at %d:%d in %s' % definition.position()) from e

        self._defined_resources[name] = resource
        return resource

    def _update_resources(self, resources):
        # TODO: what if some of the resources were already rendered?
        self._undefined_resources.update(resources)
        self._semantic_analysis()

    def _semantic_analysis(self):
        # TODO: separate
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

    def _import(self, absolute_path: str):
        if absolute_path in self._imported_configs:
            return self._imported_configs[absolute_path]
        # avoiding cycles
        self._imported_configs[absolute_path] = {}

        definitions, parents, imports = parse_file(absolute_path)
        result = self._get_resources(definitions, imports, parents, absolute_path)
        self._imported_configs[absolute_path] = result
        return result

    @staticmethod
    def _set_definition(registry, name, value):
        if name in registry:
            raise SyntaxError('Duplicate definition of resource "%s" in %s' % (name, value.source()))
        registry[name] = value

    def _get_resources(self, definitions: List[Definition], imports: List[ImportPython], parents, absolute_path):
        parent_resources = {}
        for parent in parents:
            for path in parent.get_paths():
                path = self._resolve_path(path, absolute_path)
                parent_resources.update(self._import(path))

        result = {}
        for import_ in imports:
            for what, as_ in import_.values:
                if as_ is not None:
                    name = as_.body
                else:
                    name = what
                    packages = name.split('.')
                    if len(packages) > 1:
                        name = packages[0]
                self._set_definition(result, name,
                                     LazyImport(import_.root, what, as_, import_.relative, import_.main_token))

        for definition in definitions:
            self._set_definition(result, definition.name.body, definition.value)

        parent_resources.update(result)
        return parent_resources

    # TODO: change signature to path, source, shortcut
    def _resolve_path(self, path: str, source: str = ''):
        if path.count(':') > 1:
            raise SyntaxError('The path cannot contain more than one ":" separator.')

        parts = path.split(':', 1)
        if len(parts) > 1:
            shortcut, root = parts
            if shortcut not in self._shortcuts:
                message = 'Shortcut "%s" is not recognized' % shortcut
                if source:
                    message = 'Error while processing %s:\n ' % source + message
                raise ValueError(message)
            path = os.path.join(self._shortcuts[shortcut], root)
        else:
            path = os.path.join(os.path.dirname(source), path)

        path = os.path.expanduser(path)
        return os.path.realpath(path)

    def _define_resource(self, node: Structure):
        self._definitions_stack.append(node)
        value = node.render(self)
        self._definitions_stack.pop()
        return value


read_config = ResourceManager.read_config
