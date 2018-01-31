import os
import sys
import functools
import importlib
from typing import Callable
from collections import OrderedDict

from .structures import *
from .utils import put_in_stack
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
        """Render the config and save it to `path`"""
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

        try:
            node = self._undefined_resources[name]
            # a whole new request, so clear the stack
            resource = self._define_resource(node)
            self._defined_resources[name] = resource
            return resource

        except BaseException as e:
            # TODO: is it needed here?
            if not self._definitions_stack:
                raise

            definition = self._definitions_stack[-1]
            message = 'An exception occurred while ' + definition.error_message()
            message += '\n    at %d:%d in %s' % definition.position()
            raise RuntimeError(message) from e

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

    def _get_resources(self, definitions, imports, parents, absolute_path):
        parent_resources = {}
        for parent in parents:
            parent = self._resolve_path(parent, absolute_path)
            parent_resources.update(self._import(parent))

        result = {}
        for imp in imports:
            for what, as_ in imp.values.items():
                # TODO: too ugly
                if as_ is not None:
                    name = as_.body
                else:
                    name = what
                    packages = name.split('.')
                    if len(packages) > 1:
                        name = packages[0]
                self._set_definition(result, name, LazyImport(imp.root, what, as_, imp.main_token))

        for definition in definitions:
            self._set_definition(result, definition.name.body, definition.value)

        parent_resources.update(result)
        return parent_resources

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

    # TODO: get rid of this decorator?
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
            data = self._define_resource(node.target)
            return getattr(data, node.name.body)
        if type(node) is GetItem:
            target = self._define_resource(node.target)
            args = tuple(self._define_resource(arg) for arg in node.args)
            if not node.trailing_coma and len(args) == 1:
                args = args[0]
            return target[args]
        if type(node) is Module:
            if self.get_module is None:
                raise ValueError('The function "get_module" was not provided, so your modules are unreachable')
            return self.get_module(node.module_type.body, node.module_name.body)
        if type(node) is Call:
            target = self._define_resource(node.target)
            args = []
            for vararg, arg in zip(node.varargs, node.args):
                temp = self._define_resource(arg)
                if vararg:
                    args.extend(temp)
                else:
                    args.append(temp)
            kwargs = {param.name.body: self._define_resource(param.value) for param in node.params}
            if node.lazy:
                return functools.partial(target, *args, **kwargs)
            else:
                return target(*args, **kwargs)
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


read_config = ResourceManager.read_config


class TempPlaceholder:
    pass
