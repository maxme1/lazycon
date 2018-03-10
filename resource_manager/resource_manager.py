import functools
import importlib
import sys
from typing import Union

from .helpers import Scope
from .parser import parse_file, parse_string
from .structures import *
from .tree_analysis import SyntaxTree


class ResourceManager:
    """
    A config interpreter.

    Parameters
    ----------
    shortcuts: dict, optional
        a dict that maps keywords to paths. It is used to resolve paths during import.
    """

    def __init__(self, shortcuts: dict = None):
        self._shortcuts = shortcuts or {}

        self._imported_configs = {}
        self._scopes = [Scope()]
        self._request_stack = []
        self._definitions_stack = []

    @classmethod
    def read_config(cls, source_path: str, shortcuts: dict = None):
        """
        Import the config located at `source_path` and return a ResourceManager instance.

        Parameters
        ----------
        source_path: str
            path to the config to import
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.

        Returns
        -------
        resource_manager: ResourceManager
        """
        rm = cls(shortcuts)
        rm.import_config(source_path)
        return rm

    def import_config(self, path: str):
        """Import the config located at `path`."""
        path = self._resolve_path(path, '', '')
        result = self._import(path)
        self._update_resources(result)

    def string_input(self, source: str):
        """Interpret the `source`."""
        result = self._get_resources(*parse_string(source))
        self._update_resources(result)

    def render_config(self) -> str:
        """
        Generate a string containing definitions of all the resources in the current scope.

        Returns
        -------
        config: str
        """
        assert len(self._scopes) == 1
        scope = self._scopes[0]
        result = ''
        for name, value in scope._undefined_resources.items():
            if type(value) is LazyImport:
                result += value.to_str(0)
        if result:
            result += '\n'

        for name, value in scope._undefined_resources.items():
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
        assert len(self._scopes) == 1
        return self._scopes[0].get_resource(name, self._render)

    def _update_resources(self, scope):
        assert len(self._scopes) == 1
        # TODO: what if some of the resources were already rendered?
        self._scopes[0].overwrite(scope)
        SyntaxTree.analyze(self._scopes[0])

    def _import(self, absolute_path: str) -> Scope:
        if absolute_path in self._imported_configs:
            return self._imported_configs[absolute_path]
        # avoiding cycles
        self._imported_configs[absolute_path] = Scope()

        result = self._get_resources(*parse_file(absolute_path))
        self._imported_configs[absolute_path] = result
        return result

    def _get_resources(self, definitions: List[Definition], parents, imports: List[Union[ImportPython, ImportPartial]]):
        parent_scope = Scope()
        for parent in parents:
            source_path = parent.main_token.source
            for shortcut, path in parent.get_paths():
                path = self._resolve_path(path, source_path, shortcut)
                parent_scope.overwrite(self._import(path))

        scope = Scope()
        for import_ in imports:
            for what, as_ in import_.values:
                name = get_imported_name(what, as_)
                if type(import_) is ImportPython:
                    value = LazyImport(import_.root, what, as_, import_.main_token)
                else:
                    source_path = import_.main_token.source
                    shortcut, path = import_.get_paths()[0]
                    local = self._import(self._resolve_path(path, source_path, shortcut))
                    value = local._undefined_resources[what]

                scope.set_resource(name, value)

        for definition in definitions:
            scope.set_resource(definition.name.body, definition.value)

        parent_scope.overwrite(scope)
        return parent_scope

    def _resolve_path(self, path: str, source: str, shortcut: str):
        if shortcut:
            if shortcut not in self._shortcuts:
                message = 'Shortcut "%s" is not recognized' % shortcut
                if source:
                    message = 'Error while processing %s:\n ' % source + message
                raise ValueError(message)
            path = os.path.join(self._shortcuts[shortcut], path)
        else:
            path = os.path.join(os.path.dirname(source), path)

        path = os.path.expanduser(path)
        return os.path.realpath(path)

    def _render(self, node: Structure):
        self._definitions_stack.append(node)
        try:
            value = node.render(self)
        except BaseException as e:
            if not self._definitions_stack:
                raise
            # TODO: should all the traceback be printed?
            definition = self._definitions_stack[0]
            self._definitions_stack = []

            raise RuntimeError('An exception occurred while ' + definition.error_message() +
                               '\n    at %d:%d in %s' % definition.position()) from e
        self._definitions_stack.pop()
        return value

    def _render_lambda(self, node: Lambda):
        assert self._scopes
        upper_scope = self._scopes[-1]

        def f(*args):
            if len(args) != len(node.params):
                raise ValueError('Function requires %d argument(s), but %d provided' % (len(node.params), len(args)))
            scope = Scope()
            for x, y in zip(node.params, args):
                scope.define_resource(x.body, y)
            scope.set_upper(upper_scope)

            self._scopes.append(scope)
            try:
                result = self._render(node.expression)
            finally:
                self._scopes.pop()
            return result

        return f

    def _render_resource(self, node: Resource):
        assert self._scopes
        return self._scopes[-1].get_resource(node.name.body, self._render)

    def _render_get_attribute(self, node: GetAttribute):
        data = self._render(node.target)
        return getattr(data, node.name.body)

    def _render_get_item(self, node: GetItem):
        target = self._render(node.target)
        args = tuple(self._render(arg) for arg in node.args)
        if not node.trailing_coma and len(args) == 1:
            args = args[0]
        return target[args]

    def _render_call(self, node: Call):
        target = self._render(node.target)
        args = []
        for vararg, arg in zip(node.varargs, node.args):
            temp = self._render(arg)
            if vararg:
                args.extend(temp)
            else:
                args.append(temp)
        kwargs = {param.name.body: self._render(param.value) for param in node.params}
        if node.lazy:
            return functools.partial(target, *args, **kwargs)
        return target(*args, **kwargs)

    def _render_literal(self, node: Literal):
        return eval(node.value.body)

    def _render_number(self, node: Number):
        num = eval(node.value.body)
        if node.minus:
            num = -num
        return num

    def _render_array(self, node: Array):
        return [self._render(x) for x in node.values]

    def _render_dictionary(self, node: Dictionary):
        return {self._render(key): self._render(value) for key, value in node.pairs}

    def _render_lazy_import(self, node: LazyImport):
        if not node.from_:
            result = importlib.import_module(node.what)
            packages = node.what.split('.')
            if len(packages) > 1 and not node.as_:
                # import a.b.c
                return sys.modules[packages[0]]
            return result
        try:
            return getattr(importlib.import_module(node.from_), node.what)
        except AttributeError:
            pass
        try:
            return importlib.import_module(node.what, node.from_)
        except ModuleNotFoundError:
            return importlib.import_module(node.from_ + '.' + node.what)


read_config = ResourceManager.read_config
