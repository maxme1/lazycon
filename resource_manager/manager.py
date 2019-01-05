import os
from collections import OrderedDict
from itertools import starmap
from pathlib import Path
from typing import List

from .semantics import Semantics
from .wrappers import ImportStarred, UnifiedImport
from .exceptions import ResourceError, ExceptionWrapper
from .scope import Scope, add_if_missing, Builtins
from .parser import parse_file, parse_string


class ResourceManager:
    """
    A config interpreter.

    Parameters
    ----------
    shortcuts: dict, optional
        a dict that maps keywords to paths. It is used to resolve paths during import.
    injections: dict, optional
        a dict with default values that will be used in case the config doesn't define them.
    """
    # restricting setattr to these names
    __slots__ = '_shortcuts', '_imported_configs', '_scope', '_leave_time'

    def __init__(self, shortcuts: dict = None, injections: dict = None):
        self._shortcuts = shortcuts or {}
        self._imported_configs = {}
        self._scope = Scope(Builtins(injections or {}))
        self._leave_time = {}

    @classmethod
    def read_config(cls, path: str, shortcuts: dict = None, injections: dict = None):
        """
        Import the config located at `path` and return a ResourceManager instance.
        Also this methods adds a `__file__ = pathlib.Path(path)` value to the global scope.

        Parameters
        ----------
        path: str
            path to the config to import
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        injections: dict, optional
            a dict with default values that will be used in case the config doesn't define them.

        Returns
        -------
        resource_manager: ResourceManager
        """
        key = '__file__'
        injections = dict(injections or {})
        if key in injections:
            raise ValueError('The "__file__" key is not allowed in "injections".')

        injections[key] = Path(cls._standardize_path(path))
        return cls(shortcuts, injections).import_config(path)

    @classmethod
    def read_string(cls, source: str, shortcuts: dict = None, injections: dict = None):
        """
        Interpret the `source` and return a ResourceManager instance.

        Parameters
        ----------
        source: str
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        injections: dict, optional
            a dict with default values that will be used in case the config doesn't define them.

        Returns
        -------
        resource_manager: ResourceManager
        """
        return cls(shortcuts, injections).string_input(source)

    def import_config(self, path: str):
        """Import the config located at `path`."""
        self._update_resources(self._import(path))
        return self

    def string_input(self, source: str):
        """Interpret the `source`."""
        self._update_resources(self._get_resources(*parse_string(source)))
        return self

    def render_config(self) -> str:
        """Generate a string containing definitions of all the resources in the current scope."""
        return '\n'.join(self._scope.render(self._leave_time)).strip() + '\n'

    def save_config(self, path: str):
        """Render the config and save it to `path`."""
        with open(path, 'w') as file:
            file.write(self.render_config())

    def __getattr__(self, name: str):
        try:
            return self.get_resource(name)
        except ResourceError:
            raise AttributeError('"%s" is not defined.' % name) from None

    def __getitem__(self, name: str):
        try:
            return self.get_resource(name)
        except ResourceError:
            raise KeyError('"%s" is not defined.' % name) from None

    def get_resource(self, name: str):
        try:
            return self._scope[name]
        except ExceptionWrapper as e:
            raise e.exception from None

    def _update_resources(self, scope: OrderedDict):
        if self._scope.populated:
            raise RuntimeError('The scope has already been populated with live objects. Overwriting them might cause '
                               'undefined behaviour. Please, create another instance of ResourceManager.')

        updated_scope = self._scope.get_name_to_statement()
        updated_scope.update(scope)
        self._leave_time = Semantics.analyze(updated_scope, self._scope.parent)
        list(starmap(self._scope.update_value, scope.items()))

    @staticmethod
    def _standardize_path(path: str):
        path = os.path.expanduser(path)
        path = os.path.realpath(path)
        return path

    def _import(self, path: str) -> OrderedDict:
        path = self._standardize_path(path)

        if path in self._imported_configs:
            return self._imported_configs[path]
        # avoiding cycles
        self._imported_configs[path] = {}

        result = self._get_resources(*parse_file(path))
        self._imported_configs[path] = result
        return result

    def _get_resources(self, parents: List[ImportStarred], imports: List[UnifiedImport], definitions) -> OrderedDict:

        parent_scope = OrderedDict()
        for parent in parents:
            parent_scope.update(self._import(parent.get_path(self._shortcuts)))

        scope = OrderedDict()
        for name, import_ in imports:
            if import_.is_config_import(self._shortcuts):
                # TODO: should warn about ambiguous shortcut names:
                # importlib.util.find_spec(shortcut)
                local = self._import(import_.get_path(self._shortcuts))
                what = import_.what
                assert len(what) == 1
                what = what[0]
                if what not in local:
                    raise NameError('"%s" is not defined in the config it is imported from.\n' % what +
                                    '  at %d:%d in %s' % import_.position)
                node = local[what]
            else:
                node = import_
            # TODO: replace by a list
            add_if_missing(scope, name, node)

        for name, definition in definitions:
            add_if_missing(scope, name, definition)

        final_scope = OrderedDict(parent_scope.items())
        final_scope.update(scope)
        return final_scope

    def __dir__(self):
        return list(set(self._scope.keys()) | set(super().__dir__()))

    def _ipython_key_completions_(self):
        return self._scope.keys()

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except AttributeError:
            raise AttributeError('ResourceManager\'s attribute "%s" is read-only.' % name) from None


read_config = ResourceManager.read_config
read_string = ResourceManager.read_string
