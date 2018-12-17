import os
from collections import ChainMap
from itertools import starmap
from typing import List

from .semantics import SyntaxTree
from .wrappers import ImportStarred, UnifiedImport
from .exceptions import BuildConfigError, ResourceError
from .scope import Scope, add_if_missing, Builtins
from .parser import parse_file, parse_string


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
        self._scope = Scope(Builtins())

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
        return cls(shortcuts).import_config(source_path)

    @classmethod
    def read_string(cls, source: str, shortcuts: dict = None):
        """
        Interpret the `source` and return a ResourceManager instance.

        Parameters
        ----------
        source: str
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.

        Returns
        -------
        resource_manager: ResourceManager
        """
        return cls(shortcuts).string_input(source)

    def import_config(self, path: str):
        """Import the config located at `path`."""
        self._update_resources(self._import(path))
        return self

    def string_input(self, source: str):
        """Interpret the `source`."""
        result = self._get_resources(*parse_string(source))
        self._update_resources(result)
        return self

    def render_config(self) -> str:
        """Generate a string containing definitions of all the resources in the current scope."""
        return '\n'.join(self._scope.render())

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
        return self._scope[name]

    def _update_resources(self, scope: dict):
        SyntaxTree.analyze(scope, self._scope.parent)
        list(starmap(self._scope.add_statement, scope.items()))

    def _import(self, path: str) -> dict:
        path = os.path.expanduser(path)
        path = os.path.realpath(path)

        if path in self._imported_configs:
            return self._imported_configs[path]
        # avoiding cycles
        self._imported_configs[path] = {}

        result = self._get_resources(*parse_file(path))
        self._imported_configs[path] = result
        return result

    def _get_resources(self, parents: List[ImportStarred], imports: List[UnifiedImport], definitions) -> dict:

        parent_scope = ChainMap()
        for parent in parents:
            parent_scope = parent_scope.new_child(self._import(parent.get_path(self._shortcuts)))

        scope = {}
        for name, import_ in imports:
            if import_.is_config_import(self._shortcuts):
                # TODO: should warn about ambiguous shortcut names:
                # importlib.util.find_spec(shortcut)
                local = self._import(import_.get_path(self._shortcuts))
                what = import_.what
                assert len(what) == 1
                what = what[0]
                try:
                    node = local[what]
                except KeyError:
                    raise BuildConfigError(
                        'Resource "%s" is not defined in the config it is imported from.\n' % what +
                        '  at %d:%d in %s' % import_.position) from None
            else:
                node = import_
            # TODO: replace by a list
            add_if_missing(scope, name, node)

        for name, definition in definitions:
            add_if_missing(scope, name, definition)

        return dict(parent_scope.new_child(scope).items())

    def __dir__(self):
        return list(set(self._scope.keys()) | set(super().__dir__()))

    def _ipython_key_completions_(self):
        return self._scope.keys()


read_config = ResourceManager.read_config
read_string = ResourceManager.read_string
