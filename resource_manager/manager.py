import os
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Union, Dict, Any, Sequence

from .semantics import Semantics
from .exceptions import ResourceError, ExceptionWrapper, SemanticError, ConfigImportError
from .scope import Scope, Builtins, ScopeWrapper
from .parser import parse_file, parse_string, flatten_assignment

PathLike = Union[Path, str]


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
    __slots__ = '_shortcuts', '_imported_configs', '_scope', '_node_parents'

    def __init__(self, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
        self._shortcuts = shortcuts or {}
        self._imported_configs = {}
        self._scope = Scope(Builtins(injections or {}))
        self._node_parents = {}

    @classmethod
    def read_config(cls, path: PathLike, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
        """
        Import the config located at `path` and return a ResourceManager instance.
        Also this method adds a `__file__ = pathlib.Path(path)` value to the global scope.

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
            raise ValueError('The "%s" key is not allowed in "injections".' % key)

        injections[key] = Path(cls._standardize_path(path))
        return cls(shortcuts, injections).import_config(path)

    @classmethod
    def read_string(cls, source: str, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
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

    def import_config(self, path: PathLike):
        """Import the config located at `path`."""
        self._update_resources(self._import(path))
        return self

    def string_input(self, source: str):
        """Interpret the `source`."""
        self._update_resources(self._get_resources(*parse_string(source)))
        return self

    def update(self, **values: Any):
        """Update the scope by `values`."""
        self._scope.update_values(values)
        return self

    def render_config(self, entry_points: Union[Sequence[str], str] = None) -> str:
        """
        Generate a string containing definitions of resources in the current scope.

        Parameters
        ----------
        entry_points
            the definitions that should be kept (along with their dependencies).
            If None - all the definitions are rendered.
        """
        if isinstance(entry_points, str):
            entry_points = [entry_points]
        return '\n'.join(self._scope.render(self._node_parents, entry_points)).strip() + '\n'

    def save_config(self, path: str, entry_points: Union[Sequence[str], str] = None):
        """Render the config and save it to `path`. See `render_config` for details."""
        with open(path, 'w') as file:
            file.write(self.render_config(entry_points))

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

    def eval(self, expression: str):
        """Evaluate the given `expression`."""
        try:
            return eval(expression, ScopeWrapper(self._scope))
        except ExceptionWrapper as e:
            raise e.exception from None

    def _update_resources(self, scope: OrderedDict):
        self._scope.check_populated()

        updated_scope = self._scope.get_name_to_statement()
        updated_scope.update(scope)
        self._node_parents = Semantics.analyze(updated_scope, self._scope.parent)
        self._scope.update_statements(scope.items())

    @staticmethod
    def _standardize_path(path: PathLike) -> str:
        path = str(path)
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

    def _get_resources(self, parents, imports, definitions) -> OrderedDict:
        parent_scope = OrderedDict()
        for parent in parents:
            parent_scope.update(self._import(parent.get_path(self._shortcuts)))

        scope = []
        for name, node in imports:
            if node.potentially_config():
                try:
                    local = self._import(node.get_path(self._shortcuts))
                    what, = node.what
                    if what not in local:
                        raise NameError('"%s" is not defined in the config it is imported from.\n' % what +
                                        '  at %d:%d in %s' % node.position)
                    node = local[what]
                except ConfigImportError:
                    pass

            scope.append((name, node))

        scope.extend(definitions)
        duplicates = [
            name for name, count in
            Counter(sum([flatten_assignment(pattern) for pattern, _ in scope], [])).items() if count > 1
        ]
        if duplicates:
            source_path = (imports or definitions)[0][1].source_path
            raise SemanticError('Duplicate definitions found in %s:\n    %s' % (source_path, ', '.join(duplicates)))

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
