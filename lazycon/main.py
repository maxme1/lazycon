import os
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Union, Dict, Any, Sequence

from .semantics import Semantics
from .exceptions import EntryError, ExceptionWrapper, SemanticError
from .scope import Scope, Builtins
from .parser import parse_file, parse_string
from .render import ScopeEval
from .statements import ImportConfig, Definitions

PathLike = Union[Path, str]


class Config:
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
    __slots__ = '_shortcuts', '_imported_configs', '_builtins', '_scope', '_extension', '_injections'

    def __init__(self, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
        self._shortcuts = shortcuts or {}
        self._imported_configs = {}
        self._injections = injections
        self._builtins = Builtins(injections or {})
        self._scope: Scope = Scope([], self._builtins, {})
        self._extension = '.config'

    @classmethod
    def load(cls, path: PathLike, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
        """
        Import the config located at `path` and return a Config instance.
        Also this method adds a `__file__ = pathlib.Path(path)` value to the global scope.

        Parameters
        ----------
        path: str, pathlib.Path
            path to the config to import
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        injections: dict, optional
            a dict with default values that will be used in case the config doesn't define them.

        Returns
        -------
        config: Config
        """
        # TODO: __folder__
        key = '__file__'
        injections = dict(injections or {})
        if key in injections:
            raise ValueError(f'The "{key}" key is not allowed in "injections".')

        injections[key] = cls._standardize_path(path)
        return cls(shortcuts, injections).file_input(path)

    @classmethod
    def loads(cls, source: str, shortcuts: Dict[str, PathLike] = None, injections: Dict[str, Any] = None):
        """
        Interpret the `source` and return a `Config` instance.

        Parameters
        ----------
        source: str
        shortcuts: dict, optional
            a dict that maps keywords to paths. It is used to resolve paths during import.
        injections: dict, optional
            a dict with default values that will be used in case the config doesn't define them.

        Returns
        -------
        config: Config
        """
        return cls(shortcuts, injections).string_input(source)

    def dumps(self, entry_points: Union[Sequence[str], str] = None) -> str:
        """
        Generate a string containing the names from the current scope.

        Parameters
        ----------
        entry_points
            the definitions that should be kept (along with their dependencies).
            If None - all the definitions are rendered.
        """
        if isinstance(entry_points, str):
            entry_points = [entry_points]
        return '\n'.join(self._scope.render(entry_points)).strip() + '\n'

    def dump(self, path: PathLike, entry_points: Union[Sequence[str], str] = None):
        """ Render the config and save it to `path`. See `dumps` for details. """
        with open(path, 'w') as file:
            file.write(self.dumps(entry_points))

    def file_input(self, path: PathLike) -> 'Config':
        """Import the config located at `path`."""
        self._update_scope(self._import(path))
        return self

    def string_input(self, source: str) -> 'Config':
        """Interpret the `source`."""
        self._update_scope(self._make_scope(*parse_string(source, self._extension)))
        return self

    def update(self, **values: Any) -> 'Config':
        """Update the scope by `values`."""
        try:
            self._scope.update_values(values)
            return self
        except RuntimeError:
            raise RuntimeError(
                'The scope has already been populated with live objects. Overwriting them might cause '
                'undefined behaviour. Please, create another instance or copy of Config: config.copy().update(...)'
            ) from None

    def copy(self) -> 'Config':
        """Create a copy of the config with an unpopulated scope."""
        config = type(self)(self._shortcuts, self._injections)
        config._scope = Scope.copy(self._scope)
        return config

    def __contains__(self, name: str):
        return name in self._scope

    def __getattr__(self, name: str):
        try:
            return self.get(name)
        except EntryError:
            raise AttributeError(f'"{name}" is not defined.')  # from None

    def keys(self):
        return self._scope.keys()

    def __iter__(self):
        yield from self.keys()

    def __getitem__(self, name: str):
        try:
            return self.get(name)
        except EntryError:
            raise KeyError(f'"{name}" is not defined.') from None

    def get(self, name: str):
        try:
            return self._scope[name]
        except ExceptionWrapper as e:
            raise e.exception from None

    def eval(self, expression: str):
        """Evaluate the given `expression`."""
        try:
            return eval(expression, ScopeEval(self._scope))
        except ExceptionWrapper as e:
            raise e.exception from None

    def _update_scope(self, scope: OrderedDict):
        self._scope.check_populated()
        # update
        new_scope = self._scope.definitions.copy()
        new_scope.update(scope)
        definitions = list(new_scope.values())
        # analysis
        tree = Semantics(definitions, self._builtins)
        tree.check()
        # building
        self._scope = Scope(definitions, self._builtins, tree.parents)

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

        result = self._make_scope(*parse_file(path, self._extension))
        self._imported_configs[path] = result
        return result

    def _make_scope(self, parents: Sequence[ImportConfig], definitions: Definitions) -> OrderedDict:
        parent_scope = OrderedDict()
        for parent in parents:
            parent_scope.update(self._import(parent.get_path(self._shortcuts)))

        duplicates = [name for name, count in Counter(x.name for x in definitions).items() if count > 1]
        if duplicates:
            source_path = definitions[0].statement.source_path
            duplicates = ', '.join(duplicates)
            raise SemanticError(f'Duplicate definitions found in {source_path}:\n    {duplicates}')

        final_scope = OrderedDict(parent_scope.items())
        final_scope.update((x.name, x) for x in definitions)
        return final_scope

    def __dir__(self):
        return list(set(self._scope.keys()) | set(super().__dir__()))

    def _ipython_key_completions_(self):
        return list(self._scope.keys())

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except AttributeError:
            raise AttributeError(f'Config\'s attribute "{name}" is read-only.') from None


load = Config.load
loads = Config.loads
