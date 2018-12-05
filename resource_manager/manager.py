from collections import ChainMap
from itertools import chain, starmap

from .exceptions import custom_raise, BuildConfigError
from .scope import Scope, add_if_missing
from .parser import parse_file, parse_string
from .expressions import *
from .statements import *


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
        self._scope = Scope()

        # for name, definition in parse_file(path)[0]:
        #     self._scope.set_thunk(name, NodeThunk(compile(definition.body, path, 'eval')))

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
        path = self._resolve_path(path, '', '')
        result = self._import(path)
        self._update_resources(result)
        return self

    def string_input(self, source: str):
        """Interpret the `source`."""
        result = self._get_resources(*parse_string(source))
        self._update_resources(result)
        return self

    def render_config(self) -> str:
        """Generate a string containing definitions of all the resources in the current scope."""
        return self._scope.render_config()

    def save_config(self, path: str):
        """Render the config and save it to `path`."""
        with open(path, 'w') as file:
            file.write(self.render_config())

    def __getattr__(self, item):
        return self.get_resource(item)

    def __getitem__(self, item):
        return self.get_resource(item)

    def get_resource(self, name: str):
        return self._scope[name]

    def _update_resources(self, scope: dict):
        list(starmap(self._scope.add_statement, scope.items()))
        # self._node_levels = SyntaxTree.analyze(self._scope)

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

    def _get_resources(self, definitions: List[Union[ExpressionStatement, FuncDef]],
                       parents: List[ImportStarred], imports: List[UnifiedImport]) -> dict:

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
                try:
                    node = local[what]
                except KeyError:
                    raise BuildConfigError(
                        'Resource "%s" is not defined in the config it is imported from.\n' % what +
                        '  at %d:%d in %s' % import_.position()) from None
            else:
                node = import_
            # TODO: replace by a list
            add_if_missing(scope, name, node)

        for name, definition in definitions:
            add_if_missing(scope, name, definition)

        return dict(parent_scope.new_child(scope).items())

    def _resolve_path(self, path: str, source: str, shortcut: str):
        if shortcut:
            if shortcut not in self._shortcuts:
                message = 'Shortcut "%s" is not recognized' % shortcut
                if source:
                    message = 'Error while processing %s:\n ' % source + message
                custom_raise(BuildConfigError(message))
            path = os.path.join(self._shortcuts[shortcut], path)
        else:
            path = os.path.join(os.path.dirname(source), path)

        path = os.path.expanduser(path)
        return os.path.realpath(path)

    def __dir__(self):
        return list(set(self._scope.keys()) | set(super().__dir__()))

    def _ipython_key_completions_(self):
        return self._scope.keys()


read_config = ResourceManager.read_config
read_string = ResourceManager.read_string
