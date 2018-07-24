from .exceptions import custom_raise, BuildConfigError
from .renderer import Renderer
from .scopes import GlobalScope, add_if_missing
from .parser import parse_file, parse_string
from .expressions import *
from .statements import *
from .syntax_tree import SyntaxTree


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
        self._scope = GlobalScope()
        self._node_levels = {}

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
        return self._scope.get_resource(name, Renderer.make_renderer(self._scope, self._node_levels))

    def _update_resources(self, scope):
        self._scope.overwrite(scope)
        self._node_levels = SyntaxTree.analyze(self._scope)

    def _import(self, absolute_path: str) -> dict:
        if absolute_path in self._imported_configs:
            return self._imported_configs[absolute_path]
        # avoiding cycles
        self._imported_configs[absolute_path] = {}

        result = self._get_resources(*parse_file(absolute_path))
        self._imported_configs[absolute_path] = result
        return result

    def _get_resources(self, definitions: List[Union[Definition, FuncDef]],
                       parents: List[Union[ImportPath, ImportStarred]], imports: List[UnifiedImport]) -> dict:
        parent_scope = {}
        for parent in parents:
            source_path = parent.main_token.source
            shortcut, path = parent.get_path()
            path = self._resolve_path(path, source_path, shortcut)
            parent_scope.update(self._import(path))

        scope = {}
        for import_ in imports:
            for what, as_ in import_.iterate_values():
                if import_.is_config_import(self._shortcuts):
                    source_path = import_.main_token.source
                    shortcut, path = import_.get_path()
                    # TODO: should warn about ambiguous shortcut names:
                    # importlib.util.find_spec(shortcut)
                    local = self._import(self._resolve_path(path, source_path, shortcut))
                    try:
                        node = local[what]
                    except KeyError:
                        custom_raise(BuildConfigError(
                            'Resource "%s" is not defined in the config it is imported from.\n' % what +
                            '  at %d:%d in %s' % import_.position()), None)
                else:
                    node = LazyImport(import_.get_root(), what, as_, import_.main_token)

                add_if_missing(scope, get_imported_name(what, as_), node)

        for definition in definitions:
            if isinstance(definition, FuncDef):
                add_if_missing(scope, definition.name, definition)
            else:
                for name in definition.names:
                    add_if_missing(scope, name.body, definition.value)

        parent_scope.update(scope)
        return parent_scope

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
        return list(set(self._scope.get_resource_names()) | set(super().__dir__()))

    def _ipython_key_completions_(self):
        return self._scope.get_resource_names()


read_config = ResourceManager.read_config
read_string = ResourceManager.read_string
