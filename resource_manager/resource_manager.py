from .scopes import GlobalScope
from .parser import parse_file, parse_string
from .structures import *
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
        return self

    def string_input(self, source: str):
        """Interpret the `source`."""
        result = self._get_resources(*parse_string(source))
        self._update_resources(result)
        return self

    def render_config(self) -> str:
        """
        Generate a string containing definitions of all the resources in the current scope.

        Returns
        -------
        config: str
        """
        result = ''
        for name, value in self._scope._undefined_resources.items():
            if type(value) is LazyImport:
                result += value.to_str(0)
        if result:
            result += '\n'

        for name, value in self._scope._undefined_resources.items():
            if type(value) is not LazyImport:
                result += '{} = {}\n\n'.format(name, value.to_str(0))

        return result[:-1]

    def save_config(self, path: str):
        """Render the config and save it to `path`."""
        with open(path, 'w') as file:
            file.write(self.render_config())

    def __getattr__(self, item):
        return self.get_resource(item)

    def __getitem__(self, item):
        return self.get_resource(item)

    def get_resource(self, name: str):
        return self._scope.get_resource(name)

    def _update_resources(self, scope):
        self._scope.overwrite(scope)
        SyntaxTree.analyze(self._scope)

    def _import(self, absolute_path: str) -> GlobalScope:
        if absolute_path in self._imported_configs:
            return self._imported_configs[absolute_path]
        # avoiding cycles
        self._imported_configs[absolute_path] = GlobalScope()

        result = self._get_resources(*parse_file(absolute_path))
        self._imported_configs[absolute_path] = result
        return result

    def _get_resources(self, definitions: List[Definition], parents: List[Union[ImportPath, ImportStarred]],
                       imports: List[UnifiedImport]):
        parent_scope = GlobalScope()
        for parent in parents:
            source_path = parent.main_token.source
            shortcut, path = parent.get_path()
            path = self._resolve_path(path, source_path, shortcut)
            parent_scope.overwrite(self._import(path))

        scope = GlobalScope()
        for import_ in imports:
            for what, as_ in import_.iterate_values():
                if import_.is_config_import(self._shortcuts):
                    source_path = import_.main_token.source
                    shortcut, path = import_.get_path()
                    # TODO: should warn about ambiguous shortcut names:
                    # importlib.util.find_spec(shortcut)
                    local = self._import(self._resolve_path(path, source_path, shortcut))
                    value = local._undefined_resources[what]
                else:
                    value = LazyImport(import_.get_root(), what, as_, import_.main_token)

                name = get_imported_name(what, as_)
                scope.set_node(name, value)

        for definition in definitions:
            scope.set_node(definition.name.body, definition.value)

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


read_config = ResourceManager.read_config
