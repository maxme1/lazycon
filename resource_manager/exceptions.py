class SemanticError(SyntaxError):
    pass


class ResourceError(NameError):
    pass


class ConfigImportError(ImportError):
    pass


class ExceptionWrapper(Exception):
    def __init__(self, exception):
        self.exception = exception
