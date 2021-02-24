class SemanticError(SyntaxError):
    pass


class EntryError(NameError):
    pass


class ConfigImportError(ImportError):
    pass


class ExceptionWrapper(Exception):
    def __init__(self, exception):
        self.exception = exception
