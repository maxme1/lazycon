class DeprecationError(Exception):
    pass


class SemanticError(SyntaxError):
    pass


class ResourceError(NameError):
    pass
