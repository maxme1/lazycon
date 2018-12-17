BadSyntaxError = SyntaxError


# class BadSyntaxError(SyntaxError):
#     pass


class DeprecationError(Exception):
    pass


class SemanticsError(SyntaxError):
    pass


class BuildConfigError(SyntaxError):
    pass


class ResourceError(NameError):
    pass
