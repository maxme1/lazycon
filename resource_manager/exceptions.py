BadSyntaxError = SyntaxError


# class BadSyntaxError(SyntaxError):
#     pass


class SemanticsError(SyntaxError):
    pass


class BuildConfigError(SyntaxError):
    pass


class ResourceError(NameError):
    pass
