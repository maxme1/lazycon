from .structures import *


class NoDefaultValue:
    pass


class Parameter:
    def __init__(self, name, vararg=False, positional=True, keyword=True, default=NoDefaultValue):
        self.default_exp = default
        self.vararg = vararg
        self.keyword = keyword
        self.positional = positional
        self.name = name
        self.has_default_value = default is not NoDefaultValue
        assert not (vararg and self.has_default_value)

    def set_default_value(self, value):
        assert not hasattr(self, 'default_value')
        self.default_value = value
