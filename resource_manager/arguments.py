from collections import namedtuple


class NoDefaultValue:
    pass


class Parameter:
    def __init__(self, name, vararg=False, positional=True, keyword=True, default=NoDefaultValue):
        self.default_exp = default
        self.default_value = NoDefaultValue
        self.vararg = vararg
        self.keyword = keyword
        self.positional = positional
        self.name = name
        self.has_default_value = default is not NoDefaultValue
        assert not (vararg and self.has_default_value)

    def set_default_value(self, value):
        assert self.default_value is NoDefaultValue
        self.default_value = value


class PositionalArgument(namedtuple('PA', ['vararg', 'value'])):
    def to_str(self, level):
        result = self.value.to_str(level)
        if self.vararg:
            result = '*' + result
        return result


class KeywordArgument(namedtuple('KA', ['name', 'value'])):
    def to_str(self, level):
        return self.name.body + '=' + self.value.to_str(level)
