from collections import namedtuple


class NoDefaultValue:
    pass


class Parameter:
    def __init__(self, name, vararg=False, positional=True, keyword=True, default=NoDefaultValue):
        self.default_expression = default
        self.keyword = keyword
        self.positional = positional
        self.name = name
        self.has_default_value = default is not NoDefaultValue
        self.vararg = vararg
        assert not (vararg and self.has_default_value)

    def default_value(self, renderer):
        # TODO: this should be implemented outside
        if not hasattr(self, '_default_value'):
            self._default_value = renderer(self.default_expression)
        return self._default_value


class PositionalArgument(namedtuple('PA', ['vararg', 'value'])):
    def to_str(self, level):
        result = self.value.to_str(level)
        if self.vararg:
            result = '*' + result
        return result


class KeywordArgument(namedtuple('KA', ['name', 'value'])):
    def to_str(self, level):
        return self.name.body + '=' + self.value.to_str(level)


class VariableKeywordArgument(namedtuple('VKA', ['value'])):
    def to_str(self, level):
        return '**' + self.value.to_str(level)
