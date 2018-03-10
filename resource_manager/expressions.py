from typing import List

from .structures import Structure, MAX_COLUMNS
from .token import TokenWrapper


class Lambda(Structure):
    def __init__(self, params: List[TokenWrapper], expression: Structure, main_token):
        super().__init__(main_token)
        self.expression = expression
        self.params = params

    def to_str(self, level):
        return 'lambda ' + ','.join(x.body for x in self.params) + ': ' + self.expression.to_str(level + 1)


class Resource(Structure):
    def __init__(self, name: TokenWrapper):
        super().__init__(name)
        self.name = name

    def to_str(self, level):
        return self.name.body


class GetAttribute(Structure):
    def __init__(self, target: Structure, name: TokenWrapper):
        super().__init__(name)
        self.target = target
        self.name = name

    def to_str(self, level):
        return '{}.{}'.format(self.target.to_str(level), self.name.body)

    def error_message(self):
        return 'getting attribute {} from {}'.format(self.name.body, self.target.to_str(0))


class GetItem(Structure):
    def __init__(self, target: Structure, args: list, trailing_coma: bool):
        super().__init__(target.main_token)
        self.target = target
        self.args = args
        self.trailing_coma = trailing_coma

    def to_str(self, level):
        result = self.target.to_str(level) + '['
        if not self.args:
            return result + ']'
        else:
            result += '\n'

        for arg in self.args:
            result += '    ' * (level + 1) + arg.to_str(level + 1) + ',\n'
        if not self.trailing_coma:
            result = result[:-2] + '\n'

        return result + '    ' * level + ']'

    def error_message(self):
        return 'getting item from the resource %s' % self.target.to_str(0)


class Call(Structure):
    def __init__(self, target: Structure, args: list, vararg: list, params: list, lazy: bool):
        super().__init__(target.main_token)
        self.target = target
        self.args = args
        self.varargs = vararg
        self.params = params
        self.lazy = lazy

    def to_str(self, level):
        target = self.target.to_str(level)
        body = []
        lazy = ''
        if self.lazy:
            lazy = '    ' * (level + 1) + '# lazy\n'

        for vararg, arg in zip(self.varargs, self.args):
            prefix = '    ' * (level + 1)
            if vararg:
                prefix += '*'
            body.append(prefix + arg.to_str(level + 1))

        for param in self.params:
            # TODO: there should be no space before and after `=`
            body.append('    ' * (level + 1) + param.to_str(level + 1))

        body = lazy + ',\n'.join(body)
        if body:
            body = '\n' + body + '\n' + '    ' * level

        return target + '(' + body + ')'

    def error_message(self):
        return 'calling the resource %s' % self.target.to_str(0)


class Literal(Structure):
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def to_str(self, level):
        return self.value.body


class Number(Literal):
    def __init__(self, value, minus):
        super().__init__(value)
        self.minus = minus

    def to_str(self, level):
        result = ''
        if self.minus:
            result = '-'
        return result + super().to_str(0)


class Array(Structure):
    def __init__(self, values: list, main_token):
        super().__init__(main_token)
        self.values = values

    def to_str(self, level):
        body = ', '.join(value.to_str(0) for value in self.values)
        if len(body) > MAX_COLUMNS:
            body = ',\n'.join(self.level(level + 1) + value.to_str(level + 1) for value in self.values)
            body = '\n' + body + '\n' + self.level(level)
        return '[' + body + ']'


class Dictionary(Structure):
    def __init__(self, pairs: list, main_token):
        super().__init__(main_token)
        self.pairs = pairs

    def to_str(self, level):
        body = ', '.join(key.to_str(0) + ': ' + value.to_str(0) for key, value in self.pairs)
        if len(body) > MAX_COLUMNS:
            body = ',\n'.join(self.level(level + 1) + key.to_str(level + 1) + ': ' + value.to_str(level + 1)
                              for key, value in self.pairs)
            body = '\n' + body + '\n' + self.level(level)
        return '{' + body + '}'
