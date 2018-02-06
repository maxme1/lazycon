from typing import List

from .structures import Structure
from .token import Token


class Lambda(Structure):
    def __init__(self, params: List[Token], expression: Structure, main_token):
        super().__init__(main_token)
        self.expression = expression
        self.params = params

    def to_str(self, level):
        return 'lambda ' + ','.join(x.body for x in self.params) + ': ' + self.expression.to_str(level + 1)


class Resource(Structure):
    def __init__(self, name: Token):
        super().__init__(name)
        self.name = name

    def to_str(self, level):
        return self.name.body


class GetAttribute(Structure):
    def __init__(self, target: Structure, name: Token):
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
            body.append('    ' * (level + 1) + param.to_str(level + 1))

        body = lazy + ',\n'.join(body)
        if body:
            body = '\n' + body + '\n' + '    ' * level

        return target + '(' + body + ')'

    def error_message(self):
        return 'calling the resource %s' % self.target.to_str(0)


class Module(Structure):
    def __init__(self, module_type, module_name):
        super().__init__(module_type)
        self.module_type = module_type
        self.module_name = module_name

    def to_str(self, level):
        return '{}:{}'.format(self.module_type.body, self.module_name.body)

    def error_message(self):
        return 'looking for the module %s' % self.to_str(0)


class Literal(Structure):
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def to_str(self, level):
        return self.value.body


class Array(Structure):
    def __init__(self, values: list, main_token):
        super().__init__(main_token)
        self.values = values

    def to_str(self, level):
        result = '[\n'
        for value in self.values:
            result += '    ' * (level + 1) + value.to_str(level + 1) + ',\n'
        return result + '    ' * level + ']'


class Dictionary(Structure):
    def __init__(self, pairs: list, main_token):
        super().__init__(main_token)
        self.pairs = pairs

    def to_str(self, level):
        result = '{\n'
        for key, value in self.pairs:
            result += '    ' * (level + 1) + '{}: {},\n'.format(key.to_str(level + 1), value.to_str(level + 1))
        return result[:-1] + '    ' * level + '\n}'
