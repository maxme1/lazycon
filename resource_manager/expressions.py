from typing import List, Union

from .structures import Structure, MAX_COLUMNS
from .token import TokenWrapper


class Lambda(Structure):
    def __init__(self, params: List[TokenWrapper], last_vararg: bool, expression: Structure, main_token):
        super().__init__(main_token)
        self.expression = expression

        if last_vararg:
            assert params
            params, vararg = params[:-1], params[-1]
        else:
            vararg = None

        self.params = tuple(params)
        self.vararg = vararg

    def to_str(self, level):
        params = ','.join(x.body for x in self.params)
        if self.vararg:
            if params:
                params += ', '
            params += '*' + self.vararg.body

        if params:
            params = ' ' + params
        return 'lambda' + params + ': ' + self.expression.to_str(level + 1)


class InlineIf(Structure):
    def __init__(self, condition: Structure, left: Structure, right: Structure, main_token: TokenWrapper):
        super().__init__(main_token)
        self.right = right
        self.left = left
        self.condition = condition

    def to_str(self, level):
        return '%s if %s else %s' % (self.left.to_str(level), self.condition.to_str(level), self.right.to_str(level))


class Binary(Structure):
    def __init__(self, left: Structure, right: Structure, operation: Union[TokenWrapper, tuple]):
        super().__init__(operation)
        self.operation = operation
        self.left = left
        self.right = right
        if type(operation) is tuple:
            self.key = tuple(x.type for x in operation)
        else:
            self.key = operation.type

    def to_str(self, level):
        if type(self.operation) is tuple:
            operation = ' '.join(x.body for x in self.operation)
        else:
            operation = self.operation.body
        return '%s %s %s' % (self.left.to_str(level), operation, self.right.to_str(0))


class Unary(Structure):
    def __init__(self, argument: Structure, operation: TokenWrapper):
        super().__init__(operation)
        self.operation = operation
        self.argument = argument
        self.key = operation.type

    def to_str(self, level):
        return '%s %s' % (self.operation.body, self.argument.to_str(0))


class Resource(Structure):
    def __init__(self, name: TokenWrapper):
        super().__init__(name)
        self.name = name

    def to_str(self, level):
        return self.name.body


class Parenthesis(Structure):
    def __init__(self, expression: Structure):
        super().__init__(expression.main_token)
        self.expression = expression

    def to_str(self, level):
        return '(' + self.expression.to_str(level + 1) + ')'


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


class Slice(Structure):
    def __init__(self, start: Structure, stop: Structure, step: Structure, main_token: TokenWrapper):
        super().__init__(main_token)
        self.step = step
        self.stop = stop
        self.start = start
        self.args = [start, stop, step]

    def to_str(self, level):
        result = ''
        if self.start is not None:
            result += self.start.to_str(level)
        result += ':'
        if self.stop:
            result += self.stop.to_str(level)
        if self.step:
            result += ':' + self.step.to_str(level)

        return result


class Call(Structure):
    def __init__(self, target: Structure, args: list, vararg: list, params: list, lazy: bool, main_token):
        super().__init__(main_token)
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
            param_str = param.name.body, param.value.to_str(level + 1)
            body.append('    ' * (level + 1) + '%s=%s' % param_str)

        body = lazy + ',\n'.join(body)
        if body:
            body = '\n' + body + '\n' + '    ' * level

        return target + '(' + body + ')'

    def error_message(self):
        return 'calling the resource %s' % self.target.to_str(0)


class Literal(Structure):
    def __init__(self, value: TokenWrapper):
        super().__init__(value)
        self.value = value

    def to_str(self, level):
        return self.value.body


class Starred(Structure):
    def __init__(self, expression: Structure, main_token: TokenWrapper):
        super().__init__(main_token)
        self.expression = expression

    def to_str(self, level):
        return '*' + self.expression.to_str(level)


# TODO: unify inlines

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


class Tuple(Structure):
    def __init__(self, values: list, main_token):
        super().__init__(main_token)
        self.values = values

    def to_str(self, level):
        body = ', '.join(value.to_str(0) for value in self.values)
        if len(self.values) == 1:
            body += ','
        if len(body) > MAX_COLUMNS:
            body = ',\n'.join(self.level(level + 1) + value.to_str(level + 1) for value in self.values)
            if len(self.values) == 1:
                body += ','
            body = '\n' + body + '\n' + self.level(level)
        return '(' + body + ')'


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
