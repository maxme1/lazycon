from typing import List, Union

from .arguments import Parameter
from .structures import Structure, MAX_COLUMNS
from .statements import Definition
from .token import TokenWrapper


class Function(Structure):
    def __init__(self, arguments: List[Parameter], bindings: List[Definition], expression: Structure, name, main_token):
        super().__init__(main_token)
        self.name = name
        self.bindings = bindings
        self.expression = expression
        self.arguments = arguments
        positional, self.vararg, i = [], None, 0
        for i, argument in enumerate(arguments):
            if argument.vararg:
                self.vararg = argument
                break
            positional.append(argument)

        def get_names(arr):
            return tuple(x.name.body for x in arr)

        self.positional = get_names(positional)
        self.keyword = self.positional + get_names(arguments[i + 1:])

    def draw_params(self, level):
        result = ''
        for argument in self.arguments:
            if argument.vararg:
                result += '*'
                if argument.keyword:
                    result += '*'

            result += argument.name.body
            if argument.has_default_value:
                result += '=' + argument.default_expression.to_str(level)
            result += ', '

        result = result[:-1]
        if self.arguments:
            result = result[:-1]
        return result


class Lambda(Function):
    def __init__(self, arguments: List[Parameter], expression: Structure, main_token):
        super().__init__(arguments, [], expression, '<lambda>', main_token)

    def to_str(self, level):
        return 'lambda ' + self.draw_params(level) + ': ' + self.expression.to_str(level)


class FuncDef(Function):
    def to_str(self, level, name=None):
        result = '\ndef %s(' % (name or self.name) + self.draw_params(level) + '):\n'
        for binding in self.bindings:
            result += binding.to_str(level + 1)
        return result + '    ' * (level + 1) + 'return ' + self.expression.to_str(level + 1) + '\n\n'


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

    def _operation_body(self):
        if type(self.operation) is tuple:
            return ' '.join(x.body for x in self.operation)
        else:
            return self.operation.body

    def to_str(self, level):
        return '%s %s %s' % (self.left.to_str(level), self._operation_body(), self.right.to_str(0))

    def error_message(self):
        return 'applying the "%s" operator' % self._operation_body()


class Unary(Structure):
    def __init__(self, argument: Structure, operation: TokenWrapper):
        super().__init__(operation)
        self.operation = operation
        self.argument = argument
        self.key = operation.type

    def to_str(self, level):
        return '%s %s' % (self.operation.body, self.argument.to_str(0))

    def error_message(self):
        return 'applying the "unary %s" operator' % self.operation.body


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
    def __init__(self, target: Structure, args: tuple, kwargs: tuple, lazy: bool, main_token):
        super().__init__(main_token)
        self.kwargs = kwargs
        self.target = target
        self.args = args
        self.lazy = lazy

    def to_str(self, level):
        target = self.target.to_str(level)
        lazy = ''
        if self.lazy:
            lazy = '    ' * (level + 1) + '# lazy\n'

        body = lazy + ',\n'.join('    ' * (level + 1) + x.to_str(level + 1) for x in self.args + self.kwargs)
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


class InlineContainer(Structure):
    def __init__(self, entries: list, main_token):
        super().__init__(main_token)
        self.entries = entries

    def draw_entry(self, entry, level):
        return entry.to_str(level)

    def draw_body(self, level, separator):
        return separator.join('    ' * level + self.draw_entry(entry, level) for entry in self.entries)

    def to_str(self, level):
        body = self.draw_body(0, ', ')
        if len(body) > MAX_COLUMNS:
            body = self.draw_body(level + 1, ',\n')
            body = '\n' + body + '\n' + '    ' * level
        return self.begin + body + self.end


class Array(InlineContainer):
    begin, end = '[]'


class Set(InlineContainer):
    begin, end = '{}'


class Tuple(InlineContainer):
    begin, end = '()'

    def draw_body(self, level, separator):
        body = super().draw_body(level, separator)
        if len(self.entries) == 1:
            body += ','
        return body


class Dictionary(InlineContainer):
    begin, end = '{}'

    def draw_entry(self, entry, level):
        return entry[0].to_str(level) + ': ' + entry[1].to_str(level)
