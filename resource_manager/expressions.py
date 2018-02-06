import functools
from typing import List

from .structures import Structure
from .token import Token


class Lambda(Structure):
    def __init__(self, params: List[Token], expression: Structure, main_token):
        super().__init__(main_token)
        self.expression = expression
        self.params = params

    def render(self, interpreter):
        def f(*args):
            # TODO: checks
            interpreter._scopes.append({x.body: y for x, y in zip(self.params, args)})
            result = interpreter._define_resource(self.expression)
            interpreter._scopes.pop()
            return result

        return f

    def to_str(self, level):
        return 'lambda ' + ','.join(x.body for x in self.params) + ': ' + self.expression.to_str(level + 1)


class Resource(Structure):
    def __init__(self, name: Token):
        super().__init__(name)
        self.name = name

    def render(self, interpreter):
        name = self.name.body
        for scope in reversed(interpreter._scopes):
            if name in scope:
                return scope[name]
        return interpreter._get_resource(name)

    def to_str(self, level):
        return self.name.body


class GetAttribute(Structure):
    def __init__(self, target: Structure, name: Token):
        super().__init__(name)
        self.target = target
        self.name = name

    def render(self, interpreter):
        data = interpreter._define_resource(self.target)
        return getattr(data, self.name.body)

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

    def render(self, interpreter):
        target = interpreter._define_resource(self.target)
        args = tuple(interpreter._define_resource(arg) for arg in self.args)
        if not self.trailing_coma and len(args) == 1:
            args = args[0]
        return target[args]

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

    def render(self, interpreter):
        target = interpreter._define_resource(self.target)
        args = []
        for vararg, arg in zip(self.varargs, self.args):
            temp = interpreter._define_resource(arg)
            if vararg:
                args.extend(temp)
            else:
                args.append(temp)
        kwargs = {param.name.body: interpreter._define_resource(param.value) for param in self.params}
        if self.lazy:
            return functools.partial(target, *args, **kwargs)
        else:
            return target(*args, **kwargs)

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

    def render(self, interpreter):
        if interpreter.get_module is None:
            raise ValueError('The function "get_module" was not provided, so your modules are unreachable')
        return interpreter.get_module(self.module_type.body, self.module_name.body)

    def to_str(self, level):
        return '{}:{}'.format(self.module_type.body, self.module_name.body)

    def error_message(self):
        return 'looking for the module %s' % self.to_str(0)


class Literal(Structure):
    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def render(self, interpreter):
        return eval(self.value.body)

    def to_str(self, level):
        return self.value.body


class Array(Structure):
    def __init__(self, values: list, main_token):
        super().__init__(main_token)
        self.values = values

    def render(self, interpreter):
        return [interpreter._define_resource(x) for x in self.values]

    def to_str(self, level):
        result = '[\n'
        for value in self.values:
            result += '    ' * (level + 1) + value.to_str(level + 1) + ',\n'
        return result + '    ' * level + ']'


class Dictionary(Structure):
    def __init__(self, pairs: list, main_token):
        super().__init__(main_token)
        self.pairs = pairs

    def render(self, interpreter):
        return {interpreter._define_resource(key): interpreter._define_resource(value) for key, value in self.pairs}

    def to_str(self, level):
        result = '{\n'
        for key, value in self.pairs:
            result += '    ' * (level + 1) + '{}: {},\n'.format(key.to_str(level + 1), value.to_str(level + 1))
        return result[:-1] + '    ' * level + '\n}'
