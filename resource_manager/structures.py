from typing import List

from resource_manager.token import Token


class Structure:
    def __init__(self, main_token: Token):
        self.main_token = main_token

    def position(self):
        return self.main_token.line, self.main_token.column, self.main_token.source

    def source(self):
        return self.main_token.source

    def to_str(self, level):
        pass

    def error_message(self):
        return 'building the module ' + self.to_str(0)


class ImportPython(Structure):
    def __init__(self, root: List[Token], values: dict, main_token):
        super().__init__(main_token)
        self._root = root
        self.root = '.'.join(x.body for x in root)
        self._values = values
        self.values = {'.'.join(x.body for x in value): name for value, name in values.items()}

    def to_str(self, level):
        result = ''
        if self.root:
            result += 'from %s ' % self.root
        result += 'import '
        for value, name in self.values.items():
            result += value + ' '
            if name is not None:
                result += 'as ' + name.body
            result += ', '
        return result[:-2] + '\n'


class LazyImport(Structure):
    def __init__(self, from_, what, as_, main_token: Token):
        super().__init__(main_token)
        self.from_, self.what, self.as_ = from_, what, as_

    def to_str(self, level):
        result = ''
        if self.from_:
            result += 'from %s ' % self.from_
        result += 'import %s' % self.what
        if self.as_:
            result += ' as %s' % self.as_.body
        return result + '\n'

    def error_message(self):
        result = 'importing '
        if self.from_:
            result += self.from_ + '.'
        return result + self.what


class Definition(Structure):
    def __init__(self, name: Token, value: Structure):
        super().__init__(name)
        self.name = name
        self.value = value

    def to_str(self, level):
        return '{} = {}'.format(self.name.body, self.value.to_str(level))


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
            body = '\n' + body + '    ' * level + '\n'

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
    def __init__(self, dictionary: dict, main_token):
        super().__init__(main_token)
        self.dictionary = dictionary

    def to_str(self, level):
        result = '{\n'
        for key, value in self.dictionary.items():
            result += '    ' * (level + 1) + '{}: {},\n'.format(key.body, value.to_str(level + 1))
        return result[:-1] + '    ' * level + '\n}'
