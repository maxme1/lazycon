from resource_manager.token import Token


class Structure:
    def position(self):
        return self.main_token.line, self.main_token.column, self.main_token.source

    def to_str(self, level):
        pass


class Definition(Structure):
    def __init__(self, name: Token, value: Structure):
        self.name = name
        self.value = value
        self.main_token = name

    def to_str(self, level):
        return '{} = {}'.format(self.name.body, self.value.to_str(level))


class Resource(Structure):
    def __init__(self, name: Token):
        self.name = name
        self.main_token = name

    def to_str(self, level):
        return self.name.body


class GetAttribute(Structure):
    def __init__(self, data: Structure, name: Token):
        self.data = data
        self.name = name
        self.main_token = name

    def to_str(self, level):
        return '{}.{}'.format(self.data.to_str(level), self.name.body)


class Partial(Structure):
    def __init__(self, target: Structure, params: list, lazy: bool):
        self.target = target
        self.params = params
        self.lazy = lazy

    def position(self):
        return self.target.position()

    def to_str(self, level):
        result = '{}(\n'.format(self.target.to_str(level))
        if self.lazy:
            result += '    ' * (level + 1) + '@lazy\n'

        for param in self.params:
            result += '    ' * (level + 1) + param.to_str(level + 1) + '\n'

        return result + '    ' * level + ')'


class Module(Structure):
    def __init__(self, module_type, module_name):
        self.module_type = module_type
        self.module_name = module_name
        self.main_token = module_type

    def to_str(self, level):
        return '{}:{}'.format(self.module_type.body, self.module_name.body)


class Value(Structure):
    def __init__(self, value):
        self.main_token = self.value = value

    def to_str(self, level):
        return self.value.body


class Array(Structure):
    def __init__(self, values: list):
        self.values = values

    def position(self):
        if self.values:
            return self.values[0].position()

    def to_str(self, level):
        result = '[\n'
        for value in self.values:
            result += '    ' * (level + 1) + value.to_str(level + 1) + '\n'
        return result + '    ' * level + ']'


class Dictionary(Structure):
    def __init__(self, dictionary: dict):
        self.dictionary = dictionary

    def position(self):
        if self.dictionary:
            return next(iter(self.dictionary.values())).position()
        # else:
        #     return

    def to_str(self, level):
        result = '{\n'
        for key, value in self.dictionary.items():
            result += '    ' * (level + 1) + '{}: {}\n'.format(key.body, value.to_str(level + 1))
        return result[:-1] + '    ' * level + '\n}'
