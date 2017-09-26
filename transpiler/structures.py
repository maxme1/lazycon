from .token import Token, TokenType


class Definition:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f'{self.left}: {self.right}'


def is_input(value):
    if type(value) is Array:
        for val in value.values:
            if is_input(val):
                return True
        return False

    if type(value) is Dictionary:
        for val in value.dictionary.values():
            if is_input(val):
                return True
        return False

    if type(value) is Module:
        return True

    return type(value) is Token and value.type == TokenType.IDENTIFIER


class Module:
    def __init__(self, module_type, module_name, params, init):
        self.module_type = module_type
        self.module_name = module_name
        self.params = params
        self.init = init

    def __repr__(self):
        result = '{' \
                 f'"type": {self.module_type},' \
                 f'"name": {self.module_name},'

        if self.init is not None:
            result += f'"init": {self.init},'

        add = ''
        for param in self.params:
            key, value = param.left, param.right
            if not is_input(value):
                add += f'{key}: {value},'
        if add:
            result += '"params": {' + add[:-1] + '},'

        add = ''
        for param in self.params:
            key, value = param.left, param.right
            if is_input(value):
                add += f'{key}: {value},'
        if add:
            result += '"inputs": {' + add[:-1] + '},'

        return result[:-1] + '}'


class Value:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value.body


class Array:
    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return repr(self.values)


class Dictionary:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __repr__(self):
        result = ''
        for key, value in self.dictionary.items():
            result += f'{key}: {value},'
        return '{' + result[:-1] + '}'
