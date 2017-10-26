class Definition:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_str(self, level):
        return '{} = {}'.format(self.name.body, self.value.to_str(level))


class Resource:
    def __init__(self, name):
        self.name = name

    def to_str(self, level):
        return self.name.body


class GetAttribute:
    def __init__(self, data, name):
        self.data = data
        self.name = name

    def to_str(self, level):
        return '{}.{}'.format(self.data.to_str(level), self.name.body)


class Module:
    def __init__(self, module_type, module_name, params, init):
        self.module_type = module_type
        self.module_name = module_name
        self.params = params
        self.init = init

    def to_str(self, level):
        result = '{}:{}(\n'.format(self.module_type.body, self.module_name.body)

        if self.init is not None:
            result += '    ' * (level + 1) + '@init = {}\n'.format(self.init.to_str(level))

        for param in self.params:
            result += '    ' * (level + 1) + param.to_str(level + 1) + '\n'

        return result + '    ' * level + ')'


class Value:
    def __init__(self, value):
        self.value = value

    def to_str(self, level):
        return self.value.body


class Array:
    def __init__(self, values):
        self.values = values

    def to_str(self, level):
        result = '[\n'
        for value in self.values:
            result += '    ' * (level + 1) + value.to_str(level + 1) + '\n'
        return result + '    ' * level + ']'


class Dictionary:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def to_str(self, level):
        result = '{\n'
        for key, value in self.dictionary.items():
            result += '    ' * (level + 1) + '{}: {}\n'.format(key.body, value.to_str(level + 1))
        return result[:-1] + '    ' * level + '\n}'
