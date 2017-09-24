class Definition:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f'{self.left}: {self.right}'


class Module:
    def __init__(self, module_type, module_name, params):
        self.module_type = module_type
        self.module_name = module_name
        self.params = params

    def __repr__(self):
        result = '{' \
                 f'"type": {self.module_type}, ' \
                 f'"name": {self.module_name}, ' \
                 f'"params": '

        add = ''
        for key, value in self.params:
            if type(value) is not Module:
                add += f'{key}: {value},'
        result += '{' + add[:-1] + '}, "input": '

        add = ''
        for key, value in self.params:
            if type(value) is Module:
                add += f'{key}: {value},'
        result += '{' + add[:-1] + '}}'

        return result


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
