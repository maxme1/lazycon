from .tokenizer import TokenType, tokenize
from .structures import *


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        self.indent_step = 4
        self.json_stack = []

    def params(self):
        params = []
        init = None
        if self.matches(TokenType.INIT):
            init = self.module_init()

        while self.matches(TokenType.IDENTIFIER):
            params.append(self.definition())
            self.ignore(TokenType.COMA)
        return params, init

    def module_init(self):
        self.require(TokenType.INIT)
        self.require(TokenType.EQUALS)
        init = self.require(TokenType.LITERAL)
        if init.body not in ['true', 'false']:
            raise SyntaxError('The `init` parameter can be either `true` or `false`')

        init = Value(init)
        self.ignore(TokenType.COMA)
        return init

    def module(self, json_safe=False):
        module_type = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.COLON)
        name = self.require(TokenType.IDENTIFIER)

        init = None
        params = []
        if self.matches(TokenType.BLOCK_OPEN) and not json_safe:
            self.advance()
            params, init = self.params()
            self.require(TokenType.BLOCK_CLOSE)
        if self.matches(TokenType.LAMBDA_OPEN):
            self.advance()
            params, init = self.params()
            self.require(TokenType.LAMBDA_CLOSE)

        return Module(module_type, name, params, init)

    def allowed_type(self, json_safe=False):
        if self.matches(TokenType.IDENTIFIER):
            # module
            if self.matches(TokenType.COLON, shift=1):
                return self.module(json_safe)
            else:
                # identifier
                return Resource(self.advance())
        # json
        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.DICT_OPEN):
            return self.object()
        return Value(self.require(TokenType.STRING, TokenType.NUMBER, TokenType.LITERAL))

    def get_attribute(self):
        data = self.allowed_type()
        while self.matches(TokenType.GETATTR):
            self.advance()
            name = self.require(TokenType.IDENTIFIER)
            data = GetAttribute(data, name)
        return data

    def definition(self):
        left = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        right = self.get_attribute()

        return Definition(left, right)

    def object(self):
        self.require(TokenType.DICT_OPEN)
        pairs = {}
        while not self.matches(TokenType.DICT_CLOSE):
            key = self.require(TokenType.STRING)
            self.require(TokenType.COLON)
            value = self.allowed_type(json_safe=True)
            pairs[key] = value

            self.ignore(TokenType.COMA)

        self.require(TokenType.DICT_CLOSE)
        return Dictionary(pairs)

    def array(self):
        self.require(TokenType.BRACKET_OPEN)
        values = []
        while not self.matches(TokenType.BRACKET_CLOSE):
            values.append(self.allowed_type(json_safe=True))
            self.ignore(TokenType.COMA)

        self.require(TokenType.BRACKET_CLOSE)
        return Array(values)

    def extends(self):
        self.require(TokenType.EXTENDS)
        block = self.matches(TokenType.LAMBDA_OPEN)
        if block:
            self.advance()

        paths = []
        while self.matches(TokenType.STRING):
            paths.append(self.advance().body[1:-1])
            self.ignore(TokenType.COMA)

        if block:
            self.require(TokenType.LAMBDA_CLOSE)
        return paths

    def parse(self):
        parents = []
        if self.matches(TokenType.EXTENDS):
            parents = self.extends()

        definitions = []
        while self.position < len(self.tokens):
            definitions.append(self.definition())

        return definitions, parents

    def advance(self):
        result = self.current
        self.position += 1
        return result

    @property
    def current(self):
        if self.position >= len(self.tokens):
            raise SyntaxError('Unexpected end of file')
        return self.tokens[self.position]

    def matches(self, *types, shift=0):
        try:
            temp = self.tokens[self.position + shift]
        except IndexError:
            return False

        for tokenType in types:
            if temp.type == tokenType:
                return True

        return False

    def require(self, *types):
        if not self.matches(*types):
            if self.current.type == TokenType.BLOCK_OPEN:
                message = 'Unexpected indent at line %d' % self.current.line
            else:
                message = 'Unexpected token: ' \
                          '"{}" at {}:{}'.format(self.current.body, self.current.line, self.current.column)
            raise SyntaxError(message)

        return self.advance()

    def ignore(self, *types):
        if self.matches(*types):
            self.advance()


def parse_file(source_path):
    with open(source_path) as file:
        source = file.read()

    tokens = tokenize(source, 4)
    try:
        return Parser(tokens).parse()
    except SyntaxError as e:
        raise SyntaxError('Error while parsing file '.format(source_path)) from e
