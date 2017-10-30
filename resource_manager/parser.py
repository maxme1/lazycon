from .tokenizer import TokenType, tokenize
from .structures import *


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        self.indent_step = 4
        self.inside_json = False

    def module(self):
        module_type = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.COLON)
        name = self.require(TokenType.IDENTIFIER)

        return Module(module_type, name)

    def data(self):
        if self.matches(TokenType.IDENTIFIER):
            # module
            if self.matches(TokenType.COLON, shift=1):
                return self.module()
            else:
                # identifier
                return Resource(self.advance())
        # json
        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.DICT_OPEN):
            return self.object()
        return Value(self.require(TokenType.STRING, TokenType.NUMBER, TokenType.LITERAL))

    def params(self):
        lazy = self.matches(TokenType.LAZY)
        if lazy:
            self.advance()
            self.ignore(TokenType.COMA)

        params = []
        while self.matches(TokenType.IDENTIFIER):
            params.append(self.definition())
            self.ignore(TokenType.COMA)
        return params, lazy

    def expression(self):
        data = self.data()
        if type(data) is Module and self.matches(TokenType.BLOCK_OPEN) and not self.inside_json:
            self.advance()
            data = Partial(data, *self.params())
            self.require(TokenType.BLOCK_CLOSE)
            return data

        while self.matches(TokenType.GETATTR, TokenType.LAMBDA_OPEN):
            if self.matches(TokenType.GETATTR):
                self.advance()
                name = self.require(TokenType.IDENTIFIER)
                data = GetAttribute(data, name)
                if self.matches(TokenType.BLOCK_OPEN) and not self.inside_json:
                    self.advance()
                    data = Partial(data, *self.params())
                    self.require(TokenType.BLOCK_CLOSE)
                    return data
            else:
                inside_json = self.inside_json
                self.inside_json = True
                self.advance()
                data = Partial(data, *self.params())
                self.require(TokenType.LAMBDA_CLOSE)
                self.inside_json = inside_json

        return data

    def definition(self):
        left = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        right = self.expression()

        return Definition(left, right)

    def object(self):
        inside_json = self.inside_json
        self.inside_json = True
        self.require(TokenType.DICT_OPEN)
        pairs = {}
        while not self.matches(TokenType.DICT_CLOSE):
            key = self.require(TokenType.STRING)
            self.require(TokenType.COLON)
            pairs[key] = self.expression()

            self.ignore(TokenType.COMA)

        self.require(TokenType.DICT_CLOSE)
        self.inside_json = inside_json
        return Dictionary(pairs)

    def array(self):
        inside_json = self.inside_json
        self.inside_json = True
        self.require(TokenType.BRACKET_OPEN)
        values = []
        while not self.matches(TokenType.BRACKET_CLOSE):
            values.append(self.expression())
            self.ignore(TokenType.COMA)

        self.require(TokenType.BRACKET_CLOSE)
        self.inside_json = inside_json
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

    try:
        tokens = tokenize(source, 4)
        for token in tokens:
            token.set_source(source_path)
        return Parser(tokens).parse()
    except SyntaxError as e:
        raise SyntaxError('{} in file {}'.format(e.msg, source_path)) from None
