import json

from .tokenizer import tokenize, TokenType
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
            if self.matches(TokenType.COMA):
                self.advance()
        return params, init

    def module_init(self):
        self.require(TokenType.INIT)
        self.require(TokenType.EQUALS)
        init = self.require(TokenType.LITERAL)
        if init.body not in ['true', 'false']:
            raise ValueError('The `init` parameter can be either `true` or `false`')

        init = Value(init)
        if self.matches(TokenType.COMA):
            self.advance()
        return init

    def module(self, json_safe=False):
        module_type = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.DOT)
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
            if self.matches(TokenType.DOT, shift=1):
                return self.module(json_safe)
            else:
                # identifier
                return self.advance()
        # json
        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.DICT_OPEN):
            return self.object()
        return Value(self.require(TokenType.STRING, TokenType.NUMBER, TokenType.LITERAL))

    def definition(self):
        left = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        right = self.allowed_type()

        return Definition(left, right)

    def object(self):
        self.require(TokenType.DICT_OPEN)
        pairs = {}
        while not self.matches(TokenType.DICT_CLOSE):
            key = self.require(TokenType.STRING)
            self.require(TokenType.COLON)
            value = self.allowed_type(json_safe=True)
            pairs[key] = value
            # ignore coma
            if self.matches(TokenType.COMA):
                self.advance()

        self.require(TokenType.DICT_CLOSE)
        return Dictionary(pairs)

    def array(self):
        self.require(TokenType.BRACKET_OPEN)
        values = []
        while not self.matches(TokenType.BRACKET_CLOSE):
            values.append(self.allowed_type(json_safe=True))
            # ignore coma
            if self.matches(TokenType.COMA):
                self.advance()

        self.require(TokenType.BRACKET_CLOSE)
        return Array(values)

    def compile(self):
        result = ''
        while self.position < len(self.tokens):
            result += repr(self.definition()) + ','

        # final check
        final = json.loads('{' + result[:-1] + '}')
        return json.dumps(final, indent=2)

    def advance(self):
        result = self.current
        self.position += 1
        return result

    @property
    def current(self):
        if self.position >= len(self.tokens):
            raise SyntaxError(f'Unexpected end of file')
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
                message = f'Unexpected indent at line {self.current.line}'
            else:
                message = f'Unexpected token: {repr(self.current.body)} at {self.current.line}:{self.current.column}'
            raise ValueError(message)

        return self.advance()


def transpile(source: str, indentation: int = 4):
    tokens = tokenize(source, indentation)
    return Parser(tokens).compile()
