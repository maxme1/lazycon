import json

from .tokenizer import tokenize, TokenType
from .structures import *


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        self.indent_step = 4

    def module(self, indentation):
        module_type = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.DOT)
        name = self.require(TokenType.IDENTIFIER)

        if indentation is None:
            indentation = module_type.column
        indentation = indentation + self.indent_step
        params = []
        while self.matches(TokenType.IDENTIFIER) and self.matches(TokenType.COLON, shift=1) \
                and self.indent(indentation):
            left = self.advance()
            self.require(TokenType.COLON)
            right = self.allowed_type(indentation)
            params.append((left, right))

        return Module(module_type, name, params)

    def allowed_type(self, indentation=None):
        if self.matches(TokenType.IDENTIFIER):
            # module
            if self.matches(TokenType.DOT, shift=1):
                return self.module(indentation)
            else:
                # identifier
                return self.advance()
        return self.json_type()

    def definition(self):
        left = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        right = self.allowed_type(left.column)

        return Definition(left, right)

    def json_type(self):
        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.CBRACKET_OPEN):
            return self.object()
        return Value(self.require(TokenType.STRING, TokenType.NUMBER, TokenType.LITERAL))

    def object(self):
        self.require(TokenType.CBRACKET_OPEN)
        pairs = {}
        while not self.matches(TokenType.CBRACKET_CLOSE):
            key = self.require(TokenType.STRING)
            self.require(TokenType.COLON)
            value = self.allowed_type()
            pairs[key] = value
            # ignore trailing coma
            if self.matches(TokenType.COMA):
                self.advance()

        self.require(TokenType.CBRACKET_CLOSE)
        return Dictionary(pairs)

    def array(self):
        self.require(TokenType.BRACKET_OPEN)
        values = []
        while not self.matches(TokenType.BRACKET_CLOSE):
            values.append(self.allowed_type())
            # ignore trailing coma
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
        result = self.tokens[self.position]
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

        for type in types:
            if temp.type == type:
                return True

        return False

    def indent(self, indent, strict=False):
        if self.current.column == indent:
            return True
        if strict:
            raise SyntaxError('Indentation error')
        return False

    def require(self, *types):
        if not self.matches(*types):
            raise ValueError(f'Unexpected token: {repr(self.current)} at {self.current.line}:{self.current.column}')

        return self.advance()


def transpile(source):
    tokens = tokenize(source)
    return Parser(tokens).compile()
