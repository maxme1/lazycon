from enum import Enum, auto
import re


class TokenType(Enum):
    STRING = auto()
    NUMBER = auto()
    LITERAL = auto()
    IDENTIFIER = auto()

    COLON = auto()
    COMA = auto()
    EQUALS = auto()
    DOT = auto()
    GETATTR = auto()
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    DICT_OPEN = auto()
    DICT_CLOSE = auto()

    DIRECTIVE = auto()
    LAZY = auto()
    EXTENDS = auto()
    FROM = auto()

    BLOCK_OPEN = auto()
    BLOCK_CLOSE = auto()
    LAMBDA_OPEN = auto()
    LAMBDA_CLOSE = auto()


REGEXPS = {
    TokenType.NUMBER: re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'),
    TokenType.IDENTIFIER: re.compile(r'[^\d\W]\w*'),
    TokenType.STRING: re.compile(r'"(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\0-\x1F\x7F]+)*"'),
}

RESERVED = {
    'lazy': TokenType.LAZY,
    'extends': TokenType.EXTENDS,
    'from': TokenType.FROM,
}

LITERALS = ('null', 'true', 'false')
SINGLE = {
    ',': TokenType.COMA,
    ':': TokenType.COLON,
    '.': TokenType.GETATTR,
    '=': TokenType.EQUALS,
    '[': TokenType.BRACKET_OPEN,
    ']': TokenType.BRACKET_CLOSE,
    '{': TokenType.DICT_OPEN,
    '}': TokenType.DICT_CLOSE,
    '(': TokenType.LAMBDA_OPEN,
    ')': TokenType.LAMBDA_CLOSE,
    '@': TokenType.DIRECTIVE,
}

JSON_OPEN = [TokenType.BRACKET_OPEN, TokenType.DICT_OPEN, TokenType.LAMBDA_OPEN]
JSON_CLOSE = {TokenType.BRACKET_CLOSE: TokenType.BRACKET_OPEN, TokenType.DICT_CLOSE: TokenType.DICT_OPEN,
              TokenType.LAMBDA_CLOSE: TokenType.LAMBDA_OPEN}


class Token:
    def __init__(self, body, type, line=None):
        self.body = body
        self.type = type
        self.line = line
        self.column = None
        self.source = None

    def add_info(self, line, column):
        self.line, self.column = line, column

    def set_source(self, source):
        self.source = source

    def __repr__(self):
        return '{}:{}'.format(self.type, self.body)
