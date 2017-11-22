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
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    DICT_OPEN = auto()
    DICT_CLOSE = auto()

    LAZY = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    ASTERISK = auto()

    BLOCK_OPEN = auto()
    BLOCK_CLOSE = auto()
    LAMBDA_OPEN = auto()
    LAMBDA_CLOSE = auto()


REGEXPS = {
    TokenType.NUMBER: re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'),
    TokenType.IDENTIFIER: re.compile(r'[^\d\W]\w*'),
    TokenType.STRING: re.compile(r'''(\"\"\"|\'\'\'|\"|\')((?<!\\)(\\\\)*\\\1|.)*?\1''', flags=re.DOTALL),
}

RESERVED = {
    'import': TokenType.IMPORT,
    'as': TokenType.AS,
    'from': TokenType.FROM,
}

LAZY = re.compile(r'#\s*lazy')
LITERALS = ('None', 'True', 'False')
SINGLE = {
    ',': TokenType.COMA,
    '*': TokenType.ASTERISK,
    ':': TokenType.COLON,
    '.': TokenType.DOT,
    '=': TokenType.EQUALS,
    '[': TokenType.BRACKET_OPEN,
    ']': TokenType.BRACKET_CLOSE,
    '{': TokenType.DICT_OPEN,
    '}': TokenType.DICT_CLOSE,
    '(': TokenType.LAMBDA_OPEN,
    ')': TokenType.LAMBDA_CLOSE,
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
