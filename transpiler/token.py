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
    CBRACKET_OPEN = auto()
    CBRACKET_CLOSE = auto()


REGEXPS = {TokenType.NUMBER: re.compile(r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'),
           TokenType.IDENTIFIER: re.compile(r'[^\d\W]\w*'),
           TokenType.STRING: re.compile(r'"(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\0-\x1F\x7F]+)*"')}

LITERALS = ('null', 'true', 'false')
SINGLE = {
    ',': TokenType.COMA,
    ':': TokenType.COLON,
    '.': TokenType.DOT,
    '=': TokenType.EQUALS,
    '[': TokenType.BRACKET_OPEN,
    ']': TokenType.BRACKET_CLOSE,
    '{': TokenType.CBRACKET_OPEN,
    '}': TokenType.CBRACKET_CLOSE,
}


class Token:
    def __init__(self, body, type):
        self.body = body
        self.type = type
        self.line = self.column = None

    def add_info(self, line, column):
        self.line, self.column = line, column

    def __repr__(self):
        return str(self.body)

    def __str__(self):
        if self.type == TokenType.IDENTIFIER:
            return f'"{self.body}"'
        return self.body
