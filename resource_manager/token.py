import re
from enum import Enum, auto
from tokenize import Number, String

# TODO: move to Python's tokenizer

class TokenType(Enum):
    STRING = auto()
    NUMBER = auto()
    LITERAL = auto()
    IDENTIFIER = auto()
    LAMBDA = auto()

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

    PAR_OPEN = auto()
    PAR_CLOSE = auto()


REGEXPS = {
    # TODO: we need the minus, because there is no `-` operator
    TokenType.NUMBER: re.compile('-?' + Number),
    TokenType.STRING: re.compile(String, flags=re.UNICODE),
    TokenType.IDENTIFIER: re.compile(r'[^\d\W]\w*'),
}

RESERVED = {
    'import': TokenType.IMPORT,
    'as': TokenType.AS,
    'from': TokenType.FROM,
    'lambda': TokenType.LAMBDA,
}

LAZY = re.compile(r'^#\s*lazy\s*$')
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
    '(': TokenType.PAR_OPEN,
    ')': TokenType.PAR_CLOSE,
}


class Token:
    def __init__(self, body, token_type, line=None):
        self.body = body
        self.type = token_type
        self.line = line
        self.column = None
        self.source = None

    def add_info(self, line, column):
        self.line, self.column = line, column

    def set_source(self, source):
        self.source = source
