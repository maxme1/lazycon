import re
from enum import Enum, auto
import token


class TokenType(Enum):
    STRING = token.STRING
    NUMBER = token.NUMBER
    IDENTIFIER = token.NAME

    COLON = token.COLON
    COMA = token.COMMA
    EQUALS = token.EQUAL
    DOT = token.DOT
    ASTERISK = token.STAR
    BRACKET_OPEN = token.LSQB
    BRACKET_CLOSE = token.RSQB
    DICT_OPEN = token.LBRACE
    DICT_CLOSE = token.RBRACE
    PAR_OPEN = token.LPAR
    PAR_CLOSE = token.RPAR

    # names
    IMPORT = -1
    LAMBDA = -2
    FROM = -3
    AS = -4
    LITERAL = -5
    LAZY = -6


RESERVED = {
    'import': TokenType.IMPORT,
    'as': TokenType.AS,
    'from': TokenType.FROM,
    'lambda': TokenType.LAMBDA,
    'None': TokenType.LITERAL,
    'True': TokenType.LITERAL,
    'False': TokenType.LITERAL
}

LAZY = re.compile(r'^#\s*lazy\s*$')


class TokenWrapper:
    def __init__(self, token, source, token_type):
        self.source = source
        self.token = token
        self.body = token.string
        if token_type is not None:
            self.exact_type = token_type.value

    def __getattr__(self, item):
        return getattr(self.token, item)
