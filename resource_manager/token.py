import re
import token
from enum import Enum, auto, unique


@unique
class TokenType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        # for a proper auto()
        return -len([x for x in last_values if x < 0]) - 1

    STRING = token.STRING
    NUMBER = token.NUMBER
    IDENTIFIER = token.NAME
    MINUS = token.MINUS

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
    IMPORT = auto()
    LAMBDA = auto()
    FROM = auto()
    AS = auto()
    LITERAL = auto()
    LAZY = auto()


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
