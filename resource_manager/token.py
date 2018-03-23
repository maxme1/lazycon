import re
import token
from enum import Enum, unique
from tokenize import TokenInfo


@unique
class TokenType(Enum):
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
    IMPORT, LAMBDA, FROM, AS, LITERAL, LAZY = range(-6, 0)


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
EXCLUDE = {'NEWLINE', 'NL', 'INDENT', 'DEDENT', 'ENDMARKER', 'ENCODING', 'BACKQUOTE'}


class TokenWrapper:
    def __init__(self, token: TokenInfo, source, token_type):
        self._token = token
        self.body = token.string
        if token_type is not None:
            self.exact_type = token_type.value
        else:
            self.exact_type = token.exact_type
        self.line, self.column = token.start
        self.column += 1
        self.source = source
