import re
import token
from enum import Enum, unique
from tokenize import TokenInfo, _all_string_prefixes


@unique
class TokenType(Enum):
    STRING = token.STRING
    NUMBER = token.NUMBER
    IDENTIFIER = token.NAME
    ELLIPSIS = token.ELLIPSIS

    COLON = token.COLON
    COMA = token.COMMA
    EQUAL = token.EQUAL
    DOT = token.DOT
    ASTERISK = token.STAR
    DOUBLE_ASTERISK = token.DOUBLESTAR
    BRACKET_OPEN = token.LSQB
    BRACKET_CLOSE = token.RSQB
    DICT_OPEN = token.LBRACE
    DICT_CLOSE = token.RBRACE
    PAR_OPEN = token.LPAR
    PAR_CLOSE = token.RPAR

    BIT_OR = token.VBAR
    BIT_XOR = token.CIRCUMFLEX
    BIT_AND = token.AMPER
    SHIFT_LEFT = token.LEFTSHIFT
    SHIFT_RIGHT = token.RIGHTSHIFT

    PLUS = token.PLUS
    MINUS = token.MINUS
    DIVIDE = token.SLASH
    FLOOR_DIVIDE = token.DOUBLESLASH
    MOD = token.PERCENT
    MATMUL = token.AT
    TILDE = token.TILDE

    LESS = token.LESS
    LESS_EQUAL = token.LESSEQUAL
    GREATER = token.GREATER
    GREATER_EQUAL = token.GREATEREQUAL
    IS_EQUAL = token.EQEQUAL
    NOT_EQUAL = token.NOTEQUAL

    # names
    IF, ELSE, IS, IN, AND, OR, NOT, IMPORT, RETURN, DEF, LAMBDA, FROM, AS, LITERAL, PARTIAL, *_ = range(-100, 0)


RESERVED = {
    'import': TokenType.IMPORT,
    'as': TokenType.AS,
    'from': TokenType.FROM,
    'lambda': TokenType.LAMBDA,
    'def': TokenType.DEF,
    'return': TokenType.RETURN,
    'None': TokenType.LITERAL,
    'True': TokenType.LITERAL,
    'False': TokenType.LITERAL,
    'not': TokenType.NOT,
    'in': TokenType.IN,
    'is': TokenType.IS,
    'or': TokenType.OR,
    'and': TokenType.AND,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    '...': TokenType.ELLIPSIS,
}

PARTIAL = re.compile(r'^#\s*(lazy|partial)\s*$')
EXCLUDE = {'NEWLINE', 'NL', 'INDENT', 'DEDENT', 'ENDMARKER', 'ENCODING', 'BACKQUOTE'}
INVALID_STRING_PREFIXES = tuple(x for x in _all_string_prefixes() if 'f' in x.lower())


class TokenWrapper:
    def __init__(self, token: TokenInfo, source_path, token_type):
        self._token = token
        self.body = token.string
        self.type = token_type
        self.line, self.column = token.start
        self.column += 1
        self.source = source_path
        self.token_line = token.line
