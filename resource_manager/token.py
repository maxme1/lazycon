import re
import token
from enum import Enum, unique
from tokenize import TokenInfo, _all_string_prefixes
import operator


@unique
class TokenType(Enum):
    STRING = token.STRING
    NUMBER = token.NUMBER
    IDENTIFIER = token.NAME

    COLON = token.COLON
    COMA = token.COMMA
    EQUALS = token.EQUAL
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
    IF, ELSE, IS, IN, AND, OR, NOT, IMPORT, LAMBDA, FROM, AS, LITERAL, LAZY, *_ = range(-100, 0)


RESERVED = {
    'import': TokenType.IMPORT,
    'as': TokenType.AS,
    'from': TokenType.FROM,
    'lambda': TokenType.LAMBDA,
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
}

LAZY = re.compile(r'^#\s*lazy\s*$')
EXCLUDE = {'NEWLINE', 'NL', 'INDENT', 'DEDENT', 'ENDMARKER', 'ENCODING', 'BACKQUOTE'}
INVALID_STRING_PREFIXES = tuple(x for x in _all_string_prefixes() if 'f' in x.lower())

UNARY_OPERATORS = {
    TokenType.NOT: operator.not_,
    TokenType.TILDE: operator.invert,
    TokenType.PLUS: operator.pos,
    TokenType.MINUS: operator.neg,
}

BINARY_OPERATORS = {
    TokenType.BIT_AND: operator.and_,
    TokenType.BIT_OR: operator.or_,
    TokenType.BIT_XOR: operator.xor,

    TokenType.DIVIDE: operator.truediv,
    TokenType.FLOOR_DIVIDE: operator.floordiv,
    TokenType.ASTERISK: operator.mul,
    TokenType.DOUBLE_ASTERISK: operator.pow,
    TokenType.MATMUL: operator.matmul,
    TokenType.MOD: operator.mod,
    TokenType.PLUS: operator.add,
    TokenType.MINUS: operator.sub,

    TokenType.LESS: operator.lt,
    TokenType.GREATER: operator.gt,
    TokenType.LESS_EQUAL: operator.le,
    TokenType.GREATER_EQUAL: operator.ge,
    TokenType.IS_EQUAL: operator.eq,
    TokenType.NOT_EQUAL: operator.ne,
    TokenType.IS: operator.is_,
    TokenType.IN: lambda x, y: x in y,
    (TokenType.NOT, TokenType.IN): lambda x, y: x not in y,
    (TokenType.IS, TokenType.NOT): operator.is_not,
}


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
        self.token_line = token.line
        self.type = TokenType(self.exact_type)
