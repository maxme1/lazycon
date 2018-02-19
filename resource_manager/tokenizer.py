from tokenize import tokenize as _tokenize
from token import tok_name

from io import BytesIO

from .token import LAZY, TokenWrapper, RESERVED, TokenType


def tokenize(source: str, source_path: str):
    tokens = []

    # TODO: dirty
    for token in list(_tokenize(BytesIO(source.encode('utf-8')).readline))[1:-1]:
        if tok_name[token.type] in ['NEWLINE', 'NL', 'INDENT', 'DEDENT']:
            continue
        if tok_name[token.type] == 'COMMENT':
            if not LAZY.match(token.string.strip()):
                continue
            token_type = TokenType.LAZY
        else:
            token_type = RESERVED.get(token.string)
        tokens.append(TokenWrapper(token, source_path, token_type))

    return tokens
