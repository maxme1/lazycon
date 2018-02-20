from tokenize import tokenize as _tokenize, TokenError
from token import tok_name
from io import BytesIO

from .token import LAZY, TokenWrapper, RESERVED, TokenType, EXCLUDE


def tokenize(source: str, source_path: str):
    tokens = []
    for token in list(_tokenize(BytesIO(source.encode('utf-8')).readline)):
        name = tok_name[token.type]
        if name == 'ERRORTOKEN':
            raise TokenError('Unrecognized token starting', token.start)
        if name in EXCLUDE:
            continue
        if name == 'COMMENT':
            if not LAZY.match(token.string.strip()):
                continue
            token_type = TokenType.LAZY
        else:
            token_type = RESERVED.get(token.string)
        tokens.append(TokenWrapper(token, source_path, token_type))

    return tokens
