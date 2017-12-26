from .token import *


def tokenize(source: str):
    tokens = []
    for line_number, line in enumerate(source.splitlines(), 1):
        line = line.rstrip()
        text = line.lstrip()
        # TODO: ugly
        if not text.strip() or (text.startswith('#') and not LAZY.match(text)):
            continue

        while text:
            position = len(line) - len(text) + 1

            # TODO: combine
            # comment
            if text.startswith('#'):
                text = text.strip()
                match = LAZY.match(text)
                if match:
                    token = Token(match.group(), TokenType.LAZY)
                else:
                    break
            else:
                token = next_token(text)

                if token is None:
                    err = text.split()[0]
                    raise SyntaxError('Unrecognized token: "{}" at {}:{}'.format(err, line_number, position))

            token.add_info(line_number, position)
            tokens.append(token)

            text = text[len(token.body):]
            text = text.strip()
    return tokens


def next_token(text: str):
    for literal in LITERALS:
        if text.startswith(literal):
            return Token(literal, TokenType.LITERAL)

    for tokenType, regex in REGEXPS.items():
        match = regex.match(text)
        if match:
            match = match.group()
            if match in RESERVED:
                return Token(match, RESERVED[match])
            return Token(match, tokenType)

    for char, tokeType in SINGLE.items():
        if text.startswith(char):
            return Token(char, tokeType)
