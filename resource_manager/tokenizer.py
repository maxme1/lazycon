from .token import *


def tokenize(source: str):
    tokens = []
    for line_number, line in enumerate(source.splitlines(), 1):
        line = line.rstrip()
        text = line.lstrip()

        while text:
            position = len(line) - len(text) + 1

            # comment or lazy
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
    for tokenType, regex in REGEXPS.items():
        match = regex.match(text)
        if match:
            match = match.group()
            if match in RESERVED:
                return Token(match, RESERVED[match])
            if match in LITERALS:
                return Token(match, TokenType.LITERAL)
            return Token(match, tokenType)

    for char, tokeType in SINGLE.items():
        if text.startswith(char):
            return Token(char, tokeType)
