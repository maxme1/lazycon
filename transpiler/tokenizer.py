from .token import *


def tokenize(source: str):
    tokens = []
    for line_number, line in enumerate(source.splitlines()):
        line = line.rstrip()
        text = line.lstrip()
        while text:
            position = len(line) - len(text)
            token = next_token(text)
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
            return Token(match, tokenType)

    for char, tokeType in SINGLE.items():
        if text.startswith(char):
            return Token(char, tokeType)

    for literal in LITERALS:
        if text.startswith(literal):
            return Token(literal, TokenType.LITERAL)

    raise ValueError(f'Unexpected character: {repr(text[0])}')
