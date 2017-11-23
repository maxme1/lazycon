from .token import *


def tokenize(source: str, indentation: int):
    tokens = []
    current_indent = 0
    stack = []

    for line_number, line in enumerate(source.splitlines(), 1):
        line = line.rstrip()
        text = line.lstrip()
        # TODO: ugly
        if not text.strip() or (text.startswith('#') and not LAZY.match(text)):
            continue

        # if not inside json
        if not stack:
            indent = len(line) - len(text)
            if indent % indentation:
                raise SyntaxError('Bad indentation at line {}'.format(line_number))
            indent = indent // indentation
            delta = indent - current_indent
            current_indent = indent
            if delta > 0:
                tokens.extend([Token('>>>', TokenType.BLOCK_OPEN, line_number)] * delta)
            elif delta < 0:
                tokens.extend([Token('<<<', TokenType.BLOCK_CLOSE, line_number)] * -delta)

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

            if token.type in JSON_OPEN:
                stack.append(token)
            if token.type in JSON_CLOSE:
                if not stack or JSON_CLOSE[token.type] != stack[-1].type:
                    raise SyntaxError('Invalid brackets balance at {}:{}'.format(token.line, token.column))
                stack.pop()

            tokens.append(token)

            text = text[len(token.body):]
            text = text.strip()

    # close remaining blocks
    tokens.extend([Token('<<<', TokenType.BLOCK_CLOSE)] * current_indent)
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
