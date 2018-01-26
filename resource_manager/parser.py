import os

from .tokenizer import TokenType, tokenize
from .structures import *


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def module(self):
        module_type = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.COLON)
        name = self.require(TokenType.IDENTIFIER)

        return Module(module_type, name)

    def data(self):
        if self.matches(TokenType.IDENTIFIER):
            # module
            if self.matches(TokenType.COLON, shift=1):
                return self.module()
            else:
                # identifier
                return Resource(self.advance())

        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.DICT_OPEN):
            return self.object()
        return Literal(self.require(TokenType.STRING, TokenType.NUMBER, TokenType.LITERAL))

    def params(self):
        lazy = self.ignore(TokenType.LAZY)

        vararg, args, coma = [], [], False
        while not self.matches(TokenType.PAR_CLOSE):
            # if keyword parameter
            if coma:
                self.require(TokenType.COMA)
            if self.matches(TokenType.IDENTIFIER) and self.matches(TokenType.EQUALS, shift=1):
                break

            vararg.append(self.ignore(TokenType.ASTERISK))
            args.append(self.expression())
            coma = True

        params, coma = [], False
        while self.matches(TokenType.IDENTIFIER, TokenType.COMA):
            if coma:
                self.require(TokenType.COMA)
            params.append(self.definition())
            coma = True

        if params or args:
            self.ignore(TokenType.COMA)
        return args, vararg, params, lazy

    def expression(self):
        data = self.data()
        while self.matches(TokenType.DOT, TokenType.PAR_OPEN, TokenType.BRACKET_OPEN):
            if self.matches(TokenType.DOT):
                self.advance()
                name = self.require(TokenType.IDENTIFIER)
                data = GetAttribute(data, name)
            elif self.matches(TokenType.BRACKET_OPEN):
                self.advance()

                args = [self.expression()]
                coma = False
                while self.matches(TokenType.COMA):
                    self.advance()
                    if self.matches(TokenType.BRACKET_CLOSE):
                        coma = True
                        break
                    args.append(self.expression())

                self.require(TokenType.BRACKET_CLOSE)
                data = GetItem(data, args, coma)
            else:
                self.advance()
                data = Call(data, *self.params())
                self.require(TokenType.PAR_CLOSE)

        return data

    def definition(self):
        left = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        right = self.expression()

        return Definition(left, right)

    def object(self):
        dict_begin = self.require(TokenType.DICT_OPEN)
        pairs = {}
        while not self.matches(TokenType.DICT_CLOSE):
            key = self.require(TokenType.STRING)
            self.require(TokenType.COLON)
            pairs[key] = self.expression()

            self.ignore(TokenType.COMA)

        self.require(TokenType.DICT_CLOSE)
        return Dictionary(pairs, dict_begin)

    def array(self):
        array_begin = self.require(TokenType.BRACKET_OPEN)
        values = []
        while not self.matches(TokenType.BRACKET_CLOSE):
            values.append(self.expression())
            self.ignore(TokenType.COMA)

        self.require(TokenType.BRACKET_CLOSE)
        return Array(values, array_begin)

    def dotted(self):
        result = [self.require(TokenType.IDENTIFIER)]
        while self.ignore(TokenType.DOT):
            result.append(self.require(TokenType.IDENTIFIER))
        return result

    def import_as(self, allow_dotted, token):
        value = self.dotted()
        name = None
        if self.ignore(TokenType.AS):
            name = self.require(TokenType.IDENTIFIER)
        if len(value) > 1 and not allow_dotted:
            self.throw('Dotted import is not allowed with the "from" argument', token)
        return tuple(value), name

    def import_(self):
        root = []
        local = False
        if self.ignore(TokenType.FROM):
            if self.matches(TokenType.STRING):
                root = eval(self.advance().body)
                self.require(TokenType.IMPORT)
                return self.import_config(root)

            local = self.ignore(TokenType.DOT)
            root = self.dotted()

        main_token = self.require(TokenType.IMPORT)

        if self.ignore(TokenType.ASTERISK):
            parent = ''
            if not local:
                parent = root[0].body + ':'
                root = root[1:]
            return [parent + os.sep.join(x.body for x in root) + '.config']

        # import by path
        if self.matches(TokenType.STRING) or self.matches(TokenType.STRING, shift=1):
            return self.import_config('')

        if local:
            self.throw("Local imports are only allowed for config files", main_token)

        block = self.ignore(TokenType.PAR_OPEN)
        values = {}
        value, name = self.import_as(not root, main_token)
        values[value] = name
        while self.matches(TokenType.COMA):
            self.advance()
            value, name = self.import_as(not root, main_token)
            values[value] = name
        self.ignore(TokenType.COMA)

        if block:
            self.require(TokenType.PAR_CLOSE)

        return ImportPython(root, values, main_token)

    def import_config(self, root):
        block = self.ignore(TokenType.PAR_OPEN)

        paths = [eval(self.require(TokenType.STRING).body)]
        self.ignore(TokenType.COMA)
        while self.matches(TokenType.STRING):
            paths.append(eval(self.advance().body))
            self.ignore(TokenType.COMA)

        if block:
            self.require(TokenType.PAR_CLOSE)
        if root:
            return [os.path.join(root, x) for x in paths]
        return paths

    def parse(self):
        parents, imports = [], []
        while self.matches(TokenType.IMPORT, TokenType.FROM):
            import_ = self.import_()
            if type(import_) is ImportPython:
                imports.append(import_)
            else:
                parents.extend(import_)

        definitions = []
        while self.position < len(self.tokens):
            definitions.append(self.definition())

        return definitions, parents, imports

    def advance(self):
        result = self.current
        self.position += 1
        return result

    @property
    def current(self):
        if self.position >= len(self.tokens):
            raise SyntaxError('Unexpected end of source')
        return self.tokens[self.position]

    def matches(self, *types, shift=0):
        try:
            temp = self.tokens[self.position + shift]
        except IndexError:
            return False

        for tokenType in types:
            if temp.type == tokenType:
                return True

        return False

    def throw(self, message, token):
        raise SyntaxError(message + '\n  at {}:{}'.format(self.current.line, self.current.column))

    def require(self, *types):
        if not self.matches(*types):
            raise SyntaxError('Unexpected token: '
                              '"{}"\n  at {}:{}'.format(self.current.body, self.current.line, self.current.column))
        return self.advance()

    def ignore(self, *types):
        if self.matches(*types):
            self.advance()
            return True
        return False


def parse(source, source_path):
    try:
        tokens = tokenize(source)
        for token in tokens:
            token.set_source(source_path)
        return Parser(tokens).parse()
    except SyntaxError as e:
        raise SyntaxError('{} in {}'.format(e.msg, source_path)) from None


def parse_file(config_path):
    with open(config_path) as file:
        return parse(file.read(), config_path)


def parse_string(source):
    return parse(source, '<string input>')
