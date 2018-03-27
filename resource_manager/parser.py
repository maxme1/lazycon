from tokenize import TokenError

from io import BytesIO

from .tokenizer import tokenize
from .token import TokenType
from .structures import *


class Parser:
    def __init__(self, tokens: List[TokenWrapper]):
        self.tokens = tokens
        self.position = 0

    def data(self):
        if self.matches(TokenType.IDENTIFIER):
            return Resource(self.advance())
        if self.matches(TokenType.BRACKET_OPEN):
            return self.array()
        if self.matches(TokenType.PAR_OPEN):
            return self.tuple()
        if self.matches(TokenType.DICT_OPEN):
            return self.dictionary()
        if self.matches(TokenType.LAMBDA):
            return self.lambda_()
        if self.matches(TokenType.MINUS, TokenType.NUMBER):
            return self.number()
        return Literal(self.require(TokenType.STRING, TokenType.LITERAL))

    def number(self):
        minus = self.ignore(TokenType.MINUS)
        return Number(self.require(TokenType.NUMBER), minus)

    def lambda_(self):
        token = self.require(TokenType.LAMBDA)
        params = []
        if self.matches(TokenType.IDENTIFIER):
            params = [self.advance()]
            while self.ignore(TokenType.COMA):
                params.append(self.require(TokenType.IDENTIFIER))
        self.require(TokenType.COLON)
        return Lambda(params, self.expression(), token)

    def is_keyword(self):
        return self.matches(TokenType.IDENTIFIER) and self.matches(TokenType.EQUALS, shift=1)

    def params(self):
        # TODO: need Argument class
        self.require(TokenType.PAR_OPEN)
        lazy = self.ignore(TokenType.LAZY)
        vararg, args, keyword = [], [], []
        if self.ignore(TokenType.PAR_CLOSE):
            return args, vararg, keyword, lazy

        # if has positional
        if not self.is_keyword():
            vararg.append(self.ignore(TokenType.ASTERISK))
            args.append(self.expression())

            while self.ignore(TokenType.COMA):
                if self.is_keyword() or self.matches(TokenType.PAR_CLOSE):
                    break
                vararg.append(self.ignore(TokenType.ASTERISK))
                args.append(self.expression())

        # keyword
        if self.matches(TokenType.IDENTIFIER):
            keyword.append(self.definition())

            while self.ignore(TokenType.COMA):
                if self.matches(TokenType.PAR_CLOSE):
                    break
                keyword.append(self.definition())

        self.require(TokenType.PAR_CLOSE)
        return args, vararg, keyword, lazy

    def expression(self):
        data = self.data()
        while self.matches(TokenType.DOT, TokenType.PAR_OPEN, TokenType.BRACKET_OPEN):
            if self.matches(TokenType.DOT):
                self.advance()
                name = self.require(TokenType.IDENTIFIER)
                data = GetAttribute(data, name)
            elif self.matches(TokenType.BRACKET_OPEN):
                self.advance()

                args, coma = [self.expression()], False
                while self.ignore(TokenType.COMA):
                    coma = True
                    if self.matches(TokenType.BRACKET_CLOSE):
                        break
                    args.append(self.expression())

                self.require(TokenType.BRACKET_CLOSE)
                data = GetItem(data, args, coma)
            else:
                data = Call(data, *self.params())

        return data

    def definition(self):
        name = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        return Definition(name, self.expression())

    def inline_structure(self, begin, end, constructor, get_data):
        structure_begin = self.require(begin)
        data, comas = [], 0
        if not self.ignore(end):
            data.append(get_data())
            while self.ignore(TokenType.COMA):
                comas += 1
                if self.matches(end):
                    break
                data.append(get_data())
            self.require(end)

        return constructor(data, structure_begin), comas

    def dictionary(self):
        return self.inline_structure(TokenType.DICT_OPEN, TokenType.DICT_CLOSE, Dictionary, self.pair)[0]

    def array(self):
        return self.inline_structure(TokenType.BRACKET_OPEN, TokenType.BRACKET_CLOSE, Array, self.expression)[0]

    def tuple(self):
        data, comas = self.inline_structure(TokenType.PAR_OPEN, TokenType.PAR_CLOSE, Tuple, self.expression)
        if comas == 0 and data.values:
            assert len(data.values) == 1
            return Parenthesis(data.values[0])
        return data

    def pair(self):
        key = self.expression()
        self.require(TokenType.COLON)
        return key, self.expression()

    def dotted(self):
        result = [self.require(TokenType.IDENTIFIER)]
        while self.ignore(TokenType.DOT):
            result.append(self.require(TokenType.IDENTIFIER))
        return result

    def import_as(self, allow_dotted):
        if allow_dotted:
            value = self.dotted()
        else:
            value = self.require(TokenType.IDENTIFIER),
        name = None
        if self.ignore(TokenType.AS):
            name = self.require(TokenType.IDENTIFIER)
        return tuple(value), name

    def import_(self):
        root, prefix_dots = [], 0
        if self.ignore(TokenType.FROM):
            if self.ignore(TokenType.DOT):
                prefix_dots += 1
            if self.ignore(TokenType.DOT):
                prefix_dots += 1
            root = self.dotted()

        main_token = self.require(TokenType.IMPORT)

        # import by path
        if not root and self.matches(TokenType.STRING):
            path = self.require(TokenType.STRING)
            if path.body.count(':') > 1:
                raise self.throw('The resulting path cannot contain more than one ":" separator.', path)
            return ImportPath(path, main_token)

        if self.ignore(TokenType.ASTERISK):
            return ImportStarred(root, prefix_dots, main_token)

        block = self.ignore(TokenType.PAR_OPEN)
        values = [self.import_as(not root)]
        while self.ignore(TokenType.COMA):
            values.append(self.import_as(not root))

        if block:
            self.require(TokenType.PAR_CLOSE)

        return UnifiedImport(root, values, prefix_dots, main_token)

    def parse(self):
        parents, imports = [], []
        while self.matches(TokenType.IMPORT, TokenType.FROM):
            import_ = self.import_()
            if isinstance(import_, UnifiedImport):
                imports.append(import_)
            else:
                if imports:
                    self.throw('Starred and path imports are only allowed at the top of the config', import_.main_token)
                parents.append(import_)

        definitions = []
        while self.position < len(self.tokens):
            definitions.append(self.definition())

        return definitions, parents, imports

    def advance(self) -> TokenWrapper:
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
            if temp.exact_type == tokenType.value:
                return True

        return False

    @staticmethod
    def throw(message, token):
        source = token.source or '<string input>'
        raise SyntaxError(message + '\n  at %d:%d in %s' % (token.line, token.column, source))

    def require(self, *types) -> TokenWrapper:
        if not self.matches(*types):
            self.throw('Unexpected token: "%s"' % self.current.body, self.current)
        return self.advance()

    def ignore(self, *types):
        if self.matches(*types):
            self.advance()
            return True
        return False


def parse(readline, source_path):
    try:
        tokens = tokenize(readline, source_path)
    except TokenError as e:
        source_path = source_path or '<string input>'
        raise SyntaxError(e.args[0] + ' at %d:%d in %s' % (e.args[1] + (source_path,))) from None
    return Parser(tokens).parse()


def parse_file(config_path):
    with open(config_path, 'rb') as file:
        return parse(file.readline, config_path)


def parse_string(source):
    return parse(BytesIO(source.encode()).readline, '')
