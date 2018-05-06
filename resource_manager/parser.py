from tokenize import TokenError

from io import BytesIO

from .tokenizer import tokenize
from .token import TokenType
from .structures import *


class Parser:
    def __init__(self, tokens: List[TokenWrapper]):
        self.tokens = tokens
        self.position = 0

    def primary(self):
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
        return Literal(self.require(TokenType.STRING, TokenType.LITERAL, TokenType.NUMBER))

    def lambda_(self):
        token = self.require(TokenType.LAMBDA)
        vararg, params = self.ignore(TokenType.ASTERISK), []
        if vararg:
            params = [self.require(TokenType.IDENTIFIER)]
        else:
            if self.matches(TokenType.IDENTIFIER):
                params = [self.advance()]
                while self.ignore(TokenType.COMA):
                    vararg = self.ignore(TokenType.ASTERISK)
                    params.append(self.require(TokenType.IDENTIFIER))
                    if vararg:
                        break
        self.require(TokenType.COLON)
        return Lambda(params, vararg, self.expression(), token)

    def is_keyword(self):
        return self.matches(TokenType.IDENTIFIER) and self.matches(TokenType.EQUALS, shift=1)

    def params(self):
        # TODO: need Argument class
        lazy = self.ignore(TokenType.LAZY)
        vararg, args, keyword = [], [], []
        if self.matches(TokenType.PAR_CLOSE):
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

        return args, vararg, keyword, lazy

    def inline_if(self):
        data = self.expression()
        if self.matches(TokenType.IF):
            token = self.advance()
            condition = self.expression()
            self.require(TokenType.ELSE)
            return InlineIf(condition, data, self.inline_if(), token)
        return data

    def expression(self):
        return self.or_exp()

    def binary(self, get_data, *operations):
        data = get_data()
        while self.matches(*operations):
            operation = self.advance()
            data = Binary(data, get_data(), operation)
        return data

    def unary(self, get_data, *operations):
        if self.matches(*operations):
            operation = self.advance()
            return Unary(get_data(), operation)
        return get_data()

    def or_exp(self):
        return self.binary(self.and_exp, TokenType.OR)

    def and_exp(self):
        return self.binary(self.not_exp, TokenType.AND)

    def not_exp(self):
        return self.unary(self.comparison, TokenType.NOT)

    def comparison(self):
        data = self.bitwise_or()
        while self.matches(TokenType.LESS, TokenType.GREATER, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL,
                           TokenType.IS_EQUAL, TokenType.NOT_EQUAL, TokenType.IS, TokenType.NOT, TokenType.IN):
            if self.matches(TokenType.NOT):
                operation = self.advance()
                operation = (operation, self.require(TokenType.IN))
            elif self.matches(TokenType.IS):
                operation = self.advance()
                if self.matches(TokenType.NOT):
                    operation = (operation, self.advance())
            else:
                operation = self.advance()
            data = Binary(data, self.bitwise_or(), operation)
        return data

    def bitwise_or(self):
        return self.binary(self.bitwise_xor, TokenType.BIT_OR)

    def bitwise_xor(self):
        return self.binary(self.bitwise_and, TokenType.BIT_XOR)

    def bitwise_and(self):
        return self.binary(self.shift, TokenType.BIT_AND)

    def shift(self):
        return self.binary(self.arithmetic, TokenType.SHIFT_LEFT, TokenType.SHIFT_RIGHT)

    def arithmetic(self):
        return self.binary(self.term, TokenType.PLUS, TokenType.MINUS)

    def term(self):
        return self.binary(self.factor, TokenType.ASTERISK, TokenType.MATMUL,
                           TokenType.DIVIDE, TokenType.FLOOR_DIVIDE, TokenType.MOD)

    def factor(self):
        return self.unary(self.power, TokenType.PLUS, TokenType.MINUS, TokenType.TILDE)

    def power(self):
        data = self.tailed()
        if self.matches(TokenType.DOUBLE_ASTERISK):
            operation = self.advance()
            data = Binary(data, self.factor(), operation)
        return data

    def tailed(self):
        data = self.primary()
        while self.matches(TokenType.DOT, TokenType.PAR_OPEN, TokenType.BRACKET_OPEN):
            if self.matches(TokenType.DOT):
                self.advance()
                name = self.require(TokenType.IDENTIFIER)
                data = GetAttribute(data, name)
            elif self.matches(TokenType.BRACKET_OPEN):
                self.advance()

                args, coma = [self.slice_or_if()], False
                while self.ignore(TokenType.COMA):
                    coma = True
                    if self.matches(TokenType.BRACKET_CLOSE):
                        break
                    args.append(self.slice_or_if())

                self.require(TokenType.BRACKET_CLOSE)
                data = GetItem(data, args, coma)
            else:
                main_token = self.require(TokenType.PAR_OPEN)
                params = self.params()
                self.require(TokenType.PAR_CLOSE)
                data = Call(data, *params, main_token=main_token)

        return data

    def slice(self, start):
        token = self.require(TokenType.COLON)
        args = [start]
        if self.matches(TokenType.COLON, TokenType.COMA, TokenType.BRACKET_OPEN):
            # start: ?
            args.append(None)
        else:
            # start:stop ?
            args.append(self.inline_if())

        if self.ignore(TokenType.COLON):
            if self.matches(TokenType.COMA, TokenType.BRACKET_OPEN):
                # start:?:
                args.append(None)
            else:
                # start:?:step
                args.append(self.inline_if())
        else:
            # start::
            args.append(None)

        return Slice(*args, token)

    def slice_or_if(self):
        if self.matches(TokenType.COLON):
            return self.slice(None)

        start = self.inline_if()
        if self.matches(TokenType.COLON):
            return self.slice(start)

        return start

    def definition(self):
        name = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUALS)
        return Definition(name, self.inline_if())

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
        return self.inline_structure(TokenType.BRACKET_OPEN, TokenType.BRACKET_CLOSE, Array, self.starred_or_if)[0]

    def tuple(self):
        data, comas = self.inline_structure(TokenType.PAR_OPEN, TokenType.PAR_CLOSE, Tuple, self.starred_or_if)
        if comas == 0 and data.values:
            assert len(data.values) == 1
            if type(data.values[0]) is Starred:
                self.throw('Cannot use starred expression here', data.main_token)
            return Parenthesis(data.values[0])
        return data

    def pair(self):
        key = self.inline_if()
        self.require(TokenType.COLON)
        return key, self.inline_if()

    def starred_or_if(self):
        if self.matches(TokenType.ASTERISK):
            star = self.require(TokenType.ASTERISK)
            return Starred(self.bitwise_or(), star)
        return self.inline_if()

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
            if temp.type == tokenType:
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
