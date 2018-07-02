from tokenize import TokenError

from io import BytesIO

from .tokenizer import tokenize
from .token import TokenType
from .structures import *
from .arguments import NoDefaultValue, Parameter, PositionalArgument, KeywordArgument
from .exceptions import BadSyntaxError, custom_raise


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
            return self.tuple_or_parenthesis()
        if self.matches(TokenType.DICT_OPEN):
            return self.dictionary_or_set()
        if self.matches(TokenType.LAMBDA):
            return self.lambda_()
        return Literal(self.require(TokenType.STRING, TokenType.LITERAL, TokenType.NUMBER, TokenType.ELLIPSIS))

    def params(self, end):
        params, positional = [], True
        while not self.matches(end):
            if params:
                self.require(TokenType.COMA)

            vararg = False
            if positional:
                vararg = self.ignore(TokenType.ASTERISK)
                if vararg:
                    positional = False

            name = self.require(TokenType.IDENTIFIER)
            default = NoDefaultValue
            if not vararg and self.ignore(TokenType.EQUAL):
                default = self.inline_if()
            params.append(Parameter(name, vararg, positional=positional or vararg, keyword=not vararg, default=default))
        self.require(end)
        return params

    def lambda_(self):
        token = self.require(TokenType.LAMBDA)
        params = self.params(TokenType.COLON)
        return Lambda(params, self.inline_if(), token)

    def func_def(self):
        token = self.require(TokenType.DEF)
        name = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.PAR_OPEN)
        params = self.params(TokenType.PAR_CLOSE)
        self.require(TokenType.COLON)

        bindings = []
        while self.matches(TokenType.IDENTIFIER):
            bindings.append(self.definition())

        self.require(TokenType.RETURN)
        return FuncDef(params, bindings, self.inline_if(), name.body, token)

    def arguments(self):
        lazy = self.ignore(TokenType.LAZY)

        args, kwargs, positional = [], [], True
        while not self.matches(TokenType.PAR_CLOSE):
            if args or kwargs:
                self.require(TokenType.COMA)
                # trailing coma
                if self.matches(TokenType.PAR_CLOSE):
                    break

            if not positional:
                name = self.require(TokenType.IDENTIFIER)
                self.require(TokenType.EQUAL)
                kwargs.append(KeywordArgument(name, self.inline_if()))
            else:
                vararg = self.ignore(TokenType.ASTERISK)
                data = self.inline_if()
                if not vararg and type(data) is Resource and self.ignore(TokenType.EQUAL):
                    positional = False
                    kwargs.append(KeywordArgument(data.main_token, self.inline_if()))
                else:
                    args.append(PositionalArgument(vararg, data))

        return tuple(args), tuple(kwargs), lazy

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
                args, kwargs, lazy = self.arguments()
                self.require(TokenType.PAR_CLOSE)
                data = Call(data, args, kwargs, lazy, main_token)

        return data

    def slice(self, start):
        token = self.require(TokenType.COLON)
        args = [start]
        if self.matches(TokenType.COLON, TokenType.COMA, TokenType.BRACKET_CLOSE):
            # start: ?
            args.append(None)
        else:
            # start:stop ?
            args.append(self.inline_if())

        if self.ignore(TokenType.COLON):
            if self.matches(TokenType.COMA, TokenType.BRACKET_CLOSE):
                # start:?:
                args.append(None)
            else:
                # start:?:step
                args.append(self.inline_if())
        else:
            # start::
            args.append(None)

        return Slice(args[0], args[1], args[2], token)

    def slice_or_if(self):
        if self.matches(TokenType.COLON):
            return self.slice(None)

        start = self.inline_if()
        if self.matches(TokenType.COLON):
            return self.slice(start)

        return start

    def definition(self):
        name = self.require(TokenType.IDENTIFIER)
        self.require(TokenType.EQUAL)
        return Definition([name], self.inline_if())

    def multiple_definitions(self):
        names = [self.require(TokenType.IDENTIFIER)]
        self.require(TokenType.EQUAL)
        data = self.inline_if()
        while self.ignore(TokenType.EQUAL):
            token = data.main_token
            if not isinstance(data, Resource):
                self.throw('Invalid identifier', token)
            names.append(token)
            data = self.inline_if()
        return Definition(names, data)

    def inline_container(self, begin, end, get_data) -> (InlineContainer, int):
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

        return data, structure_begin, comas

    def dictionary_or_set(self):
        data, main_token, _ = self.inline_container(TokenType.DICT_OPEN, TokenType.DICT_CLOSE, self.pair_or_value)
        types = [type(x) is tuple for x in data]
        if all(types):
            return Dictionary(data, main_token)
        if any(types):
            self.throw('Inline structure contains both set and dict elements', main_token)
        return Set(data, main_token)

    def array(self):
        data, main_token, _ = self.inline_container(TokenType.BRACKET_OPEN, TokenType.BRACKET_CLOSE, self.starred_or_if)
        return Array(data, main_token)

    def tuple_or_parenthesis(self):
        data, main_token, comas = self.inline_container(TokenType.PAR_OPEN, TokenType.PAR_CLOSE, self.starred_or_if)
        if comas == 0 and data:
            assert len(data) == 1
            if type(data[0]) is Starred:
                self.throw('Cannot use starred expression here', main_token)
            return Parenthesis(data[0])
        return Tuple(data, main_token)

    def pair_or_value(self):
        key = self.starred_or_if()
        if type(key) is not Starred and self.ignore(TokenType.COLON):
            return key, self.inline_if()
        return key

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
            while self.matches(TokenType.DOT, TokenType.ELLIPSIS):
                if self.matches(TokenType.ELLIPSIS):
                    prefix_dots += 2
                prefix_dots += 1
                self.advance()
            root = self.dotted()

        main_token = self.require(TokenType.IMPORT)

        # import by path
        if not root and self.matches(TokenType.STRING):
            path = self.require(TokenType.STRING)
            if path.body.count(':') > 1:
                self.throw('The resulting path cannot contain more than one ":" separator.', path)
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
            if self.matches(TokenType.DEF):
                val = self.func_def()
            else:
                val = self.multiple_definitions()
            definitions.append(val)

        return definitions, parents, imports

    def advance(self) -> TokenWrapper:
        result = self.current
        self.position += 1
        return result

    @property
    def current(self):
        if self.position >= len(self.tokens):
            message = 'Unexpected end of source'
            if self.tokens and self.tokens[0].source:
                message += ' in ' + self.tokens[0].source
            custom_raise(BadSyntaxError(message))
        return self.tokens[self.position]

    def matches(self, *types):
        try:
            temp = self.tokens[self.position]
        except IndexError:
            return False

        for tokenType in types:
            if temp.type == tokenType:
                return True

        return False

    @staticmethod
    def throw(message, token):
        source = token.source or '<string input>'
        custom_raise(BadSyntaxError(message + '\n  at %d:%d in %s\n    %s' %
                                    (token.line, token.column, source, token.token_line.rstrip())))

    def require(self, *types) -> TokenWrapper:
        if not self.matches(*types):
            current = self.current
            self.throw('Unexpected token: "%s"' % current.body, current)
        return self.advance()

    def ignore(self, *types):
        if self.matches(*types):
            self.advance()
            return True
        return False


def parse(readline, source_path):
    try:
        return Parser(tokenize(readline, source_path)).parse()
    except TokenError as e:
        source_path = source_path or '<string input>'
        line, col = e.args[1]
        custom_raise(BadSyntaxError(e.args[0] + ' at %d:%d in %s' % (line, col, source_path)), None)


def parse_file(config_path):
    with open(config_path, 'rb') as file:
        return parse(file.readline, config_path)


def parse_string(source):
    return parse(BytesIO(source.encode()).readline, '')
