import re
from inspect import Parameter, Signature
from io import BytesIO
from tokenize import tokenize, tok_name

from .visitor import Visitor
from .exceptions import DeprecationError
from .wrappers import *

PARTIAL = re.compile(r'^#\s*(lazy|partial)\s*$')


def throw(message, position):
    raise SyntaxError(message + '\n  at %d:%d in %s' % position)


def get_body(lines, start: ast.AST, stop_line, stop_col) -> str:
    lines = list(lines[start.lineno - 1:stop_line])

    lines[-1] = lines[-1][:stop_col]
    lines[0] = lines[0][start.col_offset:]

    return '\n'.join(lines).strip()


def tokenize_string(source):
    return tokenize(BytesIO(source.encode()).readline)


class Normalizer(Visitor):
    def __init__(self, stop, lines, source_path):
        self.lines = lines
        self.stop = stop
        self.source_path = source_path

    def get_body(self, start: ast.AST):
        if self.stop is None:
            stop_line = stop_col = None
        else:
            stop_line, stop_col = self.stop.lineno, self.stop.col_offset

        return get_body(self.lines, start, stop_line, stop_col)

    def get_position(self, node: ast.AST):
        return node.lineno, node.col_offset, self.source_path

    @staticmethod
    def normalize(node, stop, lines, source_path):
        return Normalizer(stop, lines, source_path).visit(node)

    def generic_visit(self, node, *args, **kwargs):
        throw('This syntactic structure is not supported.', self.get_position(node))

    def visit_assign(self, node: ast.Assign):
        position = self.get_position(node.value)

        last_target = node.targets[-1]
        body = self.get_body(last_target)
        assert body[:len(last_target.id)] == last_target.id
        body = body[len(last_target.id):].lstrip()
        assert body[0] == '='
        body = body[1:].lstrip()

        expression = ExpressionWrapper(node.value, body, position)

        for target in node.targets:
            if not isinstance(target, ast.Name):
                throw('This assignment syntax is not supported.', self.get_position(target))
            assert isinstance(target.ctx, ast.Store)

            yield target.id, expression

    def visit_import_from(self, node: ast.ImportFrom):
        names = node.names
        root = node.module.split('.')
        position = self.get_position(node)
        if len(names) == 1 and names[0].name == '*':
            yield None, ImportStarred(root, node.level, position)
            return

        for alias in names:
            name = alias.asname or alias.name
            yield name, UnifiedImport(root, node.level, alias.name.split(','), alias.asname is not None, position)

    def visit_import(self, node: ast.Import):
        position = self.get_position(node)

        for alias in node.names:
            name = alias.asname or alias.name
            yield name, UnifiedImport('', 0, alias.name.split('.'), alias.asname is not None, position)

    def visit_function_def(self, node: ast.FunctionDef):
        *raw_bindings, ret = node.body
        if not isinstance(ret, ast.Return):
            throw('Functions must end with a return statement.', self.get_position(ret))

        # TODO: add assertions?
        # bindings
        bindings = []
        for statement, stop in zip(raw_bindings, node.body[1:]):
            if not isinstance(statement, (ast.Assign, ast.FunctionDef)):
                throw('A function definition must consist of value or function definitions '
                      'followed by a return statement.', self.get_position(statement))

            if isinstance(statement, ast.Assign) and len(statement.targets) != 1:
                throw('Assignments inside function must have a single target.', self.get_position(statement))

            bindings.extend(Normalizer.normalize(statement, stop, self.lines, self.source_path))

        # return statement
        value = ret.value
        body = self.get_body(ret)
        assert body[:6] == 'return'
        body = body[6:].lstrip()
        expression = ExpressionWrapper(value, body, self.get_position(value))

        # TODO: add defaults
        if node.args.defaults or not all(d is None for d in node.args.kw_defaults):
            throw('Function default argument values are not supported.', self.get_position(node))

        # parameters
        args = node.args
        parameters = []
        for arg in args.args:
            parameters.append(Parameter(arg.arg, Parameter.POSITIONAL_OR_KEYWORD))
        if args.vararg is not None:
            parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
        for arg in args.kwonlyargs:
            parameters.append(Parameter(arg.arg, Parameter.KEYWORD_ONLY))
        if args.kwarg is not None:
            parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))

        # decorators
        decorators = []
        for decorator, stop in zip(node.decorator_list, node.decorator_list[1:]):
            body = get_body(self.lines, decorator, stop.lineno, stop.col_offset)
            assert body.endswith('@')
            body = body[:-1].strip()
            decorators.append(ExpressionWrapper(decorator, body, self.get_position(decorator)))

        if node.decorator_list:
            # get the `def` token position
            for token in tokenize_string(self.get_body(node)):
                if token.string == 'def':
                    break

            line, column = token.start
            decorator = node.decorator_list[-1]
            body = get_body(self.lines, decorator, line + node.lineno - 1, column)
            decorators.append(ExpressionWrapper(decorator, body, self.get_position(decorator)))

        yield node.name, Function(
            Signature(parameters), bindings, expression, decorators, node.name, self.get_position(node))


def parse(source: str, source_path: str):
    statements = ast.parse(source, source_path).body
    lines = tuple(source.splitlines())
    wrapped = []
    for statement, stop in zip(statements, statements[1:] + [None]):
        wrapped.extend(Normalizer.normalize(statement, stop, lines, source_path))

    parents, imports, definitions = [], [], []
    for name, w in wrapped:
        # TODO: this is ugly
        if isinstance(w, ImportStarred):
            assert name is None
            if imports or definitions:
                throw('Starred imports are only allowed at the top of the config.', w.position)
            parents.append(w)

        elif isinstance(w, UnifiedImport):
            if definitions:
                throw('Imports are only allowed before definitions.', w.position)
            imports.append((name, w))

        else:
            assert isinstance(w, (Function, ExpressionWrapper))
            definitions.append((name, w))

    # TODO: this is legacy
    for token in tokenize_string(source):
        if tok_name[token.type] == 'COMMENT' and PARTIAL.match(token.string.strip()):
            raise DeprecationError('The "# partial" syntax is not supported anymore.\n'
                                   '    Your config contains such a comment in %s' % source_path)

    return parents, imports, definitions


def parse_file(config_path):
    with open(config_path, 'r') as file:
        return parse(file.read(), config_path)


def parse_string(source):
    return parse(source, '<string input>')
