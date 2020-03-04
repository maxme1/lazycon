from inspect import Parameter, Signature
from io import BytesIO
from tokenize import tokenize

from .visitor import Visitor
from .wrappers import *


def throw(message, position):
    raise SyntaxError(message + '\n  at %d:%d in %s' % position)


def get_substring(lines, start_line, start_col, stop_line=None, stop_col=None) -> str:
    # TODO: remove comments
    lines = list(lines[start_line - 1:stop_line])

    lines[-1] = lines[-1][:stop_col]
    lines[0] = lines[0][start_col:]

    return '\n'.join(lines).strip()


def tokenize_string(source):
    return tokenize(BytesIO(source.encode()).readline)


def flatten_assignment(pattern):
    if isinstance(pattern, str):
        return [pattern]

    result = []
    for x in pattern:
        result.extend(flatten_assignment(x))
    return result


class Normalizer(Visitor):
    def __init__(self, stop, lines, source_path):
        self.lines = lines
        self.stop = stop
        self.source_path = source_path

    def get_body(self, start: ast.AST):
        if self.stop is None:
            stop = None, None
        else:
            stop = self.stop.lineno, self.stop.col_offset

        return get_substring(self.lines, start.lineno, start.col_offset, *stop)

    def get_position(self, node: ast.AST):
        return node.lineno, node.col_offset, self.source_path

    @classmethod
    def normalize(cls, node, stop, lines, source_path):
        return cls(stop, lines, source_path).visit(node)

    def generic_visit(self, node, *args, **kwargs):
        throw('This syntactic structure is not supported.', self.get_position(node))

    def visit_function_def(self, node: ast.FunctionDef):
        *raw_bindings, ret = node.body
        if not isinstance(ret, ast.Return):
            throw('Functions must end with a return statement.', self.get_position(ret))

        # docstring
        docstring = None
        if raw_bindings and isinstance(raw_bindings[0], ast.Expr) and isinstance(raw_bindings[0].value, ast.Str):
            docstring, raw_bindings = raw_bindings[0].value.s, raw_bindings[1:]

        # bindings
        bindings, assertions = [], []
        for statement, stop in zip(raw_bindings, node.body[1:]):
            value = LocalNormalizer.normalize(statement, stop, self.lines, self.source_path)
            if isinstance(statement, ast.Assert):
                assertions.extend(value)
            else:
                bindings.extend(value)

        # parameters
        args = node.args
        parameters = []
        for arg, default in zip(args.args, [None] * (len(args.args) - len(args.defaults)) + args.defaults):
            if default is None:
                default = Parameter.empty
            else:
                default = ExpressionWrapper(default, self.get_position(default))
            parameters.append(Parameter(arg.arg, Parameter.POSITIONAL_OR_KEYWORD, default=default))

        if args.vararg is not None:
            parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))

        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is None:
                default = Parameter.empty
            else:
                default = ExpressionWrapper(default, self.get_position(default))
            parameters.append(Parameter(arg.arg, Parameter.KEYWORD_ONLY, default=default))

        if args.kwarg is not None:
            parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))

        # decorators
        decorators = [ExpressionWrapper(decorator, self.get_position(decorator)) for decorator in node.decorator_list]

        # body
        body = self.get_body(node)
        for token in tokenize_string(body):
            if token.string == 'def':
                start = get_substring(body.splitlines(), 1, 0, *token.end)
                stop = get_substring(body.splitlines(), *token.end)
                assert stop.startswith(node.name)
                stop = stop[len(node.name):].strip()

                break

        yield node.name, Function(
            Signature(parameters), docstring, bindings, ExpressionWrapper(ret.value, self.get_position(ret.value)),
            decorators, assertions, node.name, (start, stop), self.get_position(node)
        )


class LocalNormalizer(Normalizer):
    def get_assignment_pattern(self, target):
        assert isinstance(target.ctx, ast.Store)

        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Starred):
            throw('Starred unpacking is not supported.', self.get_position(target))

        assert isinstance(target, (ast.Tuple, ast.List))
        return tuple(self.get_assignment_pattern(elt) for elt in target.elts)

    def visit_assert(self, node: ast.Assert):
        yield AssertionWrapper(node, self.get_position(node))

    def visit_assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            throw('Assignments inside functions must have a single target.', self.get_position(node))

        pattern = self.get_assignment_pattern(node.targets[0])
        expression = PatternAssignment(node.value, pattern, self.get_position(node.value))
        for name in flatten_assignment(pattern):
            yield name, expression


class GlobalNormalizer(Normalizer):
    def visit_assign(self, node: ast.Assign):
        position = self.get_position(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                throw('This assignment syntax is not supported.', self.get_position(target))
            assert isinstance(target.ctx, ast.Store)

        last_target = node.targets[-1]
        body = self.get_body(last_target)
        assert body[:len(last_target.id)] == last_target.id
        body = body[len(last_target.id):].lstrip()
        assert body[0] == '='
        body = body[1:].lstrip()

        expression = ExpressionStatement(node.value, body, position)
        for target in node.targets:
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


def parse(source: str, source_path: str):
    statements = ast.parse(source, source_path).body
    lines = tuple(source.splitlines())
    wrapped = []
    for statement, stop in zip(statements, statements[1:] + [None]):
        wrapped.extend(GlobalNormalizer.normalize(statement, stop, lines, source_path))

    parents, imports, definitions = [], [], []
    for name, w in wrapped:
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
            assert isinstance(w, (Function, ExpressionStatement))
            definitions.append((name, w))

    return parents, imports, definitions


def parse_file(config_path):
    with open(config_path, 'r') as file:
        return parse(file.read(), config_path)


def parse_string(source):
    return parse(source, '<string input>')
