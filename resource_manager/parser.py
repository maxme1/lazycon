import bisect
from inspect import Parameter, Signature
from io import BytesIO
from tokenize import tokenize

from .visitor import Visitor
from .wrappers import *


def throw(message, position):
    raise SyntaxError(message + '\n  at %d:%d in %s' % position)


def get_substring(lines: Sequence[str], start_line: int, start_col: int, stop_line: int = None,
                  stop_col: int = None, lstrip: bool = True, rstrip: bool = True, keep_line: bool = True) -> str:
    lines = list(lines[start_line - 1:stop_line])

    lines[-1] = lines[-1][:stop_col]
    lines[0] = lines[0][start_col:]
    empty = 0

    # remove comments
    if lstrip:
        line = lines[0].strip()
        while line.startswith('#') or not line:
            lines.pop(0)
            line = lines[0].strip()

    if rstrip:
        line = lines[-1].strip()
        while line.startswith('#') or not line:
            if not line:
                empty += 1

            lines.pop()
            line = lines[-1].strip()

    body = '\n'.join(lines).strip()
    if keep_line and empty > 1:
        body += '\n'

    return body


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
    def __init__(self, source_path):
        self.source_path = source_path

    def get_position(self, node: ast.AST):
        return node.lineno, node.col_offset, self.source_path

    def generic_visit(self, node, *args, **kwargs):
        throw('This syntactic structure is not supported.', self.get_position(node))

    def _prepare_function(self, node: ast.FunctionDef):
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
            value = LocalNormalizer(self.source_path).visit(statement)
            if isinstance(statement, ast.Assert):
                assertions.extend(value)
            else:
                bindings.extend(value)

        # parameters
        args = node.args
        parameters = []
        # TODO: support
        if len(getattr(args, 'posonlyargs', [])) > 0:
            throw('Positional-only arguments are not supported.', self.get_position(node))

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
        return node.name, Function(
            Signature(parameters), docstring, bindings, ExpressionWrapper(ret.value, self.get_position(ret.value)),
            decorators, assertions, node.name, self.get_position(node),
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

    def visit_function_def(self, node: ast.FunctionDef):
        yield self._prepare_function(node)

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
    def __init__(self, start, stop, lines, source_path):
        super().__init__(source_path)
        self.lines = lines
        self.start = start
        self.stop = stop

    def visit_function_def(self, node: ast.FunctionDef):
        name, func = self._prepare_function(node)

        # body
        body = get_substring(self.lines, *self.start, *self.stop)
        for token in tokenize_string(body):
            if token.string == 'def':
                start = get_substring(body.splitlines(), 1, 0, *token.end)
                stop = get_substring(body.splitlines(), *token.end)
                assert stop.startswith(node.name)
                stop = stop[len(node.name):].strip()
                func.body = start, stop
                break

        assert func.body is not None
        yield name, func

    def visit_assign(self, node: ast.Assign):
        position = self.get_position(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                throw('This assignment syntax is not supported.', self.get_position(target))
            assert isinstance(target.ctx, ast.Store)

        last_target = node.targets[-1]
        body = get_substring(self.lines, last_target.lineno, last_target.col_offset, *self.stop)
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


# need this function, because in >=3.8 the function start is considered from `def` token
#   rather then from the first decorator
def find_body_limits(source: str, source_path: str):
    def _pos(node):
        return node.lineno, node.col_offset

    statements = sorted(ast.parse(source, source_path).body, key=_pos, reverse=True)
    tokens = list(tokenize_string(source))
    if not tokens:
        return

    indices = [t.start for t in tokens]
    stop = tokens[-1].end

    for statement in statements:
        start = _pos(statement)
        if isinstance(statement, ast.FunctionDef) and statement.decorator_list:
            dec = statement.decorator_list[0]
            start = _pos(dec)
            idx = bisect.bisect_left(indices, start)
            token = tokens[idx]
            assert token.start == start
            token = tokens[idx - 1]
            assert token.string == '@'
            start = token.start

        yield statement, start, stop
        stop = start


def parse(source: str, source_path: str):
    lines = tuple(source.splitlines() + [''])
    wrapped = []
    for statement, start, stop in reversed(list(find_body_limits(source, source_path))):
        wrapped.extend(GlobalNormalizer(start, stop, lines, source_path).visit(statement))

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
