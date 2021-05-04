import sys
import ast
import bisect
from io import BytesIO
from tokenize import tokenize
from typing import Tuple, Sequence

from .visitor import Visitor
from .statements import GlobalFunction, GlobalAssign, ImportConfig, GlobalImportFrom, GlobalImport, ImportBase, \
    GlobalStatement

NO_DECORATORS = sys.version_info[:2] > (3, 7)


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
    def __init__(self, start, stop, lines, source_path, extension):
        self.source_path = source_path
        self.lines = lines
        self.start = start
        self.stop = stop
        self.extension = extension

    def get_position(self, node: ast.AST):
        return node.lineno, node.col_offset, self.source_path

    def generic_visit(self, node, *args, **kwargs):
        throw('This syntactic structure is not supported.', self.get_position(node))

    def visit_function_def(self, node: ast.FunctionDef):
        body = get_substring(self.lines, *self.start, *self.stop)
        yield GlobalFunction(node, body, self.get_position(node))

    def visit_assign(self, node: ast.Assign):
        position = self.get_position(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                throw('This assignment syntax is not supported.', self.get_position(target))
            assert isinstance(target.ctx, ast.Store), target.ctx

        last_target = node.targets[-1]
        body = get_substring(self.lines, last_target.lineno, last_target.col_offset, *self.stop)
        assert body[:len(last_target.id)] == last_target.id, (body, last_target.id)
        body = body[len(last_target.id):].lstrip()
        assert body[0] == '=', body
        body = body[1:].lstrip()

        for target in node.targets:
            yield GlobalAssign(target.id, node.value, body, position)

    def visit_import_from(self, node: ast.ImportFrom):
        names = node.names
        root = node.module.split('.')
        position = self.get_position(node)
        # starred config import
        if len(names) == 1 and names[0].name == '*':
            yield ImportConfig(root, node.level, self.extension, position)
            return
        # relative imports make no sense, as there is no base module in configs
        if node.level > 0:
            throw('Relative imports are only supported for config files.', position)
        # absolute imports
        for alias in names:
            local = ast.ImportFrom(node.module, [alias], 0)
            ast.copy_location(local, node)
            yield GlobalImportFrom(local, position)

    def visit_import(self, node: ast.Import):
        position = self.get_position(node)

        for alias in node.names:
            local = ast.Import([alias])
            ast.copy_location(local, node)
            yield GlobalImport(local, position)


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
        if NO_DECORATORS and isinstance(statement, ast.FunctionDef) and statement.decorator_list:
            dec = statement.decorator_list[0]
            start = _pos(dec)
            idx = bisect.bisect_left(indices, start)
            token = tokens[idx]
            assert token.start == start, (token, start)
            token = tokens[idx - 1]
            assert token.string == '@', token
            start = token.start

        yield statement, start, stop
        stop = start


def parse(source: str, source_path: str, extension: str) -> Tuple[Sequence[ImportConfig], Sequence[GlobalStatement]]:
    lines = tuple(source.splitlines() + [''])
    wrapped = []
    for statement, start, stop in reversed(list(find_body_limits(source, source_path))):
        wrapped.extend(Normalizer(start, stop, lines, source_path, extension).visit(statement))

    parents, imports, definitions = [], [], []
    # TODO: move this to normalizer?
    for entry in wrapped:
        if isinstance(entry, ImportConfig):
            if imports or definitions:
                throw('Starred imports are only allowed at the top of the config.', entry.position)
            parents.append(entry)

        elif isinstance(entry, ImportBase):
            if definitions:
                throw('Imports are only allowed before definitions.', entry.position)
            imports.append(entry)

        else:
            assert isinstance(entry, (GlobalAssign, GlobalFunction))
            definitions.append(entry)

    return parents, imports + definitions


def parse_file(config_path, extension):
    with open(config_path, 'r') as file:
        return parse(file.read(), config_path, extension)


def parse_string(source, extension):
    return parse(source, '<string input>', extension)
