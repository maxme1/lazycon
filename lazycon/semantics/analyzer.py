import ast
from collections import defaultdict
from enum import Enum
from typing import Iterable, List, Dict, Sequence, Optional

from .locals import LocalsGatherer, extract_assign_targets
from ..statements import GlobalStatement, GlobalFunction, GlobalAssign, GlobalImport
from ..exceptions import SemanticError
from .visitor import SemanticVisitor


def position(node: ast.AST):
    return node.lineno, node.col_offset


# TODO: __folder__
READ_ONLY = {'__file__'}
IGNORE_NAME = '_'
NodeParents = Dict[GlobalStatement, List[GlobalStatement]]


class Semantics(SemanticVisitor):
    def __init__(self, statements: Sequence[GlobalStatement], builtins: Iterable[str]):
        self.messages = defaultdict(lambda: defaultdict(set))

        # scopes
        self._builtins = builtins
        self._global_scope: Dict[str, MarkedValue] = {
            statement.name: MarkedValue(statement) for statement in statements
        }
        self._local_scopes: List[Dict[str, Marked]] = []

        # tracking
        self._global_statement: Optional[GlobalStatement] = None
        # TODO: use ordered set
        self.parents: NodeParents = defaultdict(list)

        # analysis
        for name, value in self._global_scope.items():
            if name in READ_ONLY:
                *pos, source = value.value.position
                self.add_message('The value is read-only', '"' + name + '" at %d:%d' % tuple(pos), source)

            self.visit(value.value)

    @staticmethod
    def format(message, elements):
        message += ':\n'
        for source, item in elements.items():
            message += '  in %s\n    ' % source
            message += ', '.join(item)
            message += '\n'
        return message

    def check(self):
        message = ''
        for msg, elements in self.messages.items():
            message += self.format(msg, elements)
        if message:
            raise SemanticError(message)

    def add_message(self, message, content, source=None):
        # TODO: move line info here?
        source = source or self._global_statement.source_path
        self.messages[message][source].add(content)

    # scope management

    def enter_scope(self, names: Iterable[str], visited: Iterable[str] = ()):
        scope = {}
        for name in names:
            scope[name] = Marked(VisitState.Undefined)
        for name in visited:
            scope[name] = Marked(VisitState.Defined)
        self._local_scopes.append(scope)

    def leave_scope(self):
        self._local_scopes.pop()

    def enter(self, name: str):
        if self._local_scopes:
            value = self._local_scopes[-1][name]
            # allow multiple definitions
            if value.state is not VisitState.Defined:
                value.enter()

        else:
            self._global_scope[name].enter()

    def leave(self, name: str):
        if self._local_scopes:
            value = self._local_scopes[-1][name]
            # allow multiple definitions
            if value.state is not VisitState.Defined:
                value.leave()

        else:
            self._global_scope[name].leave()

    def generic_visit(self, node: ast.AST, *args, **kwargs):
        self.add_message('This syntactic structure is not supported',
                         f'{type(node).__name__} at %d:%d' % position(node))

    # the most important part - variable resolving

    def visit_name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        name = node.id
        if name == IGNORE_NAME:
            self.add_message(f'The name "{IGNORE_NAME}" can only be used as wildcard during unpacking',
                             'at %d:%d' % position(node))

        # local scopes
        for level, scope in enumerate(reversed(self._local_scopes)):
            if name in scope:
                value = scope[name]
                # allow late binding
                if level == 0 and value.state is not VisitState.Defined:
                    self.add_message('Local variables referenced before being defined',
                                     '"' + name + '" at %d:%d' % position(node))
                return

        # global scope
        if name in self._global_scope:
            value = self._global_scope[name]
            if value.state is VisitState.Defining:
                self.add_message('Values are referenced before being completely defined (cyclic dependency)',
                                 '"' + name + '" at %d:%d' % position(node))

            self.parents[self._global_statement].append(value.value)
            return

            # builtins
        if name not in self._builtins:
            self.add_message('Undefined names found', name)

    # global definitions

    def visit_global_assign(self, statement: GlobalAssign):
        assert self._global_statement is None
        self._global_statement = statement

        self.enter(statement.name)
        self.visit(statement.expression)
        self.leave(statement.name)

        self._global_statement = None

    def visit_global_function(self, statement: GlobalFunction):
        assert self._global_statement is None
        self._global_statement = statement

        self.visit(statement.node)

        self._global_statement = None

    def visit_global_import(self, statement: GlobalImport):
        self.enter(statement.name)
        self.leave(statement.name)

    visit_global_import_from = visit_global_import

    # local definitions

    def visit_assign(self, node: ast.Assign):
        names = extract_assign_targets(node.targets)
        for name in names:
            self.enter(name)

        self.visit(node.value)

        for name in names:
            self.leave(name)

    def visit_function_def(self, node: ast.FunctionDef):
        self.enter(node.name)
        self.visit(node.args)
        # TODO: type annotations?
        self._iterate_nodes(node.decorator_list)
        self.leave(node.name)

        # ignore docstring
        body = node.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Str):
            body = body[1:]

        self.enter_scope(LocalsGatherer.gather(body), self._gather_arg_names(node.args))
        self._iterate_nodes(body)
        self.leave_scope()

    # other stuff that manages scope

    def visit_lambda(self, node: ast.Lambda):
        self.visit(node.args)

        self.enter_scope([], self._gather_arg_names(node.args))
        self.visit(node.body)
        self.leave_scope()

    def visit_list_comp(self, node):
        for comp in node.generators:
            self.visit(comp)

        self.visit(node.elt)

        for _ in node.generators:
            self.leave_scope()

    def visit_dict_comp(self, node):
        for comp in node.generators:
            self.visit(comp)

        self.visit(node.key)
        self.visit(node.value)

        for _ in node.generators:
            self.leave_scope()

    visit_set_comp = visit_generator_exp = visit_list_comp

    def visit_comprehension(self, node: ast.comprehension):
        assert not getattr(node, 'is_async', False)

        def get_names(target):
            assert isinstance(target.ctx, ast.Store)
            if isinstance(target, (ast.Tuple, ast.List)):
                names = []
                for elt in target.elts:
                    names.extend(get_names(elt))
                return names

            if isinstance(target, ast.Starred):
                return [target.value]

            assert isinstance(target, ast.Name), target
            return [target.id]

        self.visit(node.iter)
        self.enter_scope({}, get_names(node.target))

        for test in node.ifs:
            self.visit(test)

    # function-related stuff

    def visit_return(self, node: ast.Return):
        self._visit_valid(node.value)

    @staticmethod
    def _gather_arg_names(node: ast.arguments):
        args = getattr(node, 'posonlyargs', []) + node.args + node.kwonlyargs
        if node.vararg is not None:
            args.append(node.vararg)
        if node.kwarg is not None:
            args.append(node.kwarg)
        return [arg.arg for arg in args]


class VisitState(Enum):
    Undefined, Defining, Defined = 0, 1, 2


class Marked:
    def __init__(self, status: VisitState):
        self.state = status

    def enter(self):
        assert self.state is VisitState.Undefined
        self.state = VisitState.Defining

    def leave(self):
        assert self.state is not VisitState.Defined
        self.state = VisitState.Defined


class MarkedValue(Marked):
    def __init__(self, value):
        super().__init__(VisitState.Undefined)
        self.value = value
