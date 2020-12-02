import ast
from collections import defaultdict, OrderedDict
from inspect import Parameter
from typing import Iterable, List, Dict

from ..utils import reverse_mapping
from ..wrappers import ExpressionStatement, Function, Wrapper
from ..scope import ScopeDict
from ..exceptions import SemanticError
from .visitor import SemanticVisitor


def global_definition(method):
    def wrapper(self, node, *args, **kwargs):
        self._source_paths.append(node.source_path)
        value = method(self, node, *args, **kwargs)
        self._source_paths.pop()
        return value

    return wrapper


def position(node: ast.AST):
    return node.lineno, node.col_offset


READ_ONLY = {'__file__'}


class Semantics(SemanticVisitor):
    def __init__(self, name_to_node: ScopeDict, builtins: Iterable[str]):
        self.messages = defaultdict(lambda: defaultdict(set))
        self._scopes: List[Dict[str, MarkedNode]] = []
        self._source_paths = []
        self._builtins = builtins
        self.leave_time = {}
        self._current_time = 0
        self._statements: List[Wrapper] = []
        # TODO: use ordered set
        self._parents: Dict[Wrapper, List[Wrapper]] = defaultdict(list)

        self.node_to_names = reverse_mapping(name_to_node)
        self.enter_scope(name_to_node)
        self.analyze_global_scope()

    def add_message(self, message, content, source=None):
        source = source or self._source_paths[-1]
        self.messages[message][source].add(content)

    @staticmethod
    def format(message, elements):
        message += ':\n'
        for source, item in elements.items():
            message += '  in %s\n    ' % source
            message += ', '.join(item)
            message += '\n'
        return message

    @classmethod
    def analyze(cls, scope: ScopeDict, builtins: Iterable[str]):
        tree = cls(scope, builtins)
        message = ''
        for msg, elements in tree.messages.items():
            message += tree.format(msg, elements)
        if message:
            raise SemanticError(message)
        return tree._parents

    def enter_scope(self, names: ScopeDict, visited: Iterable[str] = ()):
        scope = OrderedDict()
        for name, value in names.items():
            scope[name] = MarkedNode(False, value)
        for name in visited:
            scope[name] = MarkedNode(True)
        self._scopes.append(scope)

    def leave_scope(self):
        self._scopes.pop()

    def _mark_name(self, value: 'MarkedNode'):
        node = value.node
        if node in self.node_to_names:
            # if global
            assert len(self._scopes) == 1
            scope = self._scopes[0]
            for name in self.node_to_names[node]:
                scope[name].leave()
                self.leave_time[name] = self._current_time
                self._current_time += 1
            assert value.visited
        else:
            # if local
            value.leave()

    def _visit_definition(self, value: 'MarkedNode', level=0):
        node = value.node
        assert isinstance(node, Wrapper)
        assert not value.visited and not value.visiting

        n = len(self._scopes) - level
        self._scopes, tail = self._scopes[:n], self._scopes[n:]
        is_global = len(self._scopes) == 1
        value.enter()
        if is_global:
            # new global definition
            assert value.node in self.node_to_names
            self._statements.append(value.node)

        # allowing recursion
        if isinstance(node, Function) or (
                isinstance(node, ExpressionStatement) and isinstance(node.expression, ast.Lambda)):
            self._mark_name(value)
            self.visit(node)
        else:
            self.visit(node)
            self._mark_name(value)

        if is_global:
            self._statements.pop()
        self._scopes.extend(tail)

    def analyze_global_scope(self):
        for name, value in self._scopes[-1].items():
            if name in READ_ONLY:
                *pos, source = value.node.position
                self.add_message('The value is read-only', '"' + name + '" at %d:%d' % tuple(pos), source)

            if not value.visited:
                self._visit_definition(value)

    def visit_name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        name = node.id
        for level, scope in enumerate(reversed(self._scopes)):
            if name in scope:
                value = scope[name]
                if value.visiting:
                    self.add_message('Values are referenced before being completely defined (cyclic dependency)',
                                     '"' + name + '" at %d:%d' % position(node))
                elif not value.visited:
                    self._visit_definition(value, level)

                # track dependencies
                if level == len(self._scopes) - 1:
                    assert value.node in self.node_to_names
                    self._parents[self._statements[-1]].append(value.node)
                return

        if name not in self._builtins:
            self.add_message('Undefined resources found, but are required', name)

    # global definitions

    @global_definition
    def visit_expression_statement(self, node: ExpressionStatement):
        self.visit(node.expression)

    @global_definition
    def visit_function(self, node: Function):
        bindings = {name: binding for name, binding in node.bindings}
        if len(bindings) != len(node.bindings):
            self.add_message('Duplicate binding names in function definition', 'at %d:%d' % node.position[:2])

        names = [parameter.name for parameter in node.signature.parameters.values()]
        assert len(names) == len(set(names))

        if set(names) & set(bindings):
            self.add_message('Binding names clash with argument names in function definition',
                             'at %d:%d' % node.position[:2])

        self._visit_sequence(node.decorators)

        for parameter in node.signature.parameters.values():
            if parameter.default is not Parameter.empty:
                self.visit(parameter.default)

        self.enter_scope(bindings, names)
        self._visit_sequence(node.assertions)
        self.visit(node.expression)

        for value in self._scopes[-1].values():
            if not value.visited:
                self.add_message('Definition is never used', 'at %d:%d' % value.node.position[:2])

        self.leave_scope()

    @global_definition
    def visit_unified_import(self, node):
        pass

    # other stuff that manages scope

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

    def visit_lambda(self, node: ast.Lambda):
        args = node.args
        names = [arg.arg for arg in args.args + args.kwonlyargs]
        if args.vararg:
            names.append(args.vararg.arg)
        if args.kwarg:
            names.append(args.kwarg.arg)

        self._visit_sequence(args.defaults + list(filter(None, args.kw_defaults)))

        self.enter_scope({}, names)
        self.visit(node.body)
        self.leave_scope()


class MarkedNode:
    def __init__(self, visited: bool, node: ast.AST = None):
        self.node = node
        self.status = visited

    @property
    def visited(self):
        return bool(self.status)

    @property
    def visiting(self):
        return self.status is None

    def enter(self):
        assert self.status is False
        self.status = None

    def leave(self):
        assert not self.visited
        self.status = True
