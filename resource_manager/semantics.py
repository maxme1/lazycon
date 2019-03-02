from collections import defaultdict, OrderedDict
from inspect import Parameter

from .scope import ScopeDict
from .visitor import Visitor
from .wrappers import *
from .exceptions import SemanticError


def change_source(method):
    def wrapper(self, node, *args, **kwargs):
        self._source_paths.append(node.source_path)
        value = method(self, node, *args, **kwargs)
        self._source_paths.pop()
        return value

    return wrapper


def position(node: ast.AST):
    return node.lineno, node.col_offset


READ_ONLY = {'__file__'}


class Semantics(Visitor):
    def __init__(self, name_to_node: ScopeDict, builtins: Iterable[str]):
        self.messages = defaultdict(lambda: defaultdict(set))
        self._scopes = []
        self._source_paths = []
        self._builtins = builtins
        self.leave_time = {}
        self._current_time = 0

        self.node_to_names = defaultdict(list)
        for name, node in name_to_node.items():
            self.node_to_names[node].append(name)
        self.node_to_names = dict(self.node_to_names)

        self.enter_scope(name_to_node)
        self.analyze_current_scope()

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

    @staticmethod
    def analyze(scope: ScopeDict, builtins: Iterable[str]):
        tree = Semantics(scope, builtins)
        message = ''
        for msg, elements in tree.messages.items():
            message += tree.format(msg, elements)
        if message:
            raise SemanticError(message)
        return tree.leave_time

    def enter_scope(self, names: ScopeDict, visited: Iterable[str] = ()):
        scope = OrderedDict()
        for name, value in names.items():
            scope[name] = [value, None]
        for name in visited:
            scope[name] = [None, True]
        self._scopes.append(scope)

    def leave_scope(self):
        self._scopes.pop()

    def not_visited(self, value):
        return value[1] is None

    def entered(self, value):
        return value[1] is False

    def _mark_name(self, value):
        node = value[0]
        if node in self.node_to_names:
            assert len(self._scopes) == 1
            scope = self._scopes[0]
            for name in self.node_to_names[node]:
                scope[name][1] = True
                self.leave_time[name] = self._current_time
                self._current_time += 1
            assert value[1]
        else:
            value[1] = True

    def mark_name(self, value, level=0):
        assert isinstance(value[0], Wrapper)
        assert value[1] is None
        n = len(self._scopes) - level
        self._scopes, tail = self._scopes[:n], self._scopes[n:]
        value[1] = False
        node = value[0]

        # allowing recursion
        if isinstance(node, Function) or (
                isinstance(node, ExpressionStatement) and isinstance(node.expression, ast.Lambda)):
            self._mark_name(value)
            self.visit(node)
        else:
            self.visit(node)
            self._mark_name(value)

        self._scopes.extend(tail)

    def analyze_current_scope(self):
        for name, value in self._scopes[-1].items():
            if name in READ_ONLY:
                *pos, source = value[0].position
                self.add_message('The value is read-only', '"' + name + '" at %d:%d' % tuple(pos), source)

            if self.not_visited(value):
                self.mark_name(value)

    def _visit_sequence(self, sequence: Iterable):
        for item in sequence:
            self.visit(item)

    def _visit_valid(self, value):
        if value is not None:
            self.visit(value)

    def _ignore_node(self, node):
        pass

    # visitors

    @change_source
    def visit_expression_statement(self, node: ExpressionStatement):
        self.visit(node.expression)

    visit_expression_wrapper = visit_expression_statement

    # literals

    visit_name_constant = visit_ellipsis = visit_bytes = visit_num = visit_str = _ignore_node

    def visit_formatted_value(self, node):
        assert node.format_spec is None
        self.visit(node.value)

    def visit_joined_str(self, node):
        self._visit_sequence(node.values)

    def visit_list(self, node: ast.List):
        assert isinstance(node.ctx, ast.Load)
        self._visit_sequence(node.elts)

    def visit_tuple(self, node):
        self.visit_list(node)

    def visit_set(self, node):
        self._visit_sequence(node.elts)

    def visit_dict(self, node):
        self._visit_sequence(filter(None, node.keys))
        self._visit_sequence(node.values)

    # variables

    def visit_name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        name = node.id
        for level, scope in enumerate(reversed(self._scopes)):
            if name in scope:
                if self.not_visited(scope[name]):
                    self.mark_name(scope[name], level)
                elif self.entered(scope[name]):
                    self.add_message('Values are referenced before being completely defined (cyclic dependency)',
                                     '"' + name + '" at %d:%d' % position(node))
                return

        if name not in self._builtins:
            self.add_message('Undefined resources found, but are required', name)

    def visit_starred(self, node: ast.Starred):
        self.visit(node.value)

    # expressions

    def visit_unary_op(self, node: ast.UnaryOp):
        self.visit(node.operand)

    def visit_bin_op(self, node: ast.BinOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_bool_op(self, node: ast.BoolOp):
        self._visit_sequence(node.values)

    def visit_compare(self, node: ast.Compare):
        self.visit(node.left)
        self._visit_sequence(node.comparators)

    def visit_call(self, node: ast.Call):
        self.visit(node.func)
        self._visit_sequence(node.args)
        self._visit_sequence(node.keywords)
        self._visit_valid(getattr(node, 'starargs', None))
        self._visit_valid(getattr(node, 'kwargs', None))

    def visit_keyword(self, node):
        self.visit(node.value)

    def visit_if_exp(self, node: ast.IfExp):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_attribute(self, node: ast.Attribute):
        assert isinstance(node.ctx, ast.Load)
        self.visit(node.value)

    # subscripting

    def visit_subscript(self, node: ast.Subscript):
        assert isinstance(node.ctx, ast.Load)
        self.visit(node.value)
        self.visit(node.slice)

    def visit_index(self, node):
        self.visit(node.value)

    def visit_slice(self, node):
        self._visit_valid(node.lower)
        self._visit_valid(node.upper)
        self._visit_valid(node.step)

    def visit_ext_slice(self, node):
        self._visit_sequence(node.dims)

    # comprehensions

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

    # statements

    def visit_assertion_wrapper(self, node: AssertionWrapper):
        self.visit(node.assertion.test)
        if node.assertion.msg is not None:
            self.visit(node.assertion.msg)

    # imports

    visit_unified_import = _ignore_node

    # functions

    @change_source
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
            if self.not_visited(value):
                self.add_message('Definition is never used', 'at %d:%d' % value[0].position[:2])

        self.leave_scope()

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
