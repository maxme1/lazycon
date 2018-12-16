from collections import defaultdict

from resource_manager.scope import ScopeDict
from .visitor import Visitor
from .wrappers import *
from .exceptions import BuildConfigError


def change_source(method):
    def wrapper(self, node, *args, **kwargs):
        self._source_paths.append(node.source_path)
        value = method(self, node, *args, **kwargs)
        self._source_paths.pop()
        return value

    return wrapper


class SyntaxTree(Visitor):
    def __init__(self, name_to_node: ScopeDict, builtins: Iterable[str]):
        self.messages = defaultdict(lambda: defaultdict(set))
        self._scopes = []
        self._source_paths = []
        self._builtins = builtins

        self.node_to_names = defaultdict(list)
        for name, node in name_to_node.items():
            self.node_to_names[node].append(name)
        self.node_to_names = dict(self.node_to_names)

        self.enter_scope(name_to_node)
        self.analyze_current_scope()

    def add_message(self, message, content):
        assert self._source_paths
        self.messages[message][self._source_paths[-1]].add(content)

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
        tree = SyntaxTree(scope, builtins)
        message = ''
        for msg, elements in tree.messages.items():
            message += tree.format(msg, elements)
        if message:
            raise BuildConfigError(message)

    def enter_scope(self, names: ScopeDict, visited=()):
        scope = {name: [value, None] for name, value in names.items()}
        for name in visited:
            scope[name][1] = True
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
            assert value[1]
        else:
            value[1] = True

    def mark_name(self, value, level=0):
        assert value[1] is None
        n = len(self._scopes) - level
        self._scopes, tail = self._scopes[:n], self._scopes[n:]
        value[1] = False
        node = value[0]

        # allowing recursion
        if isinstance(node, (Function, ast.Lambda)):
            self._mark_name(value)
            self.visit(node)
        else:
            self.visit(node)
            self._mark_name(value)

        self._scopes.extend(tail)

    def analyze_current_scope(self):
        for value in self._scopes[-1].values():
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
    def visit_expression_wrapper(self, node: ExpressionWrapper):
        self.visit(node.expression)

    # literals

    visit_name_constant = visit_ellipsis = visit_bytes = visit_num = visit_str = _ignore_node

    def visit_formatted_value(self, node):
        raise NotImplementedError

    def visit_joined_str(self, node):
        raise NotImplementedError

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
                    # TODO: more information
                    self.add_message('Resources are referenced before being completely defined',
                                     '"' + name + '" at %d:%d' % (node.lineno, node.col_offset))
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
        # TODO:
        #     names = [arg.name.body for arg in node.kwargs if not isinstance(arg, VariableKeywordArgument)]
        #     if len(set(names)) < len(names):
        #         self.add_message('Duplicate keyword arguments', node, 'at %d:%d' % node.position()[:2])

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

    # TODO: implement comprehensions
    def visit_list_comp(self, node):
        raise NotImplementedError

    # imports

    visit_unified_import = _ignore_node

    # functions

    @change_source
    def visit_function(self, node: Function):
        pass

    def visit_lambda(self, node):
        pass

    # def _render_function(self, node: Function):
    #     bindings = {name: binding for name, binding in node.bindings}
    #     if len(bindings) != len(node.bindings):
    #         self.add_message('Duplicate binding names in function definition', node, 'at %d:%d' % node.position()[:2])
    #
    #     names = {parameter.name: None for parameter in node.signature.parameters.values()}
    #
    #     if set(names) & set(bindings):
    #         self.add_message('Binding names clash with argument names in function definition',
    #                          node, 'at %d:%d' % node.position()[:2])
    #
    #     for parameter in node.signature.parameters.values():
    #         if parameter.default is not Parameter.empty:
    #             parameter.default.render(self)
    #
    #     bindings.update(names)
    #     self.enter_scope(bindings, names)
    #     node.expression.render(self)
    #     self.visit_current_scope()
    #     self.leave_scope()
