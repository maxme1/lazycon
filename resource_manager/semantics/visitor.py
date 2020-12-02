import ast
from typing import Iterable

from ..wrappers import AssertionWrapper, ExpressionStatement
from ..visitor import Visitor


class SemanticVisitor(Visitor):
    """Simple visitor for nodes that don't interact with the scope stack."""

    # utils

    def _visit_sequence(self, sequence: Iterable):
        for item in sequence:
            self.visit(item)

    def _visit_valid(self, value):
        if value is not None:
            self.visit(value)

    def _ignore_node(self, node):
        pass

    # expressions

    def visit_expression_statement(self, node: ExpressionStatement):
        self.visit(node.expression)

    visit_pattern_assignment = visit_expression_wrapper = visit_expression_statement

    # literals

    visit_constant = visit_name_constant = visit_ellipsis = visit_bytes = visit_num = visit_str = _ignore_node

    def visit_formatted_value(self, node):
        assert node.format_spec is None
        self.visit(node.value)

    def visit_joined_str(self, node):
        self._visit_sequence(node.values)

    def visit_list(self, node: ast.List):
        assert isinstance(node.ctx, ast.Load)
        self._visit_sequence(node.elts)

    visit_tuple = visit_list

    def visit_set(self, node):
        self._visit_sequence(node.elts)

    def visit_dict(self, node):
        self._visit_sequence(filter(None, node.keys))
        self._visit_sequence(node.values)

    # variables

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

    # statements

    def visit_assertion_wrapper(self, node: AssertionWrapper):
        self.visit(node.assertion.test)
        if node.assertion.msg is not None:
            self.visit(node.assertion.msg)

    # imports

    visit_unified_import = _ignore_node
