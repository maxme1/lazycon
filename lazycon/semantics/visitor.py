import ast

from ..visitor import Visitor


class SemanticVisitor(Visitor):
    """Simple visitor for nodes that don't interact with the scope stack."""

    # utils

    def _visit_valid(self, value):
        if value is not None:
            self.visit(value)

    def _ignore_node(self, node):
        pass

    # literals

    visit_constant = visit_name_constant = visit_ellipsis = visit_bytes = visit_num = visit_str = _ignore_node

    def visit_formatted_value(self, node):
        assert node.format_spec is None
        self.visit(node.value)

    def visit_joined_str(self, node):
        self._iterate_nodes(node.values)

    def visit_list(self, node: ast.List):
        assert isinstance(node.ctx, ast.Load)
        self._iterate_nodes(node.elts)

    visit_tuple = visit_list

    def visit_set(self, node):
        self._iterate_nodes(node.elts)

    def visit_dict(self, node):
        self._iterate_nodes(filter(None, node.keys))
        self._iterate_nodes(node.values)

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
        self._iterate_nodes(node.values)

    def visit_compare(self, node: ast.Compare):
        self.visit(node.left)
        self._iterate_nodes(node.comparators)

    def visit_call(self, node: ast.Call):
        self.visit(node.func)
        self._iterate_nodes(node.args)
        self._iterate_nodes(node.keywords)
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
        self._iterate_nodes(node.dims)

    # statements

    def visit_assert(self, node):
        self.visit(node.test)
        self._visit_valid(node.msg)

    def visit_if(self, node: ast.If):
        self.visit(node.test)
        self._iterate_nodes(node.body)
        self._iterate_nodes(node.orelse)

    def visit_expr(self, node: ast.Expr):
        self.visit(node.value)

    # helpers

    def visit_arguments(self, node: ast.arguments):
        self._iterate_nodes(node.defaults)
        self._iterate_nodes(filter(None, node.kw_defaults))
