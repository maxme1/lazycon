from collections import defaultdict
from inspect import Parameter
from typing import Iterable, Dict

from resource_manager.scope import ScopeDict
from .visitor import Visitor
from .wrappers import Wrapper
from .exceptions import BuildConfigError
from .token import TokenType, INVALID_STRING_PREFIXES


class SyntaxTree(Visitor):
    def __init__(self, name_to_node: ScopeDict, builtins: Iterable[str]):
        self.messages = defaultdict(lambda: defaultdict(set))
        self._scopes = []
        self._builtins = builtins
        self.node_levels = {}
        self.node_to_names = defaultdict(list)
        for name, node in name_to_node.items():
            self.node_to_names[node].append(name)
        self.node_to_names = dict(self.node_to_names)

        self.enter_scope(name_to_node)
        self.visit_current_scope()

    def add_message(self, message, node, content):
        self.messages[message][node.source].add(content)

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
        return tree.node_levels

    def generic_visit(self, node):
        # TODO: improve error message
        raise SyntaxError
    #
    # def enter_scope(self, names: ScopeDict, visited=()):
    #     scope = {name: [value, None] for name, value in names.items()}
    #     for name in visited:
    #         scope[name][1] = True
    #     self._scopes.append(scope)
    #
    # def leave_scope(self):
    #     self._scopes.pop()
    #
    # def not_visited(self, value):
    #     return value[1] is None
    #
    # def entered(self, value):
    #     return value[1] is False
    #
    # def mark_as_visited(self, value):
    #     node = value[0]
    #     if node in self.node_to_names:
    #         assert len(self._scopes) == 1
    #         scope = self._scopes[0]
    #         for name in self.node_to_names[node]:
    #             scope[name][1] = True
    #         assert value[1]
    #     else:
    #         value[1] = True
    #
    # def visit_resource(self, value, level=0):
    #     assert value[1] is None
    #     n = len(self._scopes) - level
    #     self._scopes, tail = self._scopes[:n], self._scopes[n:]
    #     value[1] = False
    #     node = value[0]
    #
    #     # allowing recursion
    #     if isinstance(node, Function):
    #         self.mark_as_visited(value)
    #         node.render(self)
    #     else:
    #         node.render(self)
    #         self.mark_as_visited(value)
    #
    #     self._scopes.extend(tail)
    #
    # def visit_current_scope(self):
    #     for value in self._scopes[-1].values():
    #         if self.not_visited(value):
    #             self.visit_resource(value)
    #
    # def _render_sequence(self, sequence):
    #     for item in sequence:
    #         item.render(self)
    #
    # def _render_expression_statement(self, node: ExpressionStatement):
    #     node.expression.render(self)
    #
    # def _render_expression_wrapper(self, node: ExpressionWrapper):
    #     node.expression.render(self)
    #
    # def _render_resource(self, node: Resource):
    #     name = node.name
    #     for level, scope in enumerate(reversed(self._scopes)):
    #         if name in scope:
    #             if self.not_visited(scope[name]):
    #                 self.visit_resource(scope[name], level)
    #             elif self.entered(scope[name]):
    #                 # TODO: more information
    #                 self.add_message('Resources are referenced before being completely defined',
    #                                  node, '"' + name + '" at %d:%d' % node.position()[:2])
    #             return
    #
    #     if name not in self._builtins:
    #         self.add_message('Undefined resources found, but are required', node, name)
    #
    # def _render_get_attribute(self, node: GetAttribute):
    #     node.target.render(self)
    #
    # def _render_slice(self, node: Slice):
    #     for arg in node.args:
    #         if arg is not None:
    #             arg.render(self)
    #
    # def _render_get_item(self, node: GetItem):
    #     node.target.render(self)
    #     self._render_sequence(node.args)
    #
    # def _render_call(self, node: Call):
    #     names = [arg.name.body for arg in node.kwargs if not isinstance(arg, VariableKeywordArgument)]
    #     if len(set(names)) < len(names):
    #         self.add_message('Duplicate keyword arguments', node, 'at %d:%d' % node.position()[:2])
    #
    #     node.target.render(self)
    #     for param in node.args + node.kwargs:
    #         param.value.render(self)
    #
    # def _render_starred(self, node: Starred):
    #     node.expression.render(self)
    #
    # def _render_array(self, node: Array):
    #     self._render_sequence(node.entries)
    #
    # def _render_tuple(self, node):
    #     self._render_array(node)
    #
    # def _render_set(self, node):
    #     self._render_array(node)
    #
    # def _render_dictionary(self, node: Dictionary):
    #     for key, value in node.entries:
    #         key.render(self)
    #         value.render(self)
    #
    # def _render_parenthesis(self, node: Parenthesis):
    #     node.expression.render(self)
    #
    # def _render_lambda(self, node):
    #     self._render_function(node)
    #
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
    #
    # def _render_unified_import(self, node: UnifiedImport):
    #     pass
    #
    # def _render_literal(self, node: Literal):
    #     if node.value.type == TokenType.STRING and node.value.body.startswith(INVALID_STRING_PREFIXES):
    #         self.add_message('Inline string formatting is not supported', node, 'at %d:%d' % node.position()[:2])
    #
    # def _render_binary(self, node: Binary):
    #     node.left.render(self)
    #     node.right.render(self)
    #
    # def _render_unary(self, node: Unary):
    #     node.argument.render(self)
    #
    # def _render_inline_if(self, node: InlineIf):
    #     self._render_sequence([node.condition, node.left, node.right])
