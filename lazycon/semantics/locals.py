import ast

from .visitor import SemanticVisitor
from ..parser import extract_assign_targets


class LocalsGatherer(SemanticVisitor):
    def __init__(self):
        self.names = []

    @classmethod
    def gather(cls, nodes):
        instance = cls()
        instance._iterate_nodes(nodes)
        return instance.names

    def visit_function_def(self, node: ast.FunctionDef):
        self.names.append(node.name)

    def visit_assign(self, node: ast.Assign):
        self.names.extend(extract_assign_targets(node.targets))

    def generic_visit(self, node: ast.AST, *args, **kwargs):
        pass
