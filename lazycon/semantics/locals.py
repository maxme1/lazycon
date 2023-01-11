import ast

from .visitor import SemanticVisitor
from ..parser import extract_assign_targets


class LocalsGatherer(SemanticVisitor):
    def __init__(self, source_path):
        self.source_path = source_path
        self.names = []

    @classmethod
    def gather(cls, nodes, source_path):
        instance = cls(source_path)
        instance._iterate_nodes(nodes)
        return instance.names

    def visit_function_def(self, node: ast.FunctionDef):
        self.names.append(node.name)

    def visit_assign(self, node: ast.Assign):
        self.names.extend(extract_assign_targets(node.targets, self.source_path))

    def visit_with(self, node: ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                self.names.extend(extract_assign_targets([item.optional_vars], self.source_path))

        self._iterate_nodes(node.body)

    def generic_visit(self, node: ast.AST, *args, **kwargs):
        pass
