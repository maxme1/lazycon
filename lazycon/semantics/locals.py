import ast

from .visitor import SemanticVisitor


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


def extract_assign_targets(targets):
    def _extract(target):
        assert isinstance(target.ctx, ast.Store)

        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Starred):
            yield from _extract(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from _extract(elt)

        else:
            assert False, 'unreachable code'

    result = set()
    for t in targets:
        result.update(_extract(t))
    return result
