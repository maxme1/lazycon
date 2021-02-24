import ast
import re

first_cap = re.compile('(.)([A-Z][a-z]+)')
all_cap = re.compile('([a-z0-9])([A-Z])')


def snake_case(name):
    name = first_cap.sub(r'\1_\2', name)
    return all_cap.sub(r'\1_\2', name).lower()


class Visitor:
    def visit(self, node: ast.AST, *args, **kwargs):
        method = 'visit_' + snake_case(node.__class__.__name__)
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: ast.AST, *args, **kwargs):
        raise NotImplementedError(node.__class__.__name__)
