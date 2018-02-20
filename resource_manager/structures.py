import re

from .token import TokenWrapper

first_cap = re.compile('(.)([A-Z][a-z]+)')
all_cap = re.compile('([a-z0-9])([A-Z])')
MAX_COLUMNS = 60


def snake_case(name):
    name = first_cap.sub(r'\1_\2', name)
    return all_cap.sub(r'\1_\2', name).lower()


class Structure:
    def __init__(self, main_token: TokenWrapper):
        self.main_token = main_token

    def render(self, walker):
        name = snake_case(self.__class__.__name__)
        return getattr(walker, '_render_' + name)(self)

    def to_str(self, level):
        raise NotImplementedError

    def error_message(self):
        return 'building the resource ' + self.to_str(0)

    def position(self):
        return self.main_token.line, self.main_token.column, self.source()

    def source(self):
        return self.main_token.source or '<string input>'

    def level(self, level):
        return '    ' * level


from .expressions import *
from .statements import *
