import re

from .token import TokenWrapper

first_cap = re.compile('(.)([A-Z][a-z]+)')
MAX_COLUMNS = 60


class Structure:
    def __init__(self, main_token: TokenWrapper):
        self.main_token = main_token

    def render(self, walker):
        name = first_cap.sub(r'\1_\2', self.__class__.__name__).lower()
        return getattr(walker, '_render_' + name)(self)

    def to_str(self, level):
        raise NotImplementedError

    def error_message(self):
        return 'building the resource ' + self.to_str(0)

    def position(self):
        return self.main_token.line, self.main_token.column, self.source()

    def source(self):
        return self.main_token.source or '<string input>'

    def line(self):
        return self.main_token.token_line

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' ' + self.to_str(0) + '>'


from .expressions import *
from .statements import *
