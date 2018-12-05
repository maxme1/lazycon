import re

from .token import TokenWrapper

first_cap = re.compile('(.)([A-Z][a-z]+)')
MAX_COLUMNS = 100


class Structure:
    def __init__(self, main_token: TokenWrapper):
        self.main_token = main_token

    def render(self, walker):
        name = first_cap.sub(r'\1_\2', self.__class__.__name__).lower()
        return getattr(walker, '_render_' + name)(self)

    def position(self):
        return self.line, self.main_token.column, self.source

    @property
    def source(self):
        return self.main_token.source or '<string input>'

    @property
    def line(self):
        return self.main_token.line
