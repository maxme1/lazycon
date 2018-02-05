from .token import Token


class Structure:
    def __init__(self, main_token: Token):
        self.main_token = main_token

    def render(self, interpreter):
        raise NotImplementedError

    def to_str(self, level):
        raise NotImplementedError

    def error_message(self):
        return 'building the resource ' + self.to_str(0)

    def position(self):
        return self.main_token.line, self.main_token.column, self.main_token.source

    def source(self):
        return self.main_token.source


from .expressions import *
from .statements import *
