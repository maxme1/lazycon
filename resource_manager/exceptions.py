import sys
from contextlib import contextmanager

from .structures import *

IGNORE_IN_TRACEBACK = (
    Binary, Unary, Parenthesis, Slice, Starred, Array, Tuple, Dictionary
)


class NoCause:
    pass


def custom_raise(exception, cause=NoCause):
    with modify_traceback():
        if cause is NoCause:
            raise exception
        else:
            raise exception from cause


@contextmanager
def modify_traceback():
    def custom_handler(cls, exception: Exception, traceback):
        current = exception
        while current:
            if isinstance(current, ModifiedException):
                current.__traceback__ = None
                current.update_self()
            current = current.__cause__

        return default_handler(cls, exception, traceback)

    default_handler = sys.excepthook
    sys.excepthook = custom_handler
    yield
    sys.excepthook = default_handler


class ModifiedException(Exception):
    def update_self(self):
        pass


class BadSyntaxError(ModifiedException):
    pass


class LambdaArgumentsError(ModifiedException):
    pass


class BuildConfigError(ModifiedException):
    pass


class RenderError(ModifiedException):
    def __init__(self, definitions):
        self.definitions = tuple(definitions)
        super().__init__(self.get_str())

    def update_definitions(self, definitions):
        self.definitions = self.definitions + tuple(definitions)
        self.args = (self.get_str(),)

    def get_str(self):
        stack = []
        last_position = None
        for definition in reversed(self.definitions):
            position = definition.position()
            position = position[0], position[2]

            if last_position is None or (
                    position != last_position and not isinstance(definition, IGNORE_IN_TRACEBACK)):
                stack.append('\n  at %d:%d in %s\n    ' % definition.position() + definition.line())
            last_position = position

        definition = self.definitions[-1]
        return 'An exception occurred while ' + definition.error_message() + ''.join(reversed(stack))

    def update_self(self):
        while type(self.__cause__) is RenderError:
            cause = self.__cause__
            self.update_definitions(cause.definitions)
            self.__cause__ = cause.__cause__
