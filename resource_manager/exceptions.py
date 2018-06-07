import sys
from contextlib import contextmanager

from .structures import *

IGNORE_IN_TRACEBACK = (
    Binary, Unary, Parenthesis, Slice, Starred, Array, Tuple, Dictionary
)


@contextmanager
def modify_traceback(change_exception):
    def custom_handler(cls, exception, traceback):
        return default_handler(cls, change_exception(exception), traceback)

    default_handler = sys.excepthook
    sys.excepthook = custom_handler
    yield
    sys.excepthook = default_handler


def join_traceback(exception: Exception):
    current = exception
    while current:
        if type(current) is RenderError:
            current.__traceback__ = None

            while type(current.__cause__) is RenderError:
                cause = current.__cause__
                current.update_definitions(cause.definitions)
                current.__cause__ = cause.__cause__

        current = current.__cause__

    return exception


def replace(exception: Exception, from_type, to_type):
    current = exception
    previous = None
    while current:
        if type(current) is from_type:
            temp = to_type(*current.args)
            temp.__cause__ = current.__cause__
            current = temp
            if previous is None:
                exception = temp
            else:
                previous.__cause__ = temp
            temp.__traceback__ = None

        previous = current
        current = current.__cause__

    return exception


def remove_traceback(exception: Exception):
    current = exception
    while current:
        if type(current) is BadSyntaxError:
            current.__traceback__ = None
        current = current.__cause__
    return exception


class BadSyntaxError(RuntimeError):
    pass


class RenderError(RuntimeError):
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
                line = definition.line().rstrip()
                stack.append('\n  at %d:%d in %s\n    ' % definition.position() + line)
            last_position = position

        definition = self.definitions[-1]
        return 'An exception occurred while ' + definition.error_message() + ''.join(reversed(stack))
