import sys
from contextlib import contextmanager

from .structures import *


@contextmanager
def ignore_traceback():
    def custom_handler(cls, instance: Exception, traceback):
        cause = instance
        while cause:
            if type(cause) is RenderError:
                cause.__traceback__ = None
            cause = cause.__cause__

        return default_handler(cls, instance, None)

    default_handler = sys.excepthook
    sys.excepthook = custom_handler
    yield
    sys.excepthook = default_handler


class RenderError(RuntimeError):
    def __init__(self, definitions):
        self.definitions = tuple(definitions)
        super().__init__(self.get_str())

    def get_str(self):
        return self.get_message() + self.get_traceback(self.definitions)

    def get_message(self):
        definition = self.definitions[-1]
        return 'An exception occurred while ' + definition.error_message()

    @staticmethod
    def get_traceback(definitions):
        stack = []
        last_position = None
        for definition in reversed(definitions):
            position = definition.position()
            position = position[0], position[2]

            if last_position is None or (
                    position != last_position and not isinstance(definition, IGNORE_IN_TRACEBACK)):
                line = definition.line().rstrip()
                stack.append('\n  at %d:%d in %s\n    ' % definition.position() + line)
            last_position = position
        return ''.join(reversed(stack))
