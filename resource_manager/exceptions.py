import sys
from contextlib import contextmanager

from .structures import *


@contextmanager
def ignore_traceback():
    #         last_instance = cause = instance
    #         add_message = True
    #         messages = []
    #         while cause:
    #             if hasattr(cause, '__render_error__'):
    #                 cause.__traceback__ = None
    #             else:
    #                 add_message = False
    #
    #             if add_message:
    #                 messages.append(cause.args[0])
    #                 last_instance = cause
    #
    #             cause = cause.__cause__
    #
    #         last_instance.args = '\n'.join(reversed(messages)),
    #         return default_handler(cls, last_instance, traceback)
    def custom_handler(cls, instance: Exception, traceback):
        cause = instance
        while cause:
            # if hasattr(cause, '__render_error__'):
            if type(cause) is RenderError:
                cause.__traceback__ = None
            cause = cause.__cause__

        return default_handler(cls, instance, traceback)

    default_handler = sys.excepthook
    sys.excepthook = custom_handler
    yield
    sys.excepthook = default_handler


class RenderError(RuntimeError):
    def __init__(self, definitions):
        super().__init__()
        self.definitions = tuple(definitions)
        self.args = (self.get_str(),)

    def append_definitions(self, definitions):
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
                line = definition.line()
                if line[-1] == '\n':
                    line = line[:-1]
                stack.append('\n  at %d:%d in %s\n    ' % definition.position() + line)
            last_position = position
        message = ''.join(reversed(stack))
        definition = self.definitions[-1]

        return 'An exception occurred while ' + definition.error_message() + message
