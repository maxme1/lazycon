import unittest
from pathlib import Path

import pytest

from resource_manager import read_config
from resource_manager.parser import parse_string

TESTS_PATH = Path(__file__).parent


def standardize(source):
    parents, imports, definitions = parse_string(source)
    result = '\n'.join(imp.to_str() for imp in parents) + '\n'
    result += '\n'.join(imp.to_str([name]) for name, imp in imports) + '\n'
    for name, definition in definitions:
        result += definition.to_str([name]) + '\n'
    return result


class TestParser(unittest.TestCase):
    def test_idempotency(self):
        for path in TESTS_PATH.glob('**/*.config'):
            with self.subTest(filename=path.name):
                with open(path, 'r') as file:
                    source = file.read()
                temp = standardize(source)
                assert temp == standardize(temp)

    def test_comments(self):
        config = read_config(TESTS_PATH / 'statements/comments.config')

        with open(TESTS_PATH / 'statements/no_comments.config', 'r') as file:
            source = file.read()

        assert source == config.render_config()

    def test_unexpected_token(self):
        with pytest.raises(SyntaxError):
            parse_string('a = [1, 2 3]')

    def test_unexpected_eof(self):
        with pytest.raises(SyntaxError):
            parse_string('a = [1, 2')

    def test_unrecognized_token(self):
        with pytest.raises(SyntaxError):
            parse_string('$')
