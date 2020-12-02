import pytest

from resource_manager import read_config
from resource_manager.parser import parse_string


def standardize(source):
    parents, imports, definitions = parse_string(source)
    result = '\n'.join(imp.to_str() for imp in parents) + '\n'
    result += '\n'.join(imp.to_str([name]) for name, imp in imports) + '\n'
    for name, definition in definitions:
        result += definition.to_str([name]) + '\n'
    return result


def test_idempotency(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(filename=path.name):
            with open(path, 'r') as file:
                source = file.read()
            temp = standardize(source)
            assert temp == standardize(temp)


def test_comments(tests_path):
    config = read_config(tests_path / 'statements/comments.config')
    with open(tests_path / 'statements/no_comments.config', 'r') as file:
        source = file.read()

    assert source == config.render_config()


def test_unexpected_token():
    with pytest.raises(SyntaxError):
        parse_string('a = [1, 2 3]')


def test_unexpected_eof():
    with pytest.raises(SyntaxError):
        parse_string('a = [1, 2')


def test_unrecognized_token():
    with pytest.raises(SyntaxError):
        parse_string('$')
