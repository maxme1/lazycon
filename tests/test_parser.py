from functools import partial

import pytest

from lazycon import load
from lazycon.parser import parse_string
from lazycon.scope import render_scope

parse_string = partial(parse_string, extension='.config')


def standardize(source):
    parents, scope = parse_string(source)
    result = '\n'.join(f'from {"." * imp.dots}{".".join(imp.root)} import *' for imp in parents) + '\n'
    return result + '\n'.join(render_scope(scope, {v: i for i, v in enumerate(scope)})) + '\n'


def test_idempotency(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(filename=path.name):
            with open(path, 'r') as file:
                source = file.read()
            temp = standardize(source)
            assert temp == standardize(temp)


def test_comments(tests_path):
    config = load(tests_path / 'statements/comments.config')
    with open(tests_path / 'statements/no_comments.config', 'r') as file:
        source = file.read()

    assert source == config.dumps()


def test_unexpected_token():
    with pytest.raises(SyntaxError):
        parse_string('a = [1, 2 3]')


def test_unexpected_eof():
    with pytest.raises(SyntaxError):
        parse_string('a = [1, 2')


def test_unrecognized_token():
    with pytest.raises(SyntaxError):
        parse_string('$')


def test_unsupported_assignment():
    with pytest.raises(SyntaxError):
        parse_string('x = 0; x[1] = 2')


def test_unsupported_syntax():
    with pytest.raises(SyntaxError, match='This syntactic structure is not supported'):
        parse_string('for i in range(100): pass')
