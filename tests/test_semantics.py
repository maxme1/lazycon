import builtins

import pytest

from lazycon.exceptions import SemanticError
from lazycon.parser import parse_string
from lazycon.semantics import Semantics


def test_correctness(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(path=path):
            with open(path, 'r') as file:
                parents, scope = parse_string(file.read(), '.config')
                if not parents:
                    Semantics(scope, set(vars(builtins))).check()


def validate_config(source: str):
    parents, scope = parse_string(source, '.config')
    assert not parents
    Semantics(scope, set(vars(builtins))).check()


def test_undefined():
    with pytest.raises(SemanticError):
        validate_config('''
x = 1
z = x + y
        ''')
    with pytest.raises(SemanticError):
        validate_config('''
def f(x):
    return x + y
        ''')
    with pytest.raises(SemanticError):
        validate_config('''
def f(x):
    def h(z):
        return x + y
        ''')
    with pytest.raises(SemanticError):
        validate_config('''
def f(x=y+1):
    return x
        ''')


def test_late_binding():
    validate_config('''
y = 1        
def f(x):
    return x + y
    ''')
    validate_config('''
def f(x):
    def h():
        return x + y
    y = 1
    ''')
