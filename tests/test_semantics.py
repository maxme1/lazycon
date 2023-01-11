import builtins
from functools import partial

import pytest

from lazycon.exceptions import SemanticError
from lazycon.parser import parse_string
from lazycon.semantics import Semantics

parse_string = partial(parse_string, extension='.config')


def test_correctness(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(path=path):
            with open(path, 'r') as file:
                parents, scope = parse_string(file.read())
                if not parents:
                    Semantics(scope, set(vars(builtins))).check()


def validate_config(source: str):
    parents, scope = parse_string(source)
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


def test_wildcard():
    with pytest.raises(SemanticError):
        validate_config('x, _ = 1, 1; a = _, 2')


def test_unsupported_assignment():
    with pytest.raises(SyntaxError):
        validate_config('def f(x): x[1] = 2')


def test_unsupported_structure():
    with pytest.raises(SemanticError):
        validate_config('''
def f(x): 
    while True: 
        pass
''')


def test_async_comprehension():
    with pytest.raises((SemanticError, SyntaxError)):
        validate_config('y = (x async for x in [])')


def test_local_unbound():
    with pytest.raises(SemanticError, match='Local variables referenced before being defined'):
        validate_config('def f(): y = x + 1; x = 2')


def test_circular_dependency():
    with pytest.raises(SemanticError, match='Values are referenced before being completely defined'):
        validate_config('x = x + 1')
