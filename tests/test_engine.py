import contextlib

import numpy as np
import pytest

from lazycon.exceptions import EntryError, SemanticError
from lazycon import Config, load, loads


def test_string_input():
    rm = Config()
    rm.string_input('''
a = [1,2,3]
b = sum(a)
''')
    assert rm.b == 6


def test_dumps(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(path=path):
            config = load(path)
            # for each entry point the config must be readable and give the same results
            for name in list(config) + [None, []]:
                source = config.dumps(name)
                assert config.dumps(name) == source, path
                assert loads(source).dumps() == source, path

            with pytest.raises(ValueError):
                config.dumps('~definitely-missing~')


def test_import(inside_tests):
    rm = load('imports/imports.config')
    assert rm.numpy == np
    rm.r
    rm.os
    rm.std
    rm.mean
    rm = loads('import numpy.linalg; inv = numpy.linalg.inv')
    rm.inv


def test_update():
    rm = loads('a = 1').update(a=2)
    rm2 = loads('a = 1').string_input('a = 2')
    assert rm.a == rm2.a

    rm = loads('a = 1').update(a=2).string_input('a = 3')
    assert rm.a == 3

    with pytest.raises(RuntimeError):
        rm = loads('a = 1')
        rm.a
        rm.update(a=2)

    rm = loads('a = 1')
    assert rm.a == 1
    cp = rm.copy().update(a=2)
    assert rm.a == 1
    assert cp.a == 2


def test_multiple_definitions(inside_tests):
    rm = load('statements/multiple_definitions.config')
    assert '''a = b = c = 1\nd = a\n''' == rm.dumps()
    rm = load('statements/import_from_multiple.config')
    assert '''a = 2\nb = c = 1\nd = a\n''' == rm.dumps()


@pytest.mark.xfail
def test_same_line():
    cf = loads('a = 1; b = 2; c = a + b')
    assert cf.c == 3
    assert cf.dumps() == 'a = 1\nb = 2\nc = a + b'


def test_cycle_import(inside_tests):
    try:
        load('imports/cycle_import.config')
    except RecursionError:
        pytest.fail()


def test_inheritance_order(inside_tests):
    rm = load('imports/order1.config')
    assert rm.literals is not None
    rm = load('imports/order2.config')
    assert rm.literals is None


def test_import_in_string(inside_tests):
    rm = loads('from .expressions.literals import *')
    assert rm.literals[0]


def test_upper_import(inside_tests):
    rm = load('imports/folder/upper_import.config')
    assert 'just override os' == rm.os
    assert np == rm.numpy


def test_attr_error():
    rm = Config()
    with pytest.raises(AttributeError):
        rm.undefined_value
    with pytest.raises(KeyError):
        rm['undefined_value']
    with pytest.raises(EntryError):
        rm.get('undefined_value')


def test_items(inside_tests):
    rm = load('expressions/tail.config')
    assert rm.part == 6
    assert rm.value == (1, 2)
    np.testing.assert_array_equal(rm.another_part, [1, 2, 3])


def test_tail(inside_tests):
    rm = load('expressions/tail.config')
    np.testing.assert_array_equal(rm.mean, [2, 5])
    np.testing.assert_array_equal(rm.mean2, [2, 5])
    np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
    assert rm.random.shape == (1, 1, 2, 2)


def test_local_duplicates():
    assert loads('''
def f():
    x = 1
    x = 2
    return x
    
def g(x):
    x = 1
    return 2
    ''').f() == 2


def test_func_def(inside_tests):
    rm = load('statements/funcdef.config')
    assert rm.f() == 1
    assert rm.inc_first(['a', 'b', 'c']) == (2, 1, 1)
    assert rm.qsort([2, 1, 3, 10, 10]) == [1, 2, 3, 10, 10]
    assert rm.returner(2)() == 2
    assert rm.h_with_defaults(0, n=3) == 1
    assert rm.doc.__doc__ == 'docstring'


def test_assertions(inside_tests):
    rm = load('statements/funcdef.config')
    assert rm.assertion(True)
    with pytest.raises(AssertionError):
        rm.assertion(False)


def test_unpacking(inside_tests):
    rm = load('statements/funcdef.config')
    assert rm.unpack([1, 2]) == 3
    assert rm.nested_unpack([1, [2, 3]]) == (1, 2, 3)
    assert rm.deep_unpack([[[[[[1]]]]], 2]) == (1, 2)
    assert rm.single_unpack([[[1]]]) == [[1]]
    with pytest.raises(TypeError):
        rm.unpack(1)
    with pytest.raises(ValueError):
        rm.unpack([1])
    with pytest.raises(ValueError):
        rm.unpack([1, 2, 3])
    assert loads('def f(x): a, *b = x; return a').f((1, 2, 3)) == 1

    with pytest.raises(SyntaxError):
        loads('_ = 1, 2')
    with pytest.raises(SyntaxError):
        loads('_, _ = 1, 2')
    with pytest.raises(SyntaxError):
        loads('a, a = 1, 2')


def test_wildcard_overwrite():
    cf = loads('a, b = 1, 2')
    assert cf.dumps(entry_points=['b']) == '_, b = 1, 2\n'

    cf = cf.string_input('a = 5')
    assert cf.dumps() == 'a = 5\n_, b = 1, 2\n'

    cf = cf.string_input('b = 6')
    assert cf.dumps() == 'a = 5\nb = 6\n'

    assert loads('a, _ = b = 1, 2').dumps(entry_points=['b']) == 'b = 1, 2\n'


def test_wildcards():
    cf = loads('x, y, z = range(3)\nt = x + y + z')
    assert (cf.x, cf.y, cf.z) == (0, 1, 2)
    assert cf.t == 3
    cf = loads('x, y, z = range(3)\nt = x + y + z').string_input('y = 5')
    assert (cf.x, cf.y, cf.z) == (0, 5, 2)
    assert cf.t == 7
    assert cf.dumps() == 'x, _, z = range(3)\ny = 5\nt = x + y + z\n'
    cf = loads('x, *y, z = range(4)')
    assert (cf.x, cf.y, cf.z) == (0, [1, 2], 3)
    cf = loads('x, *y, z = range(4)').string_input('y = 2')
    assert (cf.x, cf.y, cf.z) == (0, 2, 3)
    assert cf.dumps() == 'x, *_, z = range(4)\ny = 2\n'

    # iterables
    cf = loads('x, y, z = map(int, range(3))\nt = x + y + z')
    assert (cf.x, cf.y, cf.z) == (0, 1, 2)
    assert cf.t == 3
    with pytest.raises(ValueError):
        loads('x, y, z = a, b, c = map(int, range(3))\nt = x').t
    with pytest.raises(ValueError):
        loads('x, y, z = a, b, c = map(int, range(3))\nt = a + x').t


def test_decorators(inside_tests):
    rm = load('statements/funcdef.config')
    assert rm.one(0) == 1
    assert rm.two(0) == 2
    assert rm.three(0) == 3
    assert rm.order() == tuple(range(5))


def test_render_decorators(inside_tests):
    rm = load('statements/decorated.config')
    with open('statements/decorated.config') as file:
        assert rm.dumps() == file.read()


def test_lambda(inside_tests):
    rm = load('expressions/lambda_.config')
    assert rm.b(2) == 8
    assert rm.c(2, rm.a) == 8
    assert rm.d(1)(2) == [1, 2]
    assert rm.e() == 8
    assert rm.test == [1, 8, 32]
    assert rm.vararg(1, 2, 3) == (2, 3)
    assert rm.only_vararg(1) == (1,)


def test_lambda_args(inside_tests):
    rm = load('expressions/lambda_.config')
    assert rm.with_default() == (1, 2)
    assert rm.keyword(y=1) == ((), 1)
    with pytest.raises(TypeError):
        rm.b(1, 2)
    rm.vararg(x=1)


def test_eval(inside_tests):
    rm = load('statements/funcdef.config')
    assert rm.eval('f') == rm.f
    assert rm.eval('f()') == 1
    assert rm.eval('qsort([4,2,1,3])') == [1, 2, 3, 4]
    assert rm.eval('returner(10)')() == 10
    with pytest.raises(KeyError):
        loads('a = {1: 2}[3]; b = lambda: a').eval('b()')


def test_contains():
    cf = loads('a = 1; b = 2')
    assert 'a' in cf and 'b' in cf
    assert 'c' not in cf


def test_literals(inside_tests):
    rm = load('expressions/literals.config')
    assert rm.literals == [
        True, False, None, ...,
        1, 2, 3, .4, 5j, .55e-2, 0x101, 0b101,
        'abc', r'def', b'ghi', u'jkl', rb'mno',
        'string interpolation: (3, 1)', 'a: 3 b: 7',
        [], [1, 2, 1],
        (), (1, 2, 1), (1,),
        {1, 2, 1},
        {}, {1: 2, 3: 4, '5': 6}
    ]


def test_operators(inside_tests):
    rm = load('expressions/operators.config')
    assert rm.arithmetic == [
        5, 6, 0.75, 0, 2, 6, -3, -5, 5, True, 63, 36, 27, 5, 3,
        True, False, False, False, True, True, False, True, False, True,
        True, True, True
    ]
    assert rm.priority == 1 + 2 * 3 ** 4 + 1


def test_comprehensions(inside_tests):
    rm = load('expressions/comprehensions.config')
    assert rm.everything == [
        list(range(10)), [0, 6],
        [1, 2, 3], [1, 3], [1, 3]
    ]
    assert rm.set_comp == {i for i in range(10)}
    assert rm.dict_comp == {i: i + 1 for i in range(10)}
    assert list(rm.gen_comp) == list(range(10))
    assert rm.even == list(range(0, 10, 2))

    with pytest.raises(SemanticError):
        loads('v = [x for i in range(1)]')

    with pytest.raises(SemanticError):
        loads('v = [[x, i] for i in range(1)]')

    with pytest.raises(SemanticError):
        loads('v = [i for i in [[2]] if x != 2 for x in i]')


def test_if(inside_tests):
    rm = load('expressions/if_.config')
    assert rm.results == [1, 1, 1]


def test_slice(inside_tests):
    rm = load('expressions/slices.config')
    assert rm.correct_slices == rm.slices


def test_build_config(inside_tests):
    rm = load('imports/config_import.config', shortcuts={'expressions': 'expressions'})
    with open('imports/built.config') as built:
        assert rm.dumps() == built.read()


def test_bad_shortcut():
    with pytest.raises(ImportError):
        loads('from a.b import *')


def test_cached(inside_tests):
    rm = load('imports/cached/main.config')
    assert rm.x == 1


def test_overwrite(inside_tests):
    rm = load('expressions/literals.config').string_input('literals = 1')
    rm.literals
    with pytest.raises(RuntimeError):
        rm.file_input('expressions/literals.config')

    rm = load('expressions/literals.config').string_input('a = 2')
    rm.string_input('b = a + 1')
    rm.literals


def test_injections():
    with pytest.raises(SemanticError):
        Config(injections={'print': int})

    assert loads('a = 1 + b', injections={'b': 14}).a == 15


def test_statements(inside_tests):
    cf = load('statements/func_statements.config')
    assert cf.f(1) == 1
    assert cf.clip(5, 0, 10) == 5
    assert cf.clip(-10, 0, 10) == 0
    assert cf.clip(100, 0, 10) == 10
    assert cf.define_in_if(-10) == 10
    assert cf.define_in_if(10) == 10


def test_update_render():
    values = (
        (1, '1'),
        ('1', "'1'"),
        ([1, '2', b'3'], "[1, '2', b'3']"),
        ({1: 2, '3': [4]}, "{1: 2, '3': [4]}"),
        ((1, 2), '(1, 2)'),
        ({1, 2}, '{1, 2}'),
    )

    for raw, s in values:
        cf = Config()
        cf.update(x=raw)
        assert cf.dumps().strip() == 'x = ' + s

    with pytest.raises(ValueError):
        Config().update(x=object()).dumps()


def test_context_manager(inside_tests):
    @contextlib.contextmanager
    def returner(x):
        yield x

    cf = load('statements/context.config')
    assert cf.inverse(2) == 1 / 2
    assert cf.inverse(0) == 0
    assert cf.invoke_with(returner(1)) == 1
    assert cf.invoke_with_unpack(returner((1, 2, 3))) == 6
    assert cf.context_result(returner((1, 2))) == 4


def test_cycles():
    with pytest.raises(SemanticError):
        loads('''
a = a
x = {"y": y}
y = {
    'good': t,
    'list': [1, 2, z]
}
z = {'nested': x}
t = 1
''')


def test_undefined():
    with pytest.raises(SemanticError):
        loads('''
with_undefined = {
'param': dunno
}
x = another_undefined
''')


def test_duplicates():
    with pytest.raises(SemanticError):
        loads('''
a = 1
b = 2
a = 11
''')


def test_exception_type():
    with pytest.raises(KeyError):
        loads('''
a = {'0': 1}[0]
b = a
''').b
    with pytest.raises(KeyError):
        loads('''
a = {'0': 1}[0]
def b(x=a):
    return x
''').b


def test_setattr():
    with pytest.raises(AttributeError):
        rm = Config()
        rm.a = 1


def test_get():
    cf = Config()
    assert cf.get('a', 1) == 1
    with pytest.raises(EntryError):
        cf.get('a')


def test_unused():
    # unused names are ok now
    loads('''
def f():
    x = 1
    return 2
''')


def test_read_only():
    with pytest.raises(SemanticError):
        loads('''
__file__ = 1
''')
    with pytest.raises(SemanticError):
        loads('''
__abc__ = 1
''')
