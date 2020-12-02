import numpy as np
import pytest

from resource_manager.exceptions import ResourceError, SemanticError
from resource_manager import ResourceManager, read_config, read_string


def test_string_input():
    rm = ResourceManager()
    rm.string_input('''
a = [1,2,3]
b = sum(a)
''')
    assert rm.b == 6


def test_entry_points(subtests, tests_path):
    for path in tests_path.glob('**/*.config'):
        with subtests.test(filename=path.name):
            config = read_config(path)
            # for each entry point the config must be readable and give the same results
            for name in config._scope.keys():
                source = config.render_config(name)
                assert config.render_config(name) == source
                assert read_string(source).render_config() == source, path


def test_import():
    rm = read_config('imports/imports.config')
    assert rm.numpy == np
    try:
        rm.r
        rm.os
        rm.std
        rm.mean
    except BaseException:
        pytest.fail()


def test_import_partial():
    rm = read_config('imports/import_from_config.config')
    assert rm.one == [1, 2]
    rm = read_config('imports/import_twice.config')
    assert rm.link_ == rm.link
    assert rm.func_ == rm.func


def test_update():
    rm = read_string('a = 1').update(a=2)
    rm2 = read_string('a = 1').string_input('a = 2')
    assert rm.a == rm2.a

    rm = read_string('a = 1').update(a=2).string_input('a = 3')
    assert rm.a == 3

    with pytest.raises(RuntimeError):
        read_string('a = 1').update(a=2).render_config()

    with pytest.raises(RuntimeError):
        rm = read_string('a = 1')
        rm.a
        rm.update(a=2)


def test_multiple_definitions():
    rm = read_config('statements/multiple_definitions.config')
    assert '''a = b = c = 1\nd = a\n''' == rm.render_config()
    rm = read_config('statements/import_from_multiple.config')
    assert '''a = 2\nb = c = 1\nd = a\n''' == rm.render_config()


def test_import_partial_upper():
    rm = read_config('imports/folder/upper_partial.config')
    assert rm.numpy is None
    assert np == rm.np
    rm = read_config('imports/folder/child/third.config')
    assert rm.a == [1, 2, 1]


def test_cycle_import():
    try:
        read_config('imports/cycle_import.config')
    except RecursionError:
        pytest.fail()


def test_inheritance_order():
    rm = read_config('imports/order1.config')
    assert rm.literals is not None
    rm = read_config('imports/order2.config')
    assert rm.literals is None


def test_import_in_string():
    rm = read_string('from .expressions.literals import *')
    assert rm.literals[0]


def test_upper_import():
    rm = read_config('imports/folder/upper_import.config')
    assert 'just override os' == rm.os
    assert np == rm.numpy


def test_attr_error():
    rm = read_config('imports/imports.config')
    with pytest.raises(AttributeError):
        rm.undefined_value
    with pytest.raises(KeyError):
        rm['undefined_value']
    with pytest.raises(ResourceError):
        rm.get_resource('undefined_value')


def test_items():
    rm = read_config('expressions/tail.config')
    assert rm.part == 6
    assert rm.value == (1, 2)
    np.testing.assert_array_equal(rm.another_part, [1, 2, 3])


def test_tail():
    rm = read_config('expressions/tail.config')
    np.testing.assert_array_equal(rm.mean, [2, 5])
    np.testing.assert_array_equal(rm.mean2, [2, 5])
    np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
    assert rm.random.shape == (1, 1, 2, 2)


def test_bindings_clash():
    with pytest.raises(SemanticError):
        ResourceManager().string_input('''
def f(x):
    x = 1
    return 2
''')


def test_bindings_names():
    with pytest.raises(SemanticError):
        ResourceManager().string_input('''
def f(x):
    x = 1
    y = 2
    x = 3
    return 2
''')


def test_func_def():
    rm = read_config('statements/funcdef.config')
    assert rm.f() == 1
    assert rm.inc_first(['a', 'b', 'c']) == (2, 1, 1)
    assert rm.qsort([2, 1, 3, 10, 10]) == [1, 2, 3, 10, 10]
    assert rm.returner(2)() == 2
    assert rm.h_with_defaults(0, n=3) == 1
    assert rm.doc.__doc__ == 'docstring'


def test_assertions():
    rm = read_config('statements/funcdef.config')
    assert rm.assertion(True)
    with pytest.raises(AssertionError):
        rm.assertion(False)


def test_unpacking():
    rm = read_config('statements/funcdef.config')
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

    with pytest.raises(SyntaxError):
        read_string('a, b = 1, 2')
    with pytest.raises(SyntaxError):
        read_string('def f(x): a, *b = x; return a')


def test_decorators():
    rm = read_config('statements/funcdef.config')
    assert rm.one(0) == 1
    assert rm.two(0) == 2
    assert rm.three(0) == 3
    assert rm.order() == tuple(range(5))


def test_render_decorators():
    rm = read_config('statements/decorated.config')
    with open('statements/decorated.config') as file:
        assert rm.render_config() == file.read()


def test_lambda():
    rm = read_config('expressions/lambda_.config')
    assert rm.b(2) == 8
    assert rm.c(2, rm.a) == 8
    assert rm.d(1)(2) == [1, 2]
    assert rm.e() == 8
    assert rm.test == [1, 8, 32]
    assert rm.vararg(1, 2, 3) == (2, 3)
    assert rm.only_vararg(1) == (1,)


def test_lambda_args():
    rm = read_config('expressions/lambda_.config')
    assert rm.with_default() == (1, 2)
    assert rm.keyword(y=1) == ((), 1)
    with pytest.raises(TypeError):
        rm.b(1, 2)
    try:
        rm.vararg(x=1)
    except BaseException:
        pytest.fail()


def test_eval():
    rm = read_config('statements/funcdef.config')
    assert rm.eval('f') == rm.f
    assert rm.eval('f()') == 1
    assert rm.eval('qsort([4,2,1,3])') == [1, 2, 3, 4]
    assert rm.eval('returner(10)')() == 10


def test_literals():
    rm = read_config('expressions/literals.config')
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


def test_operators():
    rm = read_config('expressions/operators.config')
    assert rm.arithmetic == [
        5, 6, 0.75, 0, 2, 6, -3, -5, 5, True, 63, 36, 27, 5, 3,
        True, False, False, False, True, True, False, True, False, True,
        True, True, True
    ]
    assert rm.priority == 1 + 2 * 3 ** 4 + 1


def test_comprehensions():
    rm = read_config('expressions/comprehensions.config')
    assert rm.everything == [
        list(range(10)), [0, 6],
        [1, 2, 3], [1, 3], [1, 3]
    ]
    assert rm.set_comp == {i for i in range(10)}
    assert rm.dict_comp == {i: i + 1 for i in range(10)}
    assert list(rm.gen_comp) == list(range(10))
    assert rm.even == list(range(0, 10, 2))

    with pytest.raises(SemanticError):
        read_string('_ = [x for i in range(1)]')

    with pytest.raises(SemanticError):
        read_string('_ = [[x, i] for i in range(1)]')

    with pytest.raises(SemanticError):
        read_string('_ = [i for i in [[2]] if x != 2 for x in i]')


def test_if():
    rm = read_config('expressions/if_.config')
    assert rm.results == [1, 1, 1]


def test_slice():
    rm = read_config('expressions/slices.config')
    assert rm.correct_slices == rm.slices


def test_build_config():
    rm = read_config('imports/config_import.config', shortcuts={'expressions': 'expressions'})
    with open('imports/built.config') as built:
        assert rm.render_config() == built.read()


def test_bad_shortcut():
    with pytest.raises(ImportError):
        read_string('from a.b import *')


def test_cached():
    rm = read_config('imports/cached/main.config')
    assert rm.x == 1


def test_overwrite():
    rm = read_config('expressions/literals.config').string_input('literals = 1')
    rm.literals
    with pytest.raises(RuntimeError):
        rm.import_config('expressions/literals.config')

    rm = read_config('expressions/literals.config').string_input('a = 2')
    rm.string_input('b = a + 1')
    rm.literals


def test_injections():
    with pytest.raises(SemanticError):
        ResourceManager(injections={'print': int})

    assert read_string('a = 1 + b', injections={'b': 14}).a == 15


def test_cycles():
    with pytest.raises(SemanticError):
        read_string('''
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
        read_string('''
with_undefined = {
'param': dunno
}
x = another_undefined
''')


def test_duplicates():
    with pytest.raises(SemanticError):
        read_string('''
a = 1
b = 2
a = 11
''')


def test_bad_import():
    with pytest.raises(NameError):
        read_string('from .expressions.literals import a')


def test_exception_type():
    with pytest.raises(KeyError):
        read_string('''
a = {'0': 1}[0]
b = a
''').b


def test_setattr():
    with pytest.raises(AttributeError):
        rm = ResourceManager()
        rm.a = 1


def test_unused():
    with pytest.raises(SemanticError):
        read_string('''
def f():
    x = 1
    return 2
''')


def test_read_only():
    with pytest.raises(SemanticError):
        read_string('''
__file__ = 1
''')
