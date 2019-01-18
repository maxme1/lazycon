import unittest

import numpy as np

from resource_manager.exceptions import ResourceError, SemanticError
from resource_manager import ResourceManager, read_config, read_string


class TestResourceManager(unittest.TestCase):
    def test_string_input(self):
        rm = ResourceManager()
        rm.string_input('''
a = [1,2,3]
b = sum(a)
''')
        self.assertEqual(6, rm.b)

    def test_import(self):
        rm = read_config('imports/imports.config')
        self.assertEqual(rm.numpy, np)
        try:
            rm.r
            rm.os
            rm.std
            rm.mean
        except BaseException:
            self.fail()

    def test_import_partial(self):
        rm = read_config('imports/import_from_config.config')
        self.assertListEqual([1, 2], rm.one)
        rm = read_config('imports/import_twice.config')
        self.assertEqual(rm.link_, rm.link)
        self.assertEqual(rm.func_, rm.func)

    def test_multiple_definitions(self):
        rm = read_config('statements/multiple_definitions.config')
        self.assertEqual('''a = b = c = 1\n\nd = a\n''', rm.render_config())
        rm = read_config('statements/import_from_multiple.config')
        self.assertEqual('''a = 2\n\nb = c = 1\n\nd = a\n''', rm.render_config())

    def test_import_partial_upper(self):
        rm = read_config('imports/folder/upper_partial.config')
        self.assertIsNone(rm.numpy)
        self.assertEqual(np, rm.np)
        rm = read_config('imports/folder/child/third.config')
        self.assertListEqual([1, 2, 1], rm.a)

    def test_cycle_import(self):
        try:
            read_config('imports/cycle_import.config')
        except RecursionError:
            self.fail()

    def test_inheritance_order(self):
        rm = read_config('imports/order1.config')
        self.assertIsNotNone(rm.literals)
        rm = read_config('imports/order2.config')
        self.assertIsNone(rm.literals)

    def test_import_in_string(self):
        rm = read_string('from .expressions.literals import *')
        self.assertTrue(rm.literals[0])

    def test_upper_import(self):
        rm = read_config('imports/folder/upper_import.config')
        self.assertEqual('just override os', rm.os)
        self.assertEqual(np, rm.numpy)

    def test_attr_error(self):
        rm = read_config('imports/imports.config')
        with self.assertRaises(AttributeError):
            rm.undefined_value
        with self.assertRaises(KeyError):
            rm['undefined_value']
        with self.assertRaises(ResourceError):
            rm.get_resource('undefined_value')

    def test_items(self):
        rm = read_config('expressions/tail.config')
        self.assertEqual(rm.part, 6)
        self.assertTupleEqual((1, 2), rm.value)
        np.testing.assert_array_equal(rm.another_part, [1, 2, 3])

    def test_tail(self):
        rm = read_config('expressions/tail.config')
        np.testing.assert_array_equal(rm.mean, [2, 5])
        np.testing.assert_array_equal(rm.mean2, [2, 5])
        np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
        self.assertEqual(rm.random.shape, (1, 1, 2, 2))

    def test_bindings_clash(self):
        with self.assertRaises(SemanticError):
            ResourceManager().string_input('''
def f(x):
    x = 1
    return 2
''')

    def test_bindings_names(self):
        with self.assertRaises(SemanticError):
            ResourceManager().string_input('''
def f(x):
    x = 1
    y = 2
    x = 3
    return 2
''')

    def test_func_def(self):
        rm = read_config('statements/funcdef.config')
        self.assertEqual(1, rm.f())
        self.assertTupleEqual((2, 1, 1), rm.inc_first(['a', 'b', 'c']))
        self.assertListEqual([1, 2, 3, 10, 10], rm.qsort([2, 1, 3, 10, 10]))
        self.assertEqual(2, rm.returner(2)())
        self.assertEqual(1, rm.h_with_defaults(0, n=3))

    def test_decorators(self):
        rm = read_config('statements/funcdef.config')
        self.assertEqual(1, rm.one(0))
        self.assertEqual(2, rm.two(0))
        self.assertEqual(3, rm.three(0))

        self.assertEqual(tuple(range(5)), rm.order())

    def test_lambda(self):
        rm = read_config('expressions/lambda_.config')
        self.assertEqual(8, rm.b(2))
        self.assertEqual(8, rm.c(2, rm.a))
        self.assertListEqual([1, 2], rm.d(1)(2))
        self.assertEqual(8, rm.e())
        self.assertListEqual([1, 8, 32], rm.test)
        self.assertTupleEqual((2, 3), rm.vararg(1, 2, 3))
        self.assertTupleEqual((1,), rm.only_vararg(1))

    def test_lambda_args(self):
        rm = read_config('expressions/lambda_.config')
        self.assertTupleEqual((1, 2), rm.with_default())
        self.assertTupleEqual(((), 1), rm.keyword(y=1))
        with self.assertRaises(TypeError):
            rm.b(1, 2)
        try:
            rm.vararg(x=1)
        except BaseException:
            self.fail()

    def test_literals(self):
        rm = read_config('expressions/literals.config')
        self.assertListEqual([
            True, False, None, ...,
            1, 2, 3, .4, 5j, .55e-2, 0x101, 0b101,
            'abc', r'def', b'ghi', u'jkl', rb'mno',
            'string interpolation: (3, 1)', 'a: 3 b: 7',
            [], [1, 2, 1],
            (), (1, 2, 1), (1,),
            {1, 2, 1},
            {}, {1: 2, 3: 4, '5': 6}
        ], rm.literals)

    def test_operators(self):
        rm = read_config('expressions/operators.config')
        self.assertListEqual(rm.arithmetic, [
            5, 6, 0.75, 0, 2, 6, -3, -5, 5, True, 63, 36, 27, 5, 3,
            True, False, False, False, True, True, False, True, False, True,
            True, True, True
        ])
        self.assertEqual(1 + 2 * 3 ** 4 + 1, rm.priority)

    def test_comprehensions(self):
        rm = read_config('expressions/comprehensions.config')
        self.assertListEqual(rm.everything, [
            list(range(10)), [0, 6],
            [1, 2, 3], [1, 3], [1, 3]
        ])
        self.assertSetEqual({i for i in range(10)}, rm.set_comp)
        self.assertDictEqual({i: i + 1 for i in range(10)}, rm.dict_comp)
        self.assertListEqual(list(range(10)), list(rm.gen_comp))
        self.assertListEqual(list(range(0, 10, 2)), rm.even)

        with self.assertRaises(SemanticError):
            read_string('_ = [x for i in range(1)]')

        with self.assertRaises(SemanticError):
            read_string('_ = [[x, i] for i in range(1)]')

        with self.assertRaises(SemanticError):
            read_string('_ = [i for i in [[2]] if x != 2 for x in i]')

    def test_if(self):
        rm = read_config('expressions/if_.config')
        self.assertListEqual([1, 1, 1], rm.results)

    def test_slice(self):
        rm = read_config('expressions/slices.config')
        self.assertTupleEqual(rm.correct_slices, rm.slices)

    def test_build_config(self):
        rm = read_config('imports/config_import.config', shortcuts={'expressions': 'expressions'})
        with open('imports/built.config') as built:
            self.assertEqual(built.read(), rm.render_config())

    def test_bad_shortcut(self):
        with self.assertRaises(ImportError):
            read_string('from a.b import *')

    def test_cached(self):
        rm = read_config('imports/cached/main.config')
        self.assertEqual(1, rm.x)

    def test_overwrite(self):
        rm = read_config('expressions/literals.config').string_input('literals = 1')
        rm.literals
        with self.assertRaises(RuntimeError):
            rm.import_config('expressions/literals.config')

        rm = read_config('expressions/literals.config').string_input('a = 2')
        try:
            rm.string_input('b = a + 1')
        except:
            self.fail()

        try:
            rm.literals
        except:
            self.fail()

    def test_injections(self):
        with self.assertRaises(SemanticError):
            ResourceManager(injections={'print': int})

        self.assertEqual(15, read_string('a = 1 + b', injections={'b': 14}).a)

    def test_cycles(self):
        with self.assertRaises(SemanticError):
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

    def test_undefined(self):
        with self.assertRaises(SemanticError):
            read_string('''
with_undefined = {
    'param': dunno
}
x = another_undefined
''')

    def test_duplicates(self):
        with self.assertRaises(SemanticError):
            read_string('''
a = 1
b = 2
a = 11
''')

    def test_bad_import(self):
        with self.assertRaises(NameError):
            read_string('from .expressions.literals import a')

    def test_exception_type(self):
        with self.assertRaises(KeyError):
            read_string('''
a = {'0': 1}[0]
b = a
''').b

    def test_setattr(self):
        with self.assertRaises(AttributeError):
            rm = ResourceManager()
            rm.a = 1

    def test_unused(self):
        with self.assertRaises(SemanticError):
            read_string('''
def f():
    x = 1
    return 2
''')

    def test_read_only(self):
        with self.assertRaises(SemanticError):
            read_string('''
__file__ = 1
''')
