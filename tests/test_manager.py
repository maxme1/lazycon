import unittest

import numpy as np

from resource_manager.exceptions import BuildConfigError, RenderError, LambdaArgumentsError, BadSyntaxError
from resource_manager.manager import ResourceManager, read_config


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
        self.assertEqual(1, rm.one)

    def test_cycle_import(self):
        try:
            read_config('imports/cycle_import.config')
        except RecursionError:
            self.fail()

    def test_inheritance_order(self):
        rm = read_config('imports/order1.config')
        self.assertEqual(rm.one, 1)
        self.assertEqual(rm.two, '2')
        rm = read_config('imports/order2.config')
        self.assertIsNone(rm.one)
        self.assertIsNone(rm.two)

    def test_import_in_string(self):
        rm = ResourceManager()
        rm.string_input('from .expressions.types import *')
        self.assertEqual(1, rm.one)

    def test_upper_import(self):
        rm = read_config('imports/folder/upper_import.config')
        self.assertEqual('just override os', rm.os)
        self.assertEqual(np, rm.numpy)

    def test_attr_error(self):
        rm = read_config('imports/imports.config')
        with self.assertRaises(AttributeError):
            rm.undefined_value

    def test_items(self):
        rm = read_config('expressions/tail.config')
        self.assertEqual(rm.part, 6)
        self.assertTupleEqual((1, 2), rm.value)
        np.testing.assert_array_equal(rm.another_part, [1, 2, 3])

    def test_functions(self):
        rm = read_config('expressions/tail.config')
        np.testing.assert_array_equal(rm.mean, [2, 5])
        np.testing.assert_array_equal(rm.mean2, [2, 5])
        np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
        self.assertEqual(rm.random.shape, (1, 1, 2, 2))

    def test_bindings_clash(self):
        with self.assertRaises(BuildConfigError):
            ResourceManager().string_input('''
            def f(x):
                x = 1
                return 2
            ''')

    def test_func_def(self):
        rm = read_config('statements/funcdef.config')
        self.assertEqual(1, rm.f())
        self.assertTupleEqual((2, 1, 1), rm.inc_first(['a', 'b', 'c']))
        self.assertListEqual([1, 2, 3, 10, 10], rm.qsort([2, 1, 3, 10, 10]))

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

    def test_types(self):
        rm = read_config('expressions/types.config')
        try:
            rm.one
            rm.two
            rm.three
            rm.four
            rm.five
            rm.six
        except BaseException:
            self.fail()
        self.assertListEqual(list(range(1, 7)), rm.starred_array)
        self.assertTupleEqual(tuple(range(1, 7)), rm.starred_tuple)
        self.assertSetEqual({1, 2}, rm.set_one)
        self.assertSetEqual({1, 2, 3}, rm.starred_set)

        rm = read_config('expressions/other_cases.config')
        self.assertEqual(1, rm.parenthesis)
        self.assertEqual(tuple(), rm.empty_tuple)

    def test_operators(self):
        rm = read_config('expressions/operators.config')
        self.assertListEqual(rm.arithmetic, [
            5, 6, 0.75, 0, 2, 6, -3, -5, 5, True, 63, 36, 27, 5, 3,
            True, False, False, False, True, True, False, True, False, True
        ])
        self.assertEqual(1 + 2 * 3 ** 4 + 1, rm.priority)

    def test_if(self):
        rm = read_config('expressions/if_.config')
        self.assertListEqual([1, 1, 1], rm.results)

    def test_slice(self):
        rm = read_config('expressions/slices.config')
        self.assertTupleEqual(rm.correct_slices, rm.slices)

    def test_build_config(self):
        rm = read_config('imports/config_import.config', shortcuts={'expressions': 'expressions'})
        with open('imports/built.config') as built:
            self.assertEqual(rm.render_config(), built.read())

    def test_cached(self):
        rm = read_config('imports/cached/main.config')
        self.assertEqual(1, rm.x)

    def test_cycles(self):
        with self.assertRaises(BuildConfigError):
            read_config('misc/cycles.config')

    def test_undefined(self):
        with self.assertRaises(BuildConfigError):
            read_config('misc/static_undefined.config')

    def test_duplicates(self):
        with self.assertRaises(BadSyntaxError):
            read_config('misc/duplicate.config')

    def test_exc_handling(self):
        rm = ResourceManager().string_input('''
        a = sum(1)
        ''')
        with self.assertRaises(RenderError):
            rm.a
