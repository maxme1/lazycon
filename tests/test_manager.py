import unittest

import numpy as np

from resource_manager.resource_manager import ResourceManager, read_config


class TestResourceManager(unittest.TestCase):
    def test_string_input(self):
        rm = ResourceManager()
        rm.string_input('''
        from builtins import sum
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

    def test_attr_error(self):
        rm = read_config('imports/imports.config')
        with self.assertRaises(AttributeError):
            rm.undefined_value

    def test_items(self):
        rm = read_config('expressions/tail.config')
        self.assertEqual(rm.part, 6)
        np.testing.assert_array_equal(rm.another_part, [1, 2, 3])

    def test_functions(self):
        rm = read_config('expressions/tail.config')
        np.testing.assert_array_equal(rm.mean, [2, 5])
        np.testing.assert_array_equal(rm.mean2, [2, 5])
        np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
        self.assertEqual(rm.random.shape, (1, 1, 2, 2))

    def test_lambda(self):
        rm = read_config('expressions/lambda_.config')
        self.assertEqual(8, rm.b(2))
        self.assertEqual(8, rm.c(2, rm.a))
        self.assertListEqual([1, 2], rm.d(1)(2))
        self.assertEqual(8, rm.e())
        self.assertListEqual([1, 8, 32], rm.test)

    def test_lambda_args(self):
        rm = read_config('expressions/lambda_.config')
        with self.assertRaises(ValueError):
            rm.b(1, 2)

    def test_types(self):
        rm = read_config('expressions/types.config')
        try:
            rm.one
            rm.two
            rm.three
            rm.four
        except BaseException:
            self.fail()

    def test_build_config(self):
        rm = read_config('imports/config_import.config', shortcuts={'expressions': 'expressions'})
        with open('imports/built.config') as built:
            self.assertEqual(rm.render_config(), built.read())

    def test_cached(self):
        rm = read_config('imports/cached/main.config')
        self.assertEqual(1, rm.x)

    def test_cycles(self):
        with self.assertRaises(RuntimeError):
            read_config('misc/cycles.config')

    def test_undefined(self):
        with self.assertRaises(RuntimeError):
            read_config('misc/static_undefined.config')

    def test_duplicates(self):
        with self.assertRaises(SyntaxError):
            read_config('misc/duplicate.config')

    def test_exc_handling(self):
        rm = ResourceManager()
        rm.string_input('''
        from builtins import sum
        a = sum(1)
        ''')
        with self.assertRaises(RuntimeError):
            rm.a
